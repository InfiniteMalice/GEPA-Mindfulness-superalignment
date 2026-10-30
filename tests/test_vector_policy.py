"""CPU VPO preparation contracts; authored examples are not model experiments."""

import json
import math
import random
import unittest
from dataclasses import replace

from evaluation.causal_records import canonical_json
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)
from gepa_mindfulness.training.actionopd import DecisionKind, SemanticAction
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.trajectory import Trajectory
from gepa_mindfulness.training.vector_policy import (
    VectorCandidate,
    VectorPolicyConfig,
    VectorRollout,
    VectorScore,
    common_dirichlet_weights,
    prepare_vector_group,
    vector_set_score,
)
from gepa_mindfulness.verification.interfaces import (
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)

REVIEWER = TrustedEvaluatorContract("independent", "v1", "vector-evidence-v1")
REF = EvidenceReference("observed", EvidenceSourceKind.EXTERNAL_RECORD)
DIMS = ("verified_task_quality", "explanatory_clarity")
WEIGHTS = ((0.9, 0.1), (0.1, 0.9))


def config(**changes):
    values = dict(
        authority_domain="host",
        run_id="run",
        task_id="task",
        model_version="student-v1",
        harness_version="harness-v1",
        producer_id="student",
        evaluator=REVIEWER,
        dataset_digest="a" * 64,
        constitution_digest="b" * 64,
        dimensions=DIMS,
        weights=WEIGHTS,
        epsilon=1e-6,
    )
    values.update(changes)
    return VectorPolicyConfig(**values)


def candidate(identifier="c1", values=(1.0, 0.0), admissibility="admitted", **changes):
    names = (
        "task_fit",
        "dependencies_satisfied",
        "contradiction_status",
        "provenance_intact",
        "authorization_scope_valid",
        "claimed_outcome_supported",
    )
    review = RelationalVerificationResult(
        identifier,
        True,
        True,
        "none",
        True,
        True,
        True,
        False,
        (REF,),
        tuple(VerificationEvidenceBinding(name, (REF,)) for name in names),
    )
    fields = dict(
        candidate_id=identifier,
        action=SemanticAction(DecisionKind.GROUNDED_ANSWER, canonical_json({"answer": identifier})),
        scores=tuple(
            VectorScore(
                dimension,
                value,
                RewardProvenance(
                    dimension,
                    "independently reviewed task metric",
                    VerificationRoute.TRUSTED_EVALUATOR,
                    evaluator=REVIEWER,
                ),
            )
            for dimension, value in zip(DIMS, values)
        ),
        admissibility=admissibility,
        verification=review,
    )
    fields.update(changes)
    return VectorCandidate(**fields)


def rollout(identifier="rollout-1", candidates=None, **trajectory_changes):
    members = (candidate(),) if candidates is None else candidates
    fields = dict(
        trajectory_id=identifier,
        case_id=None,
        prompt="Original student prompt",
        response=canonical_json(
            [{"candidate_id": c.candidate_id, "action": c.action.to_dict()} for c in members]
        ),
        evidence_references=(REF,),
        model_identifier="student",
        policy_version="student-v1",
        sampling_parameters={"training_eligibility": "TRAIN"},
    )
    fields.update(trajectory_changes)
    return VectorRollout(identifier, RewardRequest(Trajectory(**fields), (REF,)), members)


def prepare(rollouts=None, **changes):
    fields = dict(
        config=config(),
        rollouts=(rollout(),) if rollouts is None else rollouts,
        authorize_evidence=lambda _: True,
        enabled=True,
    )
    fields.update(changes)
    return prepare_vector_group(**fields)


class VectorPolicyTests(unittest.TestCase):
    def test_shared_weight_maxima_match_hand_calculated_sets(self):
        self.assertEqual(vector_set_score(((1.0, 0.0), (0.0, 1.0)), WEIGHTS), 0.9)
        self.assertAlmostEqual(vector_set_score(((0.6, 0.6), (0.6, 0.6)), WEIGHTS), 0.6)
        self.assertEqual(vector_set_score(((1.0, 0.0),), ((0.5, 0.5),)), 0.5)
        self.assertEqual(vector_set_score(((-2.0, -4.0),), ((0.5, 0.5),)), -3.0)
        self.assertEqual(vector_set_score(((1.0, 0.0), (1.0, 0.0)), WEIGHTS), 0.5)

    def test_dirichlet_sampler_is_local_seeded_and_retains_simplex_rows(self):
        before = random.getstate()
        draws = common_dirichlet_weights(3, 32, seed=17)
        self.assertEqual(draws, common_dirichlet_weights(3, 32, seed=17))
        self.assertNotEqual(draws, common_dirichlet_weights(3, 32, seed=18))
        self.assertEqual(before, random.getstate())
        self.assertEqual(len(draws), 32)
        self.assertTrue(all(len(row) == 3 and min(row) >= 0 for row in draws))
        self.assertTrue(all(abs(math.fsum(row) - 1) < 1e-12 for row in draws))
        self.assertEqual(common_dirichlet_weights(1, 2, seed=1), ((1.0,), (1.0,)))

    def test_records_roundtrip_without_restoring_authority(self):
        for item in (config(), candidate(), candidate().scores[0], rollout()):
            wire = json.loads(json.dumps(item.to_dict()))
            self.assertEqual(type(item).from_dict(wire).to_dict(), wire)
            self.assertEqual(wire["training_eligibility"], "DEVELOPMENT")
            self.assertIs(wire["authority_granted"], False)
            for changes in ({"authority_granted": True}, {"training_eligibility": "TRAIN"}):
                with self.subTest(record=type(item).__name__, changes=changes):
                    with self.assertRaises(ValueError):
                        type(item).from_dict(wire | changes)
            with self.assertRaises(ValueError):
                type(item).from_dict(wire | {"unexpected": "authority"})
        with self.assertRaises(ValueError):
            require_training_eligible(prepare())

    def test_group_scores_valid_sets_and_retains_common_configuration(self):
        result = prepare(
            (
                rollout(candidates=(candidate(), candidate("c2", (0.0, 1.0)))),
                rollout("r2", (candidate("c3", (0.6, 0.6)),)),
            )
        )
        rows = result["rollouts"]
        self.assertEqual(rows[0]["reviewed_set_score"], 0.9)
        self.assertAlmostEqual(rows[1]["reviewed_set_score"], 0.6)
        self.assertGreater(rows[0]["advantage"], 0)
        self.assertLess(rows[1]["advantage"], 0)
        self.assertEqual(result["source"]["config"]["weights"], [[0.9, 0.1], [0.1, 0.9]])
        self.assertEqual(result["source"]["config"]["epsilon"], 1e-6)
        self.assertFalse(result["authority_granted"])
        self.assertFalse(result["fallback_required"])

    def test_invalid_shortcut_cannot_win_against_negative_valid_scores(self):
        for status in ("rejected", "unresolved"):
            for weights in (WEIGHTS, ((1.0, 0.0),), ((0.0, 1.0),)):
                with self.subTest(status=status, weights=weights):
                    bad = candidate("bad", (1e6, 1e6), status)
                    result = prepare(
                        (
                            rollout(candidates=(candidate(), bad)),
                            rollout("r2", (candidate("valid", (-2.0, -4.0)),)),
                        ),
                        config=config(weights=weights),
                    )
                    invalid, valid = result["rollouts"]
                    self.assertEqual(invalid["disposition"], "quarantined")
                    self.assertIsNone(invalid["reviewed_set_score"])
                    self.assertIsNone(invalid["advantage"])
                    self.assertLess(valid["reviewed_set_score"], 0)
                    self.assertEqual(valid["advantage"], 0)

    def test_empty_or_unresolved_sets_require_independent_fallback(self):
        for members in ((), (candidate(admissibility="unresolved"),)):
            result = prepare((rollout(candidates=members),))
            self.assertTrue(result["fallback_required"])
            self.assertEqual(result["decision_route"], "independent_host_fallback")
            self.assertIsNone(result["rollouts"][0]["reviewed_set_score"])

    def test_failed_or_mutating_review_cannot_release_scores(self):
        cfg = config()

        def mutate(data):
            data["config"]["constitution_digest"] = "c" * 64
            return True

        def mutate_caller(_):
            object.__setattr__(cfg.evaluator, "evaluator_version", "changed")
            return True

        for callback in (None, lambda _: False, lambda _: 1, mutate, mutate_caller):
            try:
                result = prepare(config=cfg, authorize_evidence=callback)
                self.assertFalse(result["evidence_review_accepted"])
                self.assertIsNone(result["rollouts"][0]["reviewed_set_score"])
                self.assertIsNone(result["rollouts"][0]["advantage"])
                self.assertTrue(result["fallback_required"])
            finally:
                object.__setattr__(cfg.evaluator, "evaluator_version", "v1")

    def test_disabled_adapter_does_not_inspect_inputs(self):
        self.assertIsNone(prepare_vector_group(object(), object()))
        with self.assertRaises(ValueError):
            prepare_vector_group(object(), object(), enabled="yes")

    def test_constitution_cannot_be_an_optimized_dimension(self):
        for dimension in ("non_deception", "autonomy", "constitution", "knowledge_gain"):
            with self.subTest(dimension=dimension):
                with self.assertRaises(ValueError):
                    config(dimensions=(dimension, "explanatory_clarity"))

    def test_malformed_weights_and_nonfinite_arithmetic_fail_before_review(self):
        for weights in (
            (),
            ((0.0, 0.0),),
            ((-1.0, 2.0),),
            ((0.8, 0.8),),
            ((True, 0.0),),
            ((float("nan"), 0.0),),
            ((1.0,),),
            ((1e308, 1e308),),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    config(weights=weights)
        with self.assertRaises(ValueError):
            vector_set_score(((1e308, 1e308),), WEIGHTS)
        with self.assertRaises(ValueError):
            prepare(
                (rollout(candidates=(candidate(values=(1e200, 1e200)),)), rollout("r2")),
                authorize_evidence=lambda _: self.fail("overflow reviewed"),
            )

    def test_unbound_sources_and_versions_reject_before_review(self):
        for changes in (
            {"response": "unbound generated answer"},
            {"policy_version": "different"},
            {"model_identifier": "different"},
            {"sampling_parameters": {"training_eligibility": "EVAL"}},
            {"sampling_parameters": {}},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    prepare((rollout(**changes),), authorize_evidence=lambda _: self.fail("review"))
        with self.assertRaises(ValueError):
            prepare((rollout(), rollout("r2", prompt="different public prompt")))

    def test_partial_verification_never_releases_whole_rollout_credit(self):
        for changes in (
            {"authorization_scope_valid": False},
            {"contradiction_status": "unknown"},
            {"claimed_outcome_supported": False},
            {"repeated_failed_route": True},
        ):
            with self.subTest(changes=changes):
                review = candidate().verification.to_dict() | changes
                if changes.get("contradiction_status") == "unknown":
                    review["evidence_bindings"] = [
                        binding
                        for binding in review["evidence_bindings"]
                        if binding["field_name"] != "contradiction_status"
                    ]
                if changes.get("repeated_failed_route"):
                    review["evidence_bindings"].append(
                        VerificationEvidenceBinding("repeated_failed_route", (REF,)).to_dict()
                    )
                result = prepare(
                    (
                        rollout(
                            candidates=(
                                replace(
                                    candidate(),
                                    verification=RelationalVerificationResult.from_dict(review),
                                ),
                            )
                        ),
                    )
                )
                self.assertIsNone(result["rollouts"][0]["reviewed_set_score"])

    def test_complete_output_budget_fails_before_host_review(self):
        sources = (rollout(prompt="x" * 2_200_000), rollout("r2", prompt="x" * 2_200_000))
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            prepare(sources, authorize_evidence=lambda _: self.fail("oversized output reviewed"))

    def test_host_reviews_proposed_metrics_before_they_are_released(self):
        def check(envelope):
            row = envelope["proposed_metrics"][0]
            self.assertEqual(row["reviewed_set_score"], 0.5)
            self.assertEqual(row["advantage"], 0)
            self.assertFalse(envelope["authority_granted"])
            return True

        self.assertTrue(prepare(authorize_evidence=check)["evidence_review_accepted"])

    def test_private_and_conflicting_source_references_reject_before_review(self):
        private = EvidenceReference("observed", EvidenceSourceKind.PRIVATE_REASONING)
        source = rollout(evidence_references=(REF, private))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            prepare(
                (source,), authorize_evidence=lambda _: self.fail("conflicting source reviewed")
            )

    def test_self_certified_and_mismatched_score_evaluators_reject(self):
        with self.assertRaises(ValueError):
            config(producer_id="independent")
        score = replace(
            candidate().scores[0],
            provenance=RewardProvenance(
                DIMS[0],
                "untrusted self label",
                VerificationRoute.TRUSTED_EVALUATOR,
                evaluator=TrustedEvaluatorContract("student", "v1", "self"),
            ),
        )
        source = rollout(candidates=(replace(candidate(), scores=(score, candidate().scores[1])),))
        with self.assertRaisesRegex(ValueError, "evaluator"):
            prepare((source,), authorize_evidence=lambda _: self.fail("self certified"))

    def test_corrupted_provenance_is_revalidated_before_serialization(self):
        score = candidate().scores[0]
        object.__setattr__(score.provenance, "route", "trusted_evaluator")
        with self.assertRaises(ValueError):
            score.to_dict()

    def test_score_evidence_cannot_reference_an_unobserved_source(self):
        outside = EvidenceReference("outside", EvidenceSourceKind.EXTERNAL_RECORD)
        score = replace(
            candidate().scores[0],
            provenance=RewardProvenance(
                DIMS[0],
                "observable independent metric",
                VerificationRoute.OBSERVABLE_EVIDENCE,
                (outside,),
            ),
        )
        source = rollout(candidates=(replace(candidate(), scores=(score, candidate().scores[1])),))
        with self.assertRaisesRegex(ValueError, "outside observable source"):
            prepare((source,), authorize_evidence=lambda _: self.fail("outside evidence"))


if __name__ == "__main__":
    unittest.main()
