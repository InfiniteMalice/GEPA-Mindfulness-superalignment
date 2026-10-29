"""CPU contracts for reviewed knowledge and potential proposals, never reward authority."""

# Standard library
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.causal_reward_records import request_digest
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.knowledge_progress import (
    KnowledgeCondition,
    LearningContext,
    LearningStep,
    prepare_knowledge_progress,
)
from gepa_mindfulness.training.research_credit import ResearchCreditClaim
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from gepa_mindfulness.training.trajectory import Trajectory
from gepa_mindfulness.verification.state import EvidenceClaim, EvidenceState

REVIEWER = TrustedEvaluatorContract("independent", "v1", "knowledge-evidence-v1")
REF = EvidenceReference("source-1", EvidenceSourceKind.EXTERNAL_RECORD)


def context(**changes):
    """Declare the host-owned run and explicit discount, without authenticating either."""
    values = dict(
        authority_domain="host",
        run_id="run",
        task_id="task",
        task_version="v1",
        model_version="student-v1",
        harness_version="harness-v1",
        producer_id="student",
        evaluator=REVIEWER,
        dataset_digest="a" * 64,
        gamma=0.5,
    )
    values.update(changes)
    return LearningContext(**values)


def conditions():
    """One atomic fact and two dependent, independently checkable milestones."""
    return (
        KnowledgeCondition("fact", "task", "v1", "The switch is on.", "knowledge", REVIEWER),
        KnowledgeCondition(
            "checked",
            "task",
            "v1",
            "The switch was independently checked.",
            "milestone",
            REVIEWER,
            ("fact",),
        ),
        KnowledgeCondition(
            "done",
            "task",
            "v1",
            "The authorized operation completed.",
            "milestone",
            REVIEWER,
            ("checked",),
        ),
    )


def state(statuses=("unverified", "unverified", "unverified"), ref=REF):
    """Existing evidence states remain distinct from the model's declared belief state."""
    return EvidenceState(
        tuple(
            EvidenceClaim(
                spec.condition_id,
                spec.proposition,
                (ref,) if status != "unverified" else (),
                status,
            )
            for spec, status in zip(conditions(), statuses)
        )
    )


def step(index=0, *, episode="episode-1", before=None, after=None, **changes):
    """Construct one observable action target with explicit training-only source lineage."""
    before = before or state()
    after = after or state(("supported", "unverified", "unverified"))
    refs = tuple(
        dict.fromkeys(r for s in (before, after) for c in s.claims for r in c.evidence_refs)
    )
    target = RewardRequest(
        Trajectory(
            f"action-{episode}-{index}",
            None,
            "Public prefix",
            "Public tool call",
            evidence_references=refs,
            model_identifier="student",
            policy_version="student-v1",
            sampling_parameters={"training_eligibility": "TRAIN"},
        ),
        refs,
    )
    values = dict(
        episode_id=episode,
        step_index=index,
        event_id=f"event-{episode}-{index}",
        request=target,
        before=before,
        after=after,
        termination="continuing",
        is_answer=False,
        belief_before=None,
        belief_after=None,
    )
    values.update(changes)
    return LearningStep(**values)


def prepare(steps=None, **changes):
    """The callback stands for explicitly configured host review, never model certification."""
    values = dict(
        context=context(),
        conditions=conditions(),
        steps=steps or (step(),),
        authorize_evidence=lambda _: True,
        enabled=True,
        epsilon=1e-6,
    )
    values.update(changes)
    return prepare_knowledge_progress(**values)


class KnowledgeProgressTests(unittest.TestCase):
    """Acquisition, progress and authority have independently observable outcomes."""

    def test_existing_ledger_requires_admission_and_only_reserves_one_event_component(self):
        source = step(after=state(("supported", "supported", "unverified")))
        result = prepare((source,))
        claims = tuple(ResearchCreditClaim.from_dict(value) for value in result["claims"])

        def use(selected, identifier):
            return ResearchCreditUse(
                identifier,
                "host-policy-v1",
                request_digest(source.request),
                tuple(c.claim_id for c in selected),
                tuple(
                    (
                        c.claim_id,
                        (
                            ("provider", "not_counted"),
                            ("reward_integrity", "not_counted"),
                            ("causal_reward", "not_counted"),
                        ),
                    )
                    for c in selected
                ),
            )

        with tempfile.TemporaryDirectory() as folder:
            ledger = ResearchCreditLedger(Path(folder) / "events.sqlite", authority_domain="host")
            with self.assertRaisesRegex(ValueError, "unresolved"):
                ledger.reserve(
                    source.request,
                    claims,
                    use(claims[:1], "unresolved"),
                    authorize_use=lambda _: True,
                    enabled=True,
                )
            # Only the external host may resolve normative admission after evidence review.
            admitted = tuple(replace(c, admissibility="admitted") for c in claims)
            with self.assertRaisesRegex(ValueError, "one semantic event"):
                ledger.reserve(
                    source.request,
                    admitted,
                    use(admitted, "both"),
                    authorize_use=lambda _: True,
                    enabled=True,
                )
            receipt = ledger.reserve(
                source.request,
                admitted,
                use(admitted[:1], "one"),
                authorize_use=lambda _: True,
                enabled=True,
            )
            self.assertFalse(receipt["authority_granted"])
            with self.assertRaisesRegex(ValueError, "already reserved"):
                ledger.reserve(
                    source.request,
                    admitted,
                    use(admitted[1:], "again"),
                    authorize_use=lambda _: True,
                    enabled=True,
                )

    def test_complete_output_budget_fails_before_review(self):
        original = step()
        target = replace(
            original.request,
            trajectory=replace(original.request.trajectory, prompt="x" * 1_500_000),
        )
        bounded = replace(original, request=target)
        # The request alone fits; the retained source plus complete claim/output does not.
        self.assertIsInstance(LearningStep.from_dict(bounded.to_dict()), LearningStep)
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            prepare((bounded,), authorize_evidence=lambda _: self.fail("oversized output reviewed"))

    def test_interleaved_episodes_keep_independent_continuity(self):
        known = step().after
        result = prepare(
            (step(episode="a"), step(episode="b"), step(1, episode="a", before=known, after=known))
        )
        self.assertEqual([r["knowledge_gain"] for r in result["steps"]], [1, 1, 0])

    def test_conflicting_evidence_kinds_and_wrong_model_fail_before_review(self):
        other_kind = EvidenceReference("source-1", EvidenceSourceKind.OBSERVABLE_OUTPUT)
        first = step(episode="a")
        second = step(
            episode="b", after=state(("supported", "unverified", "unverified"), other_kind)
        )
        with self.assertRaisesRegex(ValueError, "conflicting evidence"):
            prepare((first, second), authorize_evidence=lambda _: self.fail("conflict reviewed"))
        wrong = replace(
            first.request,
            trajectory=replace(first.request.trajectory, policy_version="another-model"),
        )
        with self.assertRaisesRegex(ValueError, "model/version"):
            prepare(
                (replace(first, request=wrong),),
                authorize_evidence=lambda _: self.fail("wrong model reviewed"),
            )

    def test_identity_serialization_does_not_expose_record_dictionaries(self):
        for record, field in ((context(), "evaluator"), (conditions()[0], "verifier")):
            with self.subTest(field=field):
                wire = record.to_dict()
                wire[field]["evaluator_id"] = "unreviewed"
                self.assertEqual(getattr(record, field).evaluator_id, "independent")

    def test_reviewer_closure_cannot_change_the_retained_catalog(self):
        specs = conditions()

        def mutate_original(_):
            vars(specs[0].verifier)["evaluator_id"] = "unreviewed"
            return True

        result = prepare(conditions=specs, authorize_evidence=mutate_original)
        self.assertTrue(result["evidence_review_accepted"])
        self.assertEqual(
            result["source"]["conditions"][0]["verifier"]["evaluator_id"], "independent"
        )
        self.assertEqual(content_digest(result["source"]["conditions"]), result["catalog_digest"])

    def test_conflicting_unselected_trajectory_reference_fails_before_review(self):
        original = step()
        extra = EvidenceReference("source-1", EvidenceSourceKind.PRIVATE_REASONING)
        target = replace(
            original.request,
            trajectory=replace(
                original.request.trajectory,
                evidence_references=original.request.trajectory.evidence_references + (extra,),
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicting evidence"):
            prepare(
                (replace(original, request=target),),
                authorize_evidence=lambda _: self.fail("unselected conflict reviewed"),
            )

    def test_every_action_identity_is_bounded_before_grouping(self):
        no_gain = step(episode="no-gain", after=state())
        for identifier in ("", "x" * 257):
            target = replace(
                no_gain.request,
                trajectory=replace(no_gain.request.trajectory, trajectory_id=identifier),
            )
            with self.subTest(identifier_length=len(identifier)):
                with self.assertRaisesRegex(ValueError, "action_id"):
                    prepare(
                        (step(), replace(no_gain, request=target)),
                        authorize_evidence=lambda _: self.fail("unbound action reviewed"),
                    )

    def test_nondefault_normalization_is_retained_in_reviewed_source_identity(self):
        sources = (step(episode="gain"), step(episode="no-gain", after=state()))
        default = prepare(sources)
        changed = prepare(sources, epsilon=1)
        self.assertEqual(changed["source"]["normalization_epsilon"], 1)
        self.assertEqual(default["source"]["normalization_epsilon"], 1e-6)
        self.assertNotEqual(default["source_digest"], changed["source_digest"])
        self.assertAlmostEqual(changed["steps"][0]["knowledge_advantage"], 1 / 3)

    def test_default_off_does_not_inspect_inputs(self):
        self.assertIsNone(prepare_knowledge_progress(object(), object(), object()))
        with self.assertRaises(ValueError):
            prepare_knowledge_progress(object(), object(), object(), enabled="yes")

    def test_new_knowledge_and_progress_share_event_without_combining_credit(self):
        result = prepare((step(after=state(("supported", "supported", "unverified"))),))
        row = result["steps"][0]
        self.assertEqual(row["knowledge_gain"], 1)
        self.assertEqual(row["milestone_gain"], 1)
        self.assertEqual(row["phi_before"], 0)
        self.assertEqual(row["phi_after"], 0.5)
        self.assertEqual(row["shaping"], 0.25)
        claims = result["claims"]
        self.assertEqual({c["credit_kind"] for c in claims}, {"knowledge", "progress"})
        self.assertEqual(len(result["credit_audit"]["event_overlaps"]), 1)
        self.assertTrue(all(c["admissibility"] == "unresolved" for c in claims))
        self.assertFalse(result["authority_granted"])
        with self.assertRaises(ValueError):
            require_training_eligible({"training_eligibility": "TRAIN", "source": result})

    def test_duplicate_support_and_reacquisition_do_not_mint_acquisition(self):
        known = state(("supported", "unverified", "unverified"))
        lost = state(("contradicted", "unverified", "unverified"))
        result = prepare(
            (
                step(),
                step(1, before=known, after=known),
                step(2, before=known, after=lost),
                step(3, before=lost, after=known),
            )
        )
        self.assertEqual([r["knowledge_gain"] for r in result["steps"]], [1, 0, 0, 0])
        self.assertEqual(result["steps"][2]["revoked_condition_ids"], ["fact"])
        self.assertEqual(result["steps"][3]["reacquired_condition_ids"], ["fact"])

    def test_false_belief_correction_is_separate_from_verified_acquisition(self):
        negative = state(("contradicted", "unverified", "unverified"))
        result = prepare((step(belief_before=negative, belief_after=step().after),))
        self.assertEqual(result["steps"][0]["belief_correction_ids"], ["fact"])
        self.assertEqual(result["steps"][0]["knowledge_gain"], 1)
        result = prepare(
            (step(before=step().after, belief_before=negative, belief_after=step().after),)
        )
        self.assertEqual(result["steps"][0]["knowledge_gain"], 0)
        self.assertEqual(result["steps"][0]["belief_correction_ids"], ["fact"])

    def test_grouping_uses_prior_evidence_not_posterior_or_belief(self):
        known = state(("supported", "unverified", "unverified"))
        result = prepare(
            (
                step(episode="a"),
                step(episode="b", after=state()),
                step(episode="c", before=known, after=known),
                step(episode="d", belief_before=known),
            )
        )
        rows = result["steps"]
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(rows[0]["group_id"], rows[3]["group_id"])
        self.assertNotEqual(rows[0]["group_id"], rows[2]["group_id"])
        self.assertAlmostEqual(rows[0]["knowledge_advantage"], 0.7071052811897296)
        self.assertLess(rows[1]["knowledge_advantage"], 0)
        self.assertEqual(rows[2]["knowledge_advantage"], 0)

    def test_answer_and_termination_steps_have_no_group_advantage(self):
        for fields in (
            {"is_answer": True},
            {"termination": "terminated"},
            {"termination": "truncated"},
        ):
            with self.subTest(fields=fields):
                row = prepare((step(**fields),))["steps"][0]
                self.assertIsNone(row["group_id"])
                self.assertIsNone(row["knowledge_advantage"])

    def test_exact_zero_progress_does_not_create_a_claim(self):
        specs = tuple(
            KnowledgeCondition(
                f"milestone-{i}",
                "task",
                "v1",
                f"Milestone {i} completed.",
                "milestone",
                REVIEWER,
            )
            for i in range(5)
        )
        before = EvidenceState(
            tuple(
                EvidenceClaim(
                    c.condition_id,
                    c.proposition,
                    (REF,) if i < 3 else (),
                    "supported" if i < 3 else "unverified",
                )
                for i, c in enumerate(specs)
            )
        )
        after = EvidenceState(
            tuple(
                EvidenceClaim(
                    c.condition_id,
                    c.proposition,
                    (REF,) if i < 4 else (),
                    "supported" if i < 4 else "unverified",
                )
                for i, c in enumerate(specs)
            )
        )
        for gamma, expected in ((0.75, 0.0), (0.7500000000000001, 8.881784197001253e-17)):
            with self.subTest(gamma=gamma):
                result = prepare(
                    (step(before=before, after=after),),
                    conditions=specs,
                    context=context(gamma=gamma),
                )
                self.assertEqual(result["steps"][0]["shaping"], expected)
                if expected == 0:
                    self.assertEqual(result["claims"], [])
                    self.assertIsNone(result["credit_audit"])
                else:
                    self.assertEqual(len(result["claims"]), 1)
                    self.assertEqual(result["claims"][0]["score"], expected)

    def test_nonzero_progress_survives_float_underflow(self):
        result = prepare(
            (
                step(
                    before=state(("supported", "unverified", "unverified")),
                    after=state(("supported", "supported", "unverified")),
                ),
                step(
                    episode="no-progress",
                    before=state(("supported", "unverified", "unverified")),
                    after=state(("supported", "unverified", "unverified")),
                ),
            ),
            context=context(gamma=5e-324),
        )
        row = result["steps"][0]
        self.assertEqual((row["phi_before"], row["phi_after"]), (0.0, 0.5))
        self.assertEqual(row["shaping"], 0.0)
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["credit_kind"], "progress")
        self.assertEqual(result["claims"][0]["score"], 0.0)
        self.assertEqual(result["claims"][0]["event_id"], "event-episode-1-0")

    def test_terminal_zero_potential_telescopes_without_success_credit(self):
        mid = state(("supported", "supported", "unverified"))
        done = state(("supported", "supported", "supported"))
        result = prepare(
            (step(after=mid), step(1, before=mid, after=done, termination="terminated"))
        )
        rows = result["steps"]
        self.assertEqual([r["shaping"] for r in rows], [0.25, -0.5])
        self.assertEqual(rows[0]["shaping"] + 0.5 * rows[1]["shaping"], 0)
        self.assertEqual(rows[1]["phi_after"], 0)
        self.assertEqual(rows[1]["completed_milestone_ids"], ["checked", "done"])

    def test_truncation_is_not_a_terminal_shaping_bonus(self):
        result = prepare(
            (step(after=state(("supported", "supported", "unverified")), termination="truncated"),)
        )
        self.assertIsNone(result["steps"][0]["shaping"])
        self.assertNotIn("progress", {c["credit_kind"] for c in result["claims"]})

    def test_unmet_dependency_never_contributes_to_potential(self):
        row = prepare((step(after=state(("unverified", "supported", "supported"))),))["steps"][0]
        self.assertEqual(row["completed_milestone_ids"], [])
        self.assertEqual(row["blocked_milestone_ids"], ["checked", "done"])
        self.assertEqual(row["shaping"], 0)

    def test_failed_or_mutating_host_review_produces_no_claims(self):
        def mutate(envelope):
            envelope["steps"] = []
            return True

        def explode(_):
            raise RuntimeError("not evidence")

        for callback in (None, lambda _: False, lambda _: 1, mutate, explode):
            with self.subTest(callback=callback):
                result = prepare(authorize_evidence=callback)
                self.assertFalse(result["evidence_review_accepted"])
                self.assertEqual(result["claims"], [])

    def test_inconsistent_source_rejected_before_host_review(self):
        bad_ref = EvidenceReference("foreign", EvidenceSourceKind.EXTERNAL_RECORD)
        known = step().after
        malformed = (
            (step(), step(1, before=state(), after=known)),
            (step(1),),
            (step(), step(0)),
            (step(termination="terminated"), step(1, before=known, after=known)),
            (step(after=state(ref=bad_ref)),),
        )
        # The last candidate has no supported foreign evidence; use explicit request mismatch.
        foreign = step(after=state(("supported", "unverified", "unverified"), bad_ref))
        malformed = malformed[:-1] + ((replace(foreign, request=step().request),),)
        for steps in malformed:
            with self.subTest(steps=len(steps)):
                with self.assertRaises(ValueError):
                    prepare(
                        steps, authorize_evidence=lambda _: self.fail("invalid source reviewed")
                    )

    def test_mismatched_catalog_self_review_and_cycles_fail_closed(self):
        fact, checked, done = conditions()
        for specs, ctx in (
            ((replace(fact, task_id="other"), checked, done), context()),
            ((fact, replace(checked, dependencies=("done",)), done), context()),
            ((fact, replace(checked, dependencies=("missing",)), done), context()),
            (
                (fact, checked, done),
                context(evaluator=TrustedEvaluatorContract("student", "v1", "x")),
            ),
        ):
            with self.subTest(specs=specs):
                with self.assertRaises(ValueError):
                    prepare(
                        conditions=specs,
                        context=ctx,
                        authorize_evidence=lambda _: self.fail("invalid manifest reviewed"),
                    )

    def test_held_out_lineage_never_reaches_review(self):
        original = step()
        for metadata in (
            {"training_eligibility": "EVAL"},
            {"training_eligibility": "TRAIN", "source": {"training_eligibility": "HOLDOUT"}},
        ):
            target = replace(
                original.request,
                trajectory=replace(original.request.trajectory, sampling_parameters=metadata),
            )
            with self.assertRaises(ValueError):
                prepare(
                    (replace(original, request=target),),
                    authorize_evidence=lambda _: self.fail("heldout evidence reviewed"),
                )

    def test_serialization_is_versioned_detached_and_cannot_grant_authority(self):
        for record in (context(), *conditions(), step()):
            value = record.to_dict()
            self.assertEqual(type(record).from_dict(value).to_dict(), value)
            value["training_eligibility"] = "TRAIN"
            with self.assertRaises(ValueError):
                type(record).from_dict(value)
        captured = []

        def reviewer(envelope):
            captured.append(copy.deepcopy(envelope))
            return True

        result = prepare(authorize_evidence=reviewer)
        self.assertEqual(captured[0]["steps"][0]["before"], state().to_dict())
        result["steps"][0]["knowledge_gain"] = 999
        self.assertEqual(captured[0]["metrics"][0]["knowledge_gain"], 1)

    def test_invalid_numbers_private_evidence_and_noncanonical_states_reject(self):
        for gamma in (True, 0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                context(gamma=gamma)
        for epsilon in (0, True, float("inf")):
            with self.assertRaises(ValueError):
                prepare(epsilon=epsilon)
        private = EvidenceReference("private", EvidenceSourceKind.PRIVATE_REASONING)
        with self.assertRaises(ValueError):
            prepare((step(after=state(("supported", "unverified", "unverified"), private)),))
        with self.assertRaises(ValueError):
            prepare((step(after=state(("inferred", "unverified", "unverified"))),))


if __name__ == "__main__":
    unittest.main()
