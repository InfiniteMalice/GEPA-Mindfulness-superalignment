"""CPU trace, recovery and trust-boundary contracts for opt-in DeLoop."""

# Standard library
import copy
import importlib.util
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

# Local
from evaluation.causal_records import canonical_json
from evaluation.experimental_overlays import ExperimentalOverlayConfig, enabled_overlays
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.actionopd import DecisionKind, SemanticAction
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.deloop import (
    LoopPairReview,
    LoopStep,
    audit_loops,
    deloop_dpo_loss,
    mine_loop_pairs,
)
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.trajectory import Trajectory
from gepa_mindfulness.verification.interfaces import (
    LocalVerificationResult,
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)
from mindful_trace_gepa.logging_schema import EventEnvelope

REVIEWER = TrustedEvaluatorContract("independent", "v1", "original-state-loop-review-v1")


def fixture(actions=None, observations=None, versions=None, contexts=None):
    """Capture an entire authored PEO run with unique action and observation evidence."""
    actions = actions or (
        SemanticAction(DecisionKind.TOOL_USE, '{"tool":"read","args":{"a":1,"b":2}}'),
        SemanticAction(DecisionKind.TOOL_USE, '{"args":{"b":2,"a":1},"tool":"read"}'),
        SemanticAction(DecisionKind.RESUMPTION, '{"operation":"use_observation"}'),
    )
    count = len(actions)
    observations = observations or ("same",) * count
    versions = versions or ("state-v1",) * count
    contexts = contexts or ("ordinary",) * count
    events, steps = [], []
    for index, action in enumerate(actions):
        action_id = f"action-{index}"
        evidence = EvidenceReference(
            f"observation-evidence-{index}", EvidenceSourceKind.EXTERNAL_RECORD
        )
        context = EvidenceReference(f"context-{index}", EvidenceSourceKind.EXTERNAL_RECORD)
        refs = (evidence, context)
        request = RewardRequest(
            Trajectory(
                action_id,
                None,
                f"actual public history before step {index}",
                canonical_json(action.to_dict()),
                policy_version="student-v1",
                model_identifier="student",
                prompt_token_ids=(1, index),
                evidence_references=refs,
                sampling_parameters={"training_eligibility": "TRAIN"},
            ),
            refs,
        )
        step = LoopStep(request, action, versions[index], contexts[index], (context,))
        steps.append(step)
        common = dict(
            run_id="run",
            repeat_id=0,
            model_id="student",
            model_version="student-v1",
            harness_version="host-v1",
        )
        timestamp = datetime(2026, 10, 27, tzinfo=timezone.utc) + timedelta(seconds=index * 5)

        def event(offset, kind, payload, **metadata):
            return EventEnvelope(
                "1.0",
                f"event-{index}-{offset}",
                kind,
                (timestamp + timedelta(seconds=offset)).isoformat(),
                payload=payload,
                **common,
                **metadata,
            )

        prediction = event(
            0,
            "prediction_commit",
            {
                "prediction_commit_id": f"prediction-{index}",
                "predicted_outcome": {"deloop": step.commitment()},
                "confidence": 0.5,
                "evidence_refs": [context.reference_id],
            },
            evidence_refs=(context.reference_id,),
        )
        payload = {
            "action_id": action_id,
            "action_class": action.kind.value,
            "reversible": True,
            "authorization_scope": "sandbox",
            "prediction_commit_id": f"prediction-{index}",
        }
        proposed = event(
            1,
            "action_proposed",
            payload,
            action_id=action_id,
            authorization_scope="sandbox",
            parent_event_ids=(prediction.event_id,),
        )
        executed = event(
            2,
            "action_executed",
            payload,
            action_id=action_id,
            authorization_scope="sandbox",
            parent_event_ids=(proposed.event_id,),
        )
        observed = event(
            3,
            "outcome_observed",
            {
                "observation_id": f"observation-{index}",
                "action_id": action_id,
                "actual_outcome": observations[index],
                "evidence_refs": [evidence.reference_id],
            },
            action_id=action_id,
            evidence_refs=(evidence.reference_id,),
            parent_event_ids=(executed.event_id,),
        )
        verified = event(
            4,
            "verification_result",
            {
                "verifier_id": f"observation-review-{index}",
                "verifier_version": "v1",
                "observation_id": f"observation-{index}",
                "verified": True,
                "verifier_refs": ["independent-observation-check"],
            },
            verifier_refs=("independent-observation-check",),
            parent_event_ids=(observed.event_id,),
        )
        events.extend((prediction, proposed, executed, observed, verified))
    return tuple(events), tuple(steps)


def review(candidate, *, supported=True, original=True, identity=REVIEWER):
    """Supply explicitly independent fixture findings, not model-generated certification."""
    record = candidate.to_dict()
    refs = tuple(EvidenceReference.from_dict(ref) for ref in record["evidence_refs"])

    def bindings(names):
        return tuple(VerificationEvidenceBinding(name, refs) for name in names)

    rejected_id, chosen_id = record["rejected_action_id"], record["chosen_action_id"]
    loop = RelationalVerificationResult(
        rejected_id,
        False,
        True,
        "none",
        True,
        True,
        False,
        True,
        refs,
        bindings(
            (
                "dependencies_satisfied",
                "contradiction_status",
                "provenance_intact",
                "authorization_scope_valid",
                "repeated_failed_route",
            )
        ),
    )
    local = LocalVerificationResult(
        chosen_id,
        True,
        True,
        True,
        True,
        True,
        None,
        refs,
        bindings(
            (
                "executed",
                "arguments_valid",
                "schema_valid",
                "authorization_valid",
                "intended_operation_observed",
            )
        ),
    )
    recovery_names = [
        "task_fit",
        "dependencies_satisfied",
        "contradiction_status",
        "provenance_intact",
        "authorization_scope_valid",
    ]
    if supported:
        recovery_names.append("claimed_outcome_supported")
    recovery = RelationalVerificationResult(
        chosen_id, True, True, "none", True, True, supported, False, refs, bindings(recovery_names)
    )
    return LoopPairReview(candidate.digest, identity, original, loop, local, recovery)


class LoopAuditTests(unittest.TestCase):
    """Exact repetition stays a diagnostic until independently evaluated."""

    def test_invalid_dpo_coefficient_fails_before_optional_runtime_import(self):
        """Malformed coefficients do not load a tensor backend or accept boolean beta."""
        for beta in (0, -1, True, float("inf"), 10**1000, "0.1"):
            with self.subTest(beta=beta):
                with self.assertRaises(ValueError):
                    deloop_dpo_loss(object(), object(), object(), object(), beta=beta)

    def test_existing_overlay_registry_keeps_the_feature_default_off(self):
        """Loop diagnostics extend the existing opt-in declaration, not the case registry."""
        self.assertEqual(enabled_overlays(ExperimentalOverlayConfig()), ())
        declarations = enabled_overlays(ExperimentalOverlayConfig(repeated_action_audit=True))
        self.assertEqual(tuple(value.id for value in declarations), ("repeated_action_audit",))
        self.assertIn("direct_optimizer_reward", declarations[0].prohibited_effects)

    def test_default_off_does_not_inspect_sources_or_call_review(self):
        self.assertIsNone(audit_loops(object(), object(), task_id="task"))
        self.assertIsNone(
            mine_loop_pairs(object(), object(), task_id="task", reviewer=object(), review=object())
        )

    def test_canonical_actions_and_unchanged_observations_are_structural_repeats(self):
        events, steps = fixture()
        result = audit_loops(events, steps, task_id="task", enabled=True)
        self.assertEqual([row["raw_repeat"] for row in result["entries"]], [False, True, False])
        self.assertEqual(result["entries"][1]["previous_action_id"], "action-0")
        self.assertEqual(result["entries"][1]["classification"], "unresolved_repeat")
        self.assertFalse(result["authority_granted"])
        with self.assertRaises(ValueError):
            require_training_eligible(result)

    def test_changed_observation_and_environment_are_not_unproductive_by_default(self):
        for kwargs in ({"observations": ("old", "new", "new")}, {"versions": ("v1", "v2", "v2")}):
            with self.subTest(kwargs=kwargs):
                events, steps = fixture(**kwargs)
                result = audit_loops(events, steps, task_id="task", enabled=True)
                self.assertFalse(result["entries"][1]["raw_repeat"])
                self.assertEqual(
                    result["entries"][1]["classification"], "informative_or_state_changed_repeat"
                )

    def test_human_rechecks_and_independent_verification_are_exempt(self):
        for context in ("human_recheck", "independent_verification", "productive_repeat"):
            events, steps = fixture(contexts=("ordinary", context, "ordinary"))
            result = audit_loops(events, steps, task_id="task", enabled=True)
            self.assertTrue(result["entries"][1]["raw_repeat"])
            self.assertEqual(result["entries"][1]["classification"], "exempt_repeat")
            mined = mine_loop_pairs(
                events,
                steps,
                task_id="task",
                reviewer=REVIEWER,
                review=lambda _: self.fail("exempt repeat reviewed"),
                enabled=True,
            )
            self.assertEqual(mined["pairs"], [])

    def test_repeated_clarification_has_only_a_case_17_overlay_subtype(self):
        clarify = SemanticAction(
            DecisionKind.CLARIFICATION, '{"question":"Which approved target?"}'
        )
        events, steps = fixture(
            actions=(
                clarify,
                clarify,
                SemanticAction(DecisionKind.RESUMPTION, '{"operation":"resume"}'),
            )
        )
        result = audit_loops(events, steps, task_id="task", enabled=True)
        self.assertEqual(
            result["entries"][1]["diagnostic_subtype"], "repeated_clarification_unchanged_state"
        )
        self.assertNotIn("canonical_case_id", result["entries"][1])

    def test_input_serialization_is_versioned_and_detached(self):
        _events, steps = fixture()
        wire = steps[0].to_dict()
        self.assertEqual(LoopStep.from_dict(wire).to_dict(), wire)
        wire["schema_version"] = "future"
        with self.assertRaises(ValueError):
            LoopStep.from_dict(wire)

    def test_omitted_or_reordered_actions_and_foreign_state_fail_closed(self):
        events, steps = fixture()
        for changed in (
            steps[1:],
            tuple(reversed(steps)),
            (replace(steps[0], state_version="foreign"),) + steps[1:],
        ):
            with self.subTest(changed=changed[0].state_version):
                with self.assertRaises(ValueError):
                    audit_loops(events, changed, task_id="task", enabled=True)

    def test_unverified_observation_cannot_become_negative_training_signal(self):
        events, steps = fixture()
        payload = dict(events[9].payload) | {"verified": False}
        events = events[:9] + (replace(events[9], payload=payload),) + events[10:]
        result = mine_loop_pairs(
            events,
            steps,
            task_id="task",
            reviewer=REVIEWER,
            review=lambda _: self.fail("unverified source reviewed"),
            enabled=True,
        )
        self.assertEqual(result["pairs"], [])

    def test_existing_typed_verification_events_are_supported(self):
        """Typed relational provenance can accompany the captured source observation."""
        events, steps = fixture()
        changed = []
        for event in events:
            if event.event_type != "verification_result":
                changed.append(event)
                continue
            index = int(event.event_id.split("-")[1])
            refs = steps[index].request.observable_references
            fields = (
                "task_fit",
                "dependencies_satisfied",
                "contradiction_status",
                "provenance_intact",
                "authorization_scope_valid",
                "claimed_outcome_supported",
            )
            value = RelationalVerificationResult(
                f"action-{index}",
                True,
                True,
                "none",
                True,
                True,
                True,
                False,
                refs,
                tuple(VerificationEvidenceBinding(name, refs) for name in fields),
            )
            changed.append(
                replace(
                    event,
                    action_id=value.action_id,
                    evidence_refs=tuple(ref.reference_id for ref in refs),
                    payload={
                        "verification_level": "relational_evidence",
                        "result": value.to_dict(),
                        "verifier_refs": list(event.verifier_refs),
                    },
                )
            )
        result = mine_loop_pairs(
            tuple(changed), steps, task_id="task", reviewer=REVIEWER, review=review, enabled=True
        )
        self.assertEqual(len(result["pairs"]), 1)

    def test_exemption_requires_observable_source_bound_evidence(self):
        events, steps = fixture()
        with self.assertRaises(ValueError):
            replace(steps[1], repeat_context="human_recheck", context_refs=())
        with self.assertRaises(ValueError):
            audit_loops(
                events,
                steps[:1] + (replace(steps[1], repeat_context="human_recheck"),) + steps[2:],
                task_id="task",
                enabled=True,
            )

    def test_action_class_must_match_the_committed_semantic_action(self):
        """The execution record cannot silently describe another kind of action."""
        events, steps = fixture()
        changed = tuple(
            (
                replace(event, payload=dict(event.payload) | {"action_class": "foreign"})
                if event.event_type in ("action_proposed", "action_executed")
                else event
            )
            for event in events
        )
        with self.assertRaises(ValueError):
            audit_loops(changed, steps, task_id="task", enabled=True)

    def test_answered_clarification_uses_a_source_bound_diagnostic(self):
        """An observed adequate answer does not become another canonical case."""
        clarify = SemanticAction(DecisionKind.CLARIFICATION, '{"question":"Which target?"}')
        events, steps = fixture(
            actions=(clarify, clarify), contexts=("ordinary", "answered_clarification")
        )
        result = audit_loops(events, steps, task_id="task", enabled=True)
        self.assertEqual(
            result["entries"][1]["diagnostic_subtype"], "repeated_clarification_after_answer"
        )

    def test_aggregate_budget_rejects_before_review(self):
        """A group of individually bounded sources cannot exceed the shared input budget."""
        events, steps = fixture()
        oversized = tuple(
            replace(
                step,
                request=replace(
                    step.request,
                    trajectory=replace(step.request.trajectory, prompt="x" * 2_200_000),
                ),
            )
            for step in steps
        )
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            mine_loop_pairs(
                events,
                oversized,
                task_id="task",
                reviewer=REVIEWER,
                review=lambda _: self.fail("over-budget source reviewed"),
                enabled=True,
            )


class LoopPairTests(unittest.TestCase):
    """Recovery proposals preserve the rejected state's prefix and independent findings."""

    def test_conflicting_cross_step_evidence_identity_fails_before_review(self):
        """A later observation cannot replace the source kind of an earlier typed reference."""
        events, steps = fixture()
        conflicting = EvidenceReference(
            "observation-evidence-0", EvidenceSourceKind.OBSERVABLE_OUTPUT
        )
        changed = replace(
            steps[2],
            request=replace(
                steps[2].request,
                trajectory=replace(
                    steps[2].request.trajectory,
                    evidence_references=steps[2].request.trajectory.evidence_references
                    + (conflicting,),
                ),
                observable_references=steps[2].request.observable_references + (conflicting,),
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicting evidence"):
            mine_loop_pairs(
                events,
                steps[:2] + (changed,),
                task_id="task",
                reviewer=REVIEWER,
                review=lambda _: self.fail("conflicting source reached reviewer"),
                enabled=True,
            )

    def test_verified_pair_uses_original_state_and_first_recovery(self):
        events, steps = fixture()
        result = mine_loop_pairs(
            events, steps, task_id="task", reviewer=REVIEWER, review=review, enabled=True
        )
        self.assertEqual(len(result["pairs"]), 1)
        pair = result["pairs"][0]
        self.assertEqual(pair["prompt"], steps[1].request.trajectory.prompt)
        self.assertNotEqual(pair["prompt"], steps[2].request.trajectory.prompt)
        self.assertEqual(pair["chosen"], steps[2].request.trajectory.response)
        self.assertEqual(pair["rejected"], steps[1].request.trajectory.response)
        self.assertEqual(pair["chosen_action_id"], "action-2")
        with self.assertRaises(ValueError):
            require_training_eligible({"training_eligibility": "TRAIN", "source": result})

    def test_different_observation_from_same_call_is_not_a_positive_action(self):
        actions = fixture()[1]
        same = actions[0].action
        events, steps = fixture(
            actions=(same, same, same, actions[2].action),
            observations=("old", "old", "new", "done"),
        )
        result = mine_loop_pairs(
            events, steps, task_id="task", reviewer=REVIEWER, review=review, enabled=True
        )
        self.assertEqual(result["pairs"][0]["chosen_action_id"], "action-3")

    def test_unsupported_or_future_dependent_recovery_yields_no_pair(self):
        events, steps = fixture()
        for kwargs in ({"supported": False}, {"original": False}, {"original": None}):
            with self.subTest(kwargs=kwargs):
                result = mine_loop_pairs(
                    events,
                    steps,
                    task_id="task",
                    reviewer=REVIEWER,
                    review=lambda c: review(c, **kwargs),
                    enabled=True,
                )
                self.assertEqual(result["pairs"], [])
                self.assertTrue(result["rejected_proposals"])

    def test_failed_first_recovery_is_not_replaced_with_a_later_success(self):
        base = fixture()[1]
        events, steps = fixture(
            actions=(
                base[0].action,
                base[1].action,
                SemanticAction(DecisionKind.IDK, '{"reason":"stop"}'),
                base[2].action,
            )
        )
        called = []

        def evaluate(candidate):
            called.append(candidate.to_dict()["chosen_action_id"])
            return review(candidate, supported=False)

        result = mine_loop_pairs(
            events, steps, task_id="task", reviewer=REVIEWER, review=evaluate, enabled=True
        )
        self.assertEqual(called, ["action-2"])
        self.assertEqual(result["pairs"], [])

    def test_self_review_and_wrong_subject_are_rejected(self):
        events, steps = fixture()
        with self.assertRaises(ValueError):
            mine_loop_pairs(
                events,
                steps,
                task_id="task",
                reviewer=TrustedEvaluatorContract("student", "v1", "self"),
                review=review,
                enabled=True,
            )
        with self.assertRaises(ValueError):
            mine_loop_pairs(
                events,
                steps,
                task_id="task",
                reviewer=REVIEWER,
                review=lambda c: replace(review(c), candidate_digest="0" * 64),
                enabled=True,
            )

    def test_review_round_trip_retains_independent_result_bindings(self):
        events, steps = fixture()
        captured = []

        def evaluate(candidate):
            value = review(candidate)
            self.assertEqual(LoopPairReview.from_dict(value.to_dict()).to_dict(), value.to_dict())
            captured.append(copy.deepcopy(value.to_dict()))
            return value

        result = mine_loop_pairs(
            events, steps, task_id="task", reviewer=REVIEWER, review=evaluate, enabled=True
        )
        self.assertEqual(result["pairs"][0]["review"], captured[0])

    def test_contradictory_repeat_finding_is_not_a_usable_preference(self):
        """Independent disagreement stays unresolved instead of becoming optimizer supervision."""
        events, steps = fixture()

        def disagree(candidate):
            checked = review(candidate)
            return replace(checked, loop=replace(checked.loop, contradiction_status="contradicted"))

        result = mine_loop_pairs(
            events, steps, task_id="task", reviewer=REVIEWER, review=disagree, enabled=True
        )
        self.assertEqual(result["pairs"], [])

    def test_review_record_rejects_wrong_result_types_before_serializing(self):
        """Malformed host findings fail with ValueError rather than incidental attribute errors."""
        events, steps = fixture()
        with self.assertRaises(ValueError):
            mine_loop_pairs(
                events,
                steps,
                task_id="task",
                reviewer=REVIEWER,
                review=lambda c: replace(review(c), local_recovery=object()),
                enabled=True,
            )

    def test_review_cannot_support_outcomes_using_only_unrelated_context(self):
        """Outcome findings must retain the source observation references they concern."""
        events, steps = fixture()

        def wrong_support(candidate):
            checked = review(candidate)
            refs = tuple(
                ref
                for ref in checked.recovery.evidence_refs
                if ref.reference_id.startswith("context")
            )
            bindings = tuple(
                replace(binding, evidence_refs=refs)
                for binding in checked.recovery.evidence_bindings
            )
            return replace(
                checked,
                recovery=replace(checked.recovery, evidence_refs=refs, evidence_bindings=bindings),
            )

        with self.assertRaises(ValueError):
            mine_loop_pairs(
                events, steps, task_id="task", reviewer=REVIEWER, review=wrong_support, enabled=True
            )


@unittest.skipUnless(importlib.util.find_spec("torch"), "real PyTorch unavailable")
class LoopLossTests(unittest.TestCase):
    """DPO acts on sequence log probabilities, with a detached frozen reference."""

    def test_dpo_gradient_and_reference_detachment(self):
        import torch

        chosen = torch.tensor([-2.0], requires_grad=True)
        rejected = torch.tensor([-3.0], requires_grad=True)
        reference_chosen = torch.tensor([-2.0], requires_grad=True)
        reference_rejected = torch.tensor([-3.0], requires_grad=True)
        loss = deloop_dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1)
        self.assertAlmostEqual(loss.item(), 0.69314718056, places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        self.assertIsNone(reference_chosen.grad)
        self.assertIsNone(reference_rejected.grad)

    def test_dpo_rejects_empty_nonfinite_and_mismatched_inputs(self):
        import torch

        for value in (torch.tensor([]), torch.tensor([float("nan")]), torch.ones((2, 1))):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    deloop_dpo_loss(value, torch.ones(1), torch.ones(1), torch.ones(1), beta=0.1)

    def test_dpo_rejects_overflowing_margin_even_when_softplus_would_return_zero(self):
        """A saturated loss cannot conceal a nonfinite intermediate preference margin."""
        import torch

        low = torch.tensor([-1e308], dtype=torch.float64, requires_grad=True)
        high = torch.tensor([-1.0], dtype=torch.float64)
        with self.assertRaises(ValueError):
            deloop_dpo_loss(low, low, low, high, beta=100.0)
