"""CPU checks for separated ActionOPD agreement and localized distillation."""

# Standard library
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

# Local
from evaluation.recommendations import load_research_reference_registry
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.actionopd import (
    DecisionKind,
    SemanticAction,
    TeacherActionSample,
    actionopd_advantages,
    agreement_reward,
    evaluate_action_group,
    public_state_digest,
)
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.locd import build_local_segment, localized_forward_kl
from gepa_mindfulness.training.trajectory import Trajectory
from gepa_mindfulness.verification.interfaces import (
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)


def target(identifier="action", prompt="student prefix"):
    """Capture a real student rollout with explicit training provenance."""
    ref = EvidenceReference("observed", EvidenceSourceKind.EXTERNAL_RECORD)
    trajectory = Trajectory(
        identifier,
        None,
        prompt,
        "response",
        prompt_token_ids=(1, 2),
        response_token_ids=(3, 4, 5, 6),
        policy_version="student-v1",
        evidence_references=(ref,),
        sampling_parameters={"training_eligibility": "TRAIN"},
    )
    return RewardRequest(trajectory, (ref,))


def action(value="answer", kind=DecisionKind.GROUNDED_ANSWER):
    """Use host-normalized action content rather than teacher reasoning."""
    return SemanticAction(kind, json.dumps({"answer": value}))


def teacher(value="answer", sample_id="teacher-1", request=None):
    """Retain teacher identity, sampling configuration and exact public state."""
    return TeacherActionSample(
        sample_id,
        TrustedEvaluatorContract("teacher", "v1", "generation-only"),
        public_state_digest(request or target()),
        '{"temperature":0.7}',
        action(value),
    )


def verdict(identifier="action", supported=False):
    """Record independent negative outcome evidence without changing agreement."""
    ref = target().observable_references[0]
    names = ("task_fit", "dependencies_satisfied", "provenance_intact", "authorization_scope_valid")
    if supported:
        names += ("claimed_outcome_supported",)
    return RelationalVerificationResult(
        identifier,
        True,
        True,
        "none",
        True,
        True,
        supported,
        False,
        (ref,),
        tuple(VerificationEvidenceBinding(name, (ref,)) for name in names)
        + (VerificationEvidenceBinding("contradiction_status", (ref,)),),
    )


class ActionOPDTests(unittest.TestCase):
    """Agreement is a sampled semantic signal, never a correctness certificate."""

    def test_canonical_json_and_typed_decisions(self):
        """Argument-key order can match, while action types remain distinct."""
        left = SemanticAction(DecisionKind.TOOL_USE, '{"b":2,"a":1}')
        right = SemanticAction(DecisionKind.TOOL_USE, '{"a":1,"b":2}')
        self.assertEqual(left, right)
        self.assertNotEqual(action(), action(kind=DecisionKind.IDK))
        self.assertEqual(SemanticAction.from_dict(left.to_dict()), left)
        for payload in ('{"x":NaN}', '{"x":1,"x":2}', "[]"):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    SemanticAction(DecisionKind.TOOL_USE, payload)

    def test_published_reward_cases_and_empty_supervision(self):
        """Repeated samples contribute frequency; K=0 supplies no labels."""
        self.assertIsNone(agreement_reward(0, 0, valid=False, frequency_bonus=0.5))
        self.assertEqual(agreement_reward(2, 3, valid=True, frequency_bonus=0.5), 1.25)
        self.assertEqual(agreement_reward(1, 3, valid=True, frequency_bonus=0.5), 1)
        self.assertEqual(agreement_reward(0, 3, valid=True, frequency_bonus=0.5), 0)
        self.assertEqual(agreement_reward(0, 3, valid=False, frequency_bonus=0.5), -1)
        self.assertEqual(agreement_reward(1, 1, valid=True, frequency_bonus=0.5), 1)
        for count, size in ((True, 3), (4, 3), (-1, 3)):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    agreement_reward(count, size, valid=True, frequency_bonus=0.5)

    def test_shared_batch_whitening_differs_from_per_group_normalization(self):
        """All retained candidates share one population standard deviation."""
        values = actionopd_advantages(
            ((0.0, 2.0), (0.0, 4.0), (1.0, 1.0), (None, None)), epsilon=1e-8
        )
        scale = math.sqrt(2.5) + 1e-8
        self.assertAlmostEqual(values[0][0], -1 / scale)
        self.assertAlmostEqual(values[1][0], -2 / scale)
        self.assertIsNone(values[2])
        self.assertIsNone(values[3])

    def test_group_retains_multiset_and_separates_verified_failure(self):
        """An attractive unsupported teacher answer can match but remains unverified."""
        teachers = (teacher(), teacher(sample_id="teacher-2"), teacher("other", "teacher-3"))
        report = evaluate_action_group(
            (target(),),
            (action(),),
            teachers,
            (verdict(),),
            frequency_bonus=0.5,
            enabled=True,
        )
        self.assertEqual(report["candidates"][0]["agreement_reward"], 1.25)
        self.assertFalse(report["candidates"][0]["verification"]["claimed_outcome_supported"])
        self.assertFalse(report["authority_granted"])
        self.assertEqual(len(report["teacher_samples"]), 3)
        with self.assertRaises(ValueError):
            require_training_eligible(report)

    def test_wrong_state_model_and_verifier_identity_fail_closed(self):
        """Teacher generation cannot silently move supervision to another state."""
        for requests, teachers, outcomes in (
            ((target(), target("second", "different")), (teacher(),), (None, None)),
            ((target(),), (teacher(request=target(prompt="different")),), (None,)),
            ((target(),), (teacher(),), (verdict("foreign"),)),
        ):
            with self.subTest(requests=requests):
                with self.assertRaises(ValueError):
                    evaluate_action_group(
                        requests,
                        tuple(action() for _ in requests),
                        teachers,
                        outcomes,
                        frequency_bonus=0.5,
                        enabled=True,
                    )

    def test_default_off_does_not_inspect_records(self):
        """Ordinary training paths are unaffected by this optional adapter."""
        self.assertIsNone(evaluate_action_group(object(), object(), object(), object()))

    def test_absent_teacher_or_invalid_student_does_not_invent_a_match(self):
        """No teacher sample is not a negative teacher label."""
        empty = evaluate_action_group(
            (target(),), (action(),), (), (None,), frequency_bonus=0.5, enabled=True
        )
        self.assertIsNone(empty["candidates"][0]["agreement_reward"])
        invalid = evaluate_action_group(
            (target(),), (None,), (teacher(),), (None,), frequency_bonus=0.5, enabled=True
        )
        self.assertEqual(invalid["candidates"][0]["agreement_reward"], -1)

    def test_partial_teacher_set_uses_nominal_count_and_empty_valid_set_is_unknown(self):
        """Upstream retry filtering does not renormalize the declared sample budget."""
        from dataclasses import replace

        partial = evaluate_action_group(
            (target(),),
            (action(),),
            (teacher(), teacher(sample_id="second")),
            (None,),
            nominal_teacher_count=3,
            frequency_bonus=0.5,
            enabled=True,
        )
        self.assertEqual(partial["candidates"][0]["agreement_reward"], 1.25)
        empty = evaluate_action_group(
            (target(),),
            (None,),
            (replace(teacher(), action=None),),
            (None,),
            nominal_teacher_count=3,
            enabled=True,
        )
        self.assertIsNone(empty["candidates"][0]["agreement_reward"])

    def test_teacher_samples_are_exact_versioned_snapshots(self):
        """Configuration and generation identity survive serialization."""
        value = teacher()
        self.assertEqual(TeacherActionSample.from_dict(value.to_dict()), value)
        wire = value.to_dict()
        wire["schema_version"] = "future"
        with self.assertRaises(ValueError):
            TeacherActionSample.from_dict(wire)

    def test_primary_sources_are_in_the_existing_reference_registry(self):
        """New adapter citations extend the canonical bibliography instead of forking it."""
        refs = {value.reference_id: value for value in load_research_reference_registry()}
        self.assertEqual(refs["REF-ACTIONOPD"].arxiv_id, "2610.19820")
        self.assertEqual(refs["REF-LOCD"].arxiv_id, "2610.21427")

    def test_tool_schema_change_is_a_different_public_state(self):
        """Matching prompt text alone cannot erase a changed visible tool interface."""
        from dataclasses import replace

        other = RewardRequest(
            replace(target().trajectory, sampling_parameters={"tools": ["new"]}),
            target().observable_references,
        )
        self.assertNotEqual(public_state_digest(target()), public_state_digest(other))

    def test_group_resource_budget_stops_before_restoring_all_requests(self):
        """Retained input limits apply while ingesting large candidate groups."""
        from gepa_mindfulness.training import actionopd

        requests = tuple(target(str(i), "x" * 2_200_000) for i in range(3))
        with patch.object(actionopd, "_read_request", wraps=actionopd._read_request) as restore:
            with self.assertRaises(ValueError):
                evaluate_action_group(requests, (action(),) * 3, (), (None,) * 3, enabled=True)
            self.assertEqual(restore.call_count, 1)


class LocalSegmentTests(unittest.TestCase):
    """Student inputs retain the original prefix and exclude privileged guidance."""

    def test_local_span_masks_only_corrected_tokens(self):
        """Public prefix and original source remain unchanged after local preparation."""
        report = build_local_segment(
            target(),
            start=1,
            end=3,
            teacher_tokens=(8, 9),
            teacher_context="trusted private hint",
            enabled=True,
        )
        self.assertEqual(report["student_response_token_ids"], [3, 8, 9])
        self.assertEqual(report["loss_mask"], [False, True, True])
        self.assertNotIn("trusted private hint", json.dumps(report["student_inputs"]))
        self.assertEqual(report["teacher_context"], "trusted private hint")
        self.assertEqual(target().trajectory.response_token_ids, (3, 4, 5, 6))
        with self.assertRaises(ValueError):
            require_training_eligible(report)

    def test_invalid_or_whole_trajectory_region_rejected(self):
        """A local correction cannot silently become wholesale imitation."""
        for start, end, tokens in (
            (-1, 2, (8,)),
            (2, 1, (8,)),
            (0, 4, (8, 9, 10, 11)),
            (1, 3, (8,)),
            (True, 3, (8, 9)),
            (1, 5, (8, 9, 10, 11)),
        ):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    build_local_segment(
                        target(),
                        start=start,
                        end=end,
                        teacher_tokens=tokens,
                        teacher_context="hint",
                        enabled=True,
                    )

    def test_disabled_preparation_does_not_inspect_input(self):
        """No implicit token preparation occurs on default paths."""
        self.assertIsNone(build_local_segment(object()))

    def test_original_prompt_and_response_tokens_cannot_be_negative(self):
        """Source token validation is as strict as teacher continuation validation."""
        from dataclasses import replace

        for changes in ({"prompt_token_ids": (-1,)}, {"response_token_ids": (-2, 4, 5, 6)}):
            with self.subTest(changes=changes):
                source = RewardRequest(replace(target().trajectory, **changes))
                with self.assertRaises(ValueError):
                    build_local_segment(
                        source,
                        start=1,
                        end=3,
                        teacher_tokens=(8, 9),
                        teacher_context="hint",
                        enabled=True,
                    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable locally")
class LocalizedLossTests(unittest.TestCase):
    """Deterministic CPU tensor checks run in the repository's RL CI environment."""

    def test_teacher_detaches_and_prefix_has_zero_gradient(self):
        """Only student logits at the selected segment receive gradient."""
        import torch

        student = torch.zeros((1, 3, 2), requires_grad=True)
        teacher_logits = torch.tensor([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]], requires_grad=True)
        mask = torch.tensor([[False, True, True]])
        loss = localized_forward_kl(student, teacher_logits, mask)
        loss.backward()
        self.assertIsNone(teacher_logits.grad)
        self.assertEqual(student.grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(student.grad[0, 1:].abs().sum().item(), 0)

    def test_loss_is_zero_for_matching_distributions_and_rejects_empty_mask(self):
        """Masked normalization is finite and never treats empty data as successful training."""
        import torch

        logits = torch.zeros((1, 2, 3), requires_grad=True)
        self.assertAlmostEqual(
            localized_forward_kl(logits, logits, torch.ones((1, 2), dtype=torch.bool)).item(), 0
        )
        with self.assertRaises(ValueError):
            localized_forward_kl(logits, logits, torch.zeros((1, 2), dtype=torch.bool))

    def test_each_local_instance_has_equal_weight_despite_length(self):
        """LoCD averages each span's token mean before averaging separate instances."""
        import torch

        student = torch.zeros((2, 3, 2), requires_grad=True)
        teacher_logits = torch.tensor([[[0.0, 0.0]] * 3, [[2.0, 0.0]] * 3])
        mask = torch.tensor([[True, False, False], [True, True, True]])
        first = localized_forward_kl(student[:1], teacher_logits[:1], mask[:1])
        second = localized_forward_kl(student[1:], teacher_logits[1:], mask[1:])
        self.assertAlmostEqual(
            localized_forward_kl(student, teacher_logits, mask).item(),
            ((first + second) / 2).item(),
        )

    def test_large_masked_float64_prefix_cannot_corrupt_gradient(self):
        """Unused extreme logits retain zero prefix gradient and finite active gradients."""
        import torch

        student = torch.tensor(
            [[[1e300, -1e300], [0.0, 0.0]]], dtype=torch.float64, requires_grad=True
        )
        teacher_logits = torch.tensor(
            [[[-1e300, 1e300], [2.0, 0.0]]], dtype=torch.float64, requires_grad=True
        )
        loss = localized_forward_kl(student, teacher_logits, torch.tensor([[False, True]]))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(student.grad).all()))
        self.assertEqual(student.grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(student.grad[0, 1].abs().sum().item(), 0)
        self.assertIsNone(teacher_logits.grad)


if __name__ == "__main__":
    unittest.main()
