"""Adversarial source and span binding tests for representation metrics."""

# Standard library
from __future__ import annotations

# Third-party
import pytest

# Local
from semantic_intent_robustness.representation import (
    RepresentationCandidate,
    RepresentationChannel,
    RepresentationLattice,
    SourceSpan,
    candidate_id_for,
    source_digest_for,
)
from semantic_intent_robustness.representation_metrics import (
    ExpectedCandidateIdentity,
    RepresentationEvaluationCase,
    RepresentationEvaluationResult,
    SemanticLaunderingAssessment,
    candidate_recall_at_k,
)
from semantic_intent_robustness.representation_routing import RepresentationDecision
from semantic_intent_robustness.taxonomy import PolicyAction


def _result(*, source_id: str = "metric-source") -> RepresentationEvaluationResult:
    raw_text = "teh then teh"
    candidate = RepresentationCandidate(
        source_span=SourceSpan(source_id, 9, 12, "teh"),
        candidate_text="the",
        transform_channel=RepresentationChannel.ORTHOGRAPHIC,
        orthographic_score=0.9,
        phonetic_score=0.0,
        contextual_score=0.5,
        semantic_similarity=0.5,
        confidence=0.9,
        provenance=("fixture:tail-span",),
        generation_reason="Hand-authored tail-span candidate.",
    )
    candidate_id = candidate_id_for(candidate)
    return RepresentationEvaluationResult(
        case_id="bound-case",
        lattice=RepresentationLattice(source_id, raw_text, (candidate,), 1),
        applied_candidate_id=None,
        decision=RepresentationDecision(
            selected_candidate_ids=(candidate_id,),
            disagreement=False,
            policy_action=PolicyAction.ALLOW,
            explanation="Hand-authored source-binding fixture.",
        ),
        semantic_laundering_assessment=SemanticLaunderingAssessment(
            harmful_intent_reworded=False,
            multi_turn_fragmentation=False,
            virtue_language_masks_risk=False,
            model_tracks_underlying_intent=True,
            identifies_real_world_target=True,
            distinguishes_topic_from_intent=True,
            checks_multi_turn_trajectory=True,
            preserves_safe_alternative=True,
        ),
    )


def _case(*, start: int = 9, end: int = 12) -> RepresentationEvaluationCase:
    raw_text = "teh then teh"
    return RepresentationEvaluationCase(
        case_id="bound-case",
        expected_source_id="metric-source",
        expected_source_digest=source_digest_for("metric-source", raw_text),
        expected_candidates=(ExpectedCandidateIdentity(start, end, "the"),),
        clean_input=False,
        abstention_expected=False,
        laundering_expected=False,
        expected_policy_action=PolicyAction.ALLOW,
    )


def test_recall_matches_expected_span_and_text_not_text_alone() -> None:
    assert candidate_recall_at_k((_case(),), (_result(),), k=1) == 1.0
    assert candidate_recall_at_k((_case(start=0, end=3),), (_result(),), k=1) == 0.0


def test_metric_pair_rejects_mismatched_source_identity() -> None:
    with pytest.raises(ValueError, match="expected source"):
        candidate_recall_at_k((_case(),), (_result(source_id="other-source"),), k=1)


def test_metric_pair_rejects_forged_source_digest() -> None:
    case = _case()
    object.__setattr__(case, "expected_source_digest", "representation-source-v1:" + "0" * 64)

    with pytest.raises(ValueError, match="expected source digest"):
        candidate_recall_at_k((case,), (_result(),), k=1)


def test_metric_pair_rejects_expected_span_outside_bound_source() -> None:
    case = _case()
    object.__setattr__(
        case,
        "expected_candidates",
        (ExpectedCandidateIdentity(9, 99, "the"),),
    )

    with pytest.raises(ValueError, match="expected candidate span"):
        candidate_recall_at_k((case,), (_result(),), k=1)
