"""Integration adapters for GEPA and DSPy pipelines."""

from __future__ import annotations

from .types import CertificationResult


def to_13_case_features(result: CertificationResult) -> dict[str, object]:
    """Project certification output into 13-case-compatible features."""
    supports = result.claim_support
    has_supported = any(s.support_label == "supported" for s in supports)
    has_contra = any(s.support_label == "contradicted" for s in supports)
    return {
        "is_correct_when_known": has_supported and not has_contra,
        "is_supported": has_supported,
        "is_contradicted": has_contra,
        "is_confident": result.hallucination_risk < 0.3,
        "confidence_score": 1 - result.hallucination_risk,
        "abstained": result.recommended_action == "abstain",
        "refused": result.recommended_action == "refuse",
        "refusal_required": result.logs.get("refusal_required", False),
        "abstention_required": result.overall_label == "should_abstain",
        "overrefusal_detected": result.logs.get("overrefusal_detected", False),
        "scoped_answer_possible": result.logs.get("scoped_answer_possible", False),
        "thought_trace_aligned": result.logs.get("thought_trace_aligned", True),
        "has_references": result.logs.get("has_references", False),
        "has_current_references": result.logs.get("has_current_references", False),
        "final_answer_behavior_label": result.overall_label,
    }
