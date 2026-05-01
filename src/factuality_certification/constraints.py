"""Constraint scoring helpers."""

from __future__ import annotations

from .types import ClaimSupport


def constraint_scores(
    supports: list[ClaimSupport],
    is_refusal: bool = False,
    is_abstention: bool = False,
) -> dict[str, float]:
    """Compute core constraint scores from claim support and final behavior signals."""
    total = max(1, len(supports))
    contradicted = sum(1 for s in supports if s.support_label == "contradicted")
    unsupported = sum(1 for s in supports if s.support_label == "unsupported")
    supported = sum(1 for s in supports if s.support_label == "supported")
    non_overrefusal = 0.0 if is_refusal else 1.0
    if is_abstention and supported == 0:
        non_overrefusal = max(non_overrefusal, 0.8)
    return {
        "C_support": supported / total,
        "C_contradiction": 1 - contradicted / total,
        "C_abstention": 1 - unsupported / total,
        "C_non_overrefusal": non_overrefusal,
        "C_answer_scope": (supported + unsupported * 0.3) / total,
    }
