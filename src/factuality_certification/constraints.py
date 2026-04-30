"""Constraint scoring helpers."""

from __future__ import annotations

from .types import ClaimSupport


def constraint_scores(supports: list[ClaimSupport]) -> dict[str, float]:
    total = max(1, len(supports))
    contradicted = sum(1 for s in supports if s.support_label == "contradicted")
    unsupported = sum(1 for s in supports if s.support_label == "unsupported")
    supported = sum(1 for s in supports if s.support_label == "supported")
    return {
        "C_support": supported / total,
        "C_contradiction": 1 - contradicted / total,
        "C_abstention": 1 - unsupported / total,
        "C_non_overrefusal": 1.0,
        "C_answer_scope": (supported + unsupported * 0.3) / total,
    }
