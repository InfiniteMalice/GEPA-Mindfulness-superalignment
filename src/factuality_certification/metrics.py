"""Evaluation metrics for certification behavior."""

from __future__ import annotations


def ratio(num: float, den: float) -> float:
    """Safe ratio helper."""
    return 0.0 if den == 0 else num / den
