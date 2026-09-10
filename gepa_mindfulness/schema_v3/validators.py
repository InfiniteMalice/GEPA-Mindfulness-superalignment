"""Validation helpers for Schema V3 overlays."""

from __future__ import annotations

import math

from .case_v3 import CASE_NAMES, CaseV3Result


def validate_case_v3(result: CaseV3Result) -> None:
    """Validate canonical case IDs while preserving Case 0 as a non-canonical fallback."""
    if result.case_id != 0 and result.case_id not in CASE_NAMES:
        raise ValueError(
            "case_id must be non-canonical fallback 0 or a canonical ID from 1 through 17"
        )
    if result.reward_components.r_thought < 0.0:
        raise ValueError("r_thought must never be negative")
    if not math.isfinite(result.threshold_tau):
        raise ValueError("threshold_tau must be a real value")
