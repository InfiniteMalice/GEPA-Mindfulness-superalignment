"""Validation helpers for Schema V3 overlays."""

from __future__ import annotations

import math

from .case_v3 import CaseV3Result


def validate_case_v3(result: CaseV3Result) -> None:
    """Validate invariants that preserve the base 13+0 reward schema."""
    if result.case_id < 0 or result.case_id > 13:
        raise ValueError("case_id must remain in the 13+0 range")
    if result.reward_components.r_thought < 0.0:
        raise ValueError("r_thought must never be negative")
    if not math.isfinite(result.threshold_tau):
        raise ValueError("threshold_tau must be a real value")
