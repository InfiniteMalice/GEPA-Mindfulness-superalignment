"""Validation helpers for Schema V3 overlays."""

from __future__ import annotations

import math

from .case_v3 import _CASE_NAMES_BY_ID, CaseV3Result, build_compact_label


def validate_case_v3(result: CaseV3Result) -> None:
    """Validate exact canonical or fallback identity plus numeric invariants."""
    try:
        expected_name = _CASE_NAMES_BY_ID[result.case_id]
    except KeyError as exc:
        raise ValueError(
            "case_id must be non-canonical fallback 0 or a canonical ID from 1 through 17"
        ) from exc
    if result.base_case_name != expected_name:
        raise ValueError(f"base_case_name must be {expected_name!r} for case_id {result.case_id}")
    expected_label = build_compact_label(result)
    if result.compact_label != expected_label:
        raise ValueError(
            f"compact_label must be {expected_label!r} for the supplied structured fields"
        )
    if result.reward_components.r_thought < 0.0:
        raise ValueError("r_thought must never be negative")
    if not math.isfinite(result.threshold_tau):
        raise ValueError("threshold_tau must be a finite real value")
