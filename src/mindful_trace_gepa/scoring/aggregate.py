"""Aggregation logic for the wisdom scoring tiers."""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Dict, Final, List, Mapping, Sequence

from .schema import DIMENSIONS, AggregateScores, TierScores

DEFAULT_WEIGHTS: Final[Dict[str, float]] = {
    "heuristic": 0.2,
    "judge": 0.5,
    "classifier": 0.3,
}
DEFAULT_THRESHOLDS: Final[Dict[str, float]] = {dim: 0.75 for dim in DIMENSIONS}
DEFAULT_DISAGREEMENT_PENALTY: Final[float] = 0.25
DEFAULT_ESCALATE_FLOOR: Final[float] = 0.5

DEFAULT_CONFIG: Dict[str, Any] = {
    "weights": dict(DEFAULT_WEIGHTS),
    "abstention_thresholds": dict(DEFAULT_THRESHOLDS),
    "disagreement_penalty": DEFAULT_DISAGREEMENT_PENALTY,
    "escalate_if_any_below": DEFAULT_ESCALATE_FLOOR,
}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_float_mapping(
    data: Any,
    fallback: Mapping[str, float],
    *,
    ignore_none: bool = False,
) -> Dict[str, float]:
    result = {str(key): float(value) for key, value in fallback.items()}
    if not isinstance(data, Mapping):
        return result
    for key, value in data.items():
        if value is None and ignore_none:
            continue
        key_str = str(key)
        default_value = result.get(key_str, 0.0)
        result[key_str] = _safe_float(value, default_value)
    return result


def build_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return a scoring config merged with :data:`DEFAULT_CONFIG`."""

    cfg = deepcopy(DEFAULT_CONFIG)
    weights_base = _merge_float_mapping(cfg.get("weights"), DEFAULT_WEIGHTS)
    cfg["weights"] = weights_base
    thresholds_base = _merge_float_mapping(
        cfg.get("abstention_thresholds"),
        DEFAULT_THRESHOLDS,
    )
    cfg["abstention_thresholds"] = thresholds_base

    if not isinstance(overrides, Mapping):
        return cfg

    weights_override = overrides.get("weights")
    if isinstance(weights_override, Mapping):
        cfg["weights"] = _merge_float_mapping(
            weights_override,
            weights_base,
            ignore_none=True,
        )

    thresholds_override = overrides.get("abstention_thresholds")
    if isinstance(thresholds_override, Mapping):
        cfg["abstention_thresholds"] = _merge_float_mapping(
            thresholds_override,
            thresholds_base,
            ignore_none=True,
        )

    for key, value in overrides.items():
        if key in {"weights", "abstention_thresholds"}:
            continue
        if value is None:
            continue
        if key == "disagreement_penalty":
            cfg[key] = _safe_float(value, DEFAULT_DISAGREEMENT_PENALTY)
        elif key == "escalate_if_any_below":
            cfg[key] = _safe_float(value, DEFAULT_ESCALATE_FLOOR)
        else:
            cfg[key] = value

    return cfg


def _weight_for(tier: TierScores, config: Mapping[str, float]) -> float:
    raw = config.get(tier.tier, 0.0)
    return _safe_float(raw, 0.0)


def _pairwise_disagreement(scores: Sequence[TierScores]) -> Dict[str, int]:
    gaps = {dim: 0 for dim in DIMENSIONS}
    for left, right in itertools.combinations(scores, 2):
        for dim in DIMENSIONS:
            diff = abs(int(left.scores[dim]) - int(right.scores[dim]))
            gaps[dim] = max(gaps[dim], diff)
    return gaps


def aggregate_tiers(
    tiers: Sequence[TierScores],
    config: Mapping[str, Any] | None = None,
) -> AggregateScores:
    """Combine tier scores with disagreement-aware confidences."""

    if not tiers:
        raise ValueError("At least one tier score is required for aggregation")

    cfg = build_config(config)

    weight_cfg = _merge_float_mapping(cfg.get("weights"), DEFAULT_WEIGHTS)

    thresholds = _merge_float_mapping(
        cfg.get("abstention_thresholds"),
        DEFAULT_THRESHOLDS,
    )

    penalty = _safe_float(
        cfg.get("disagreement_penalty", DEFAULT_DISAGREEMENT_PENALTY),
        DEFAULT_DISAGREEMENT_PENALTY,
    )
    escalate_floor = _safe_float(
        cfg.get("escalate_if_any_below", DEFAULT_ESCALATE_FLOOR),
        DEFAULT_ESCALATE_FLOOR,
    )

    threshold_values = {}
    for dim in DIMENSIONS:
        default_threshold = DEFAULT_THRESHOLDS.get(dim, 0.75)
        override_threshold = thresholds.get(dim, default_threshold)
        threshold_values[dim] = _safe_float(override_threshold, default_threshold)

    final_scores: Dict[str, int] = {}
    final_confidence: Dict[str, float] = {}
    reasons: List[str] = []

    gaps = _pairwise_disagreement(tiers)

    for dim in DIMENSIONS:
        weighted_score = 0.0
        weighted_conf = 0.0
        weight_total = 0.0
        for tier in tiers:
            weight = _weight_for(tier, weight_cfg)
            if weight <= 0.0:
                continue
            tier_conf = float(tier.confidence.get(dim, 0.0))
            weighted_score += weight * tier_conf * int(tier.scores[dim])
            weighted_conf += weight * tier_conf
            weight_total += weight
        score = 0
        conf = 0.0
        if weight_total > 0 and weighted_conf > 0:
            score = int(round(weighted_score / max(weighted_conf, 1e-6)))
            conf = min(1.0, weighted_conf / weight_total)
        final_scores[dim] = max(0, min(4, score))
        disagreement_gap = gaps.get(dim, 0)
        adjusted_conf = max(0.0, conf - (penalty if disagreement_gap >= 2 else 0.0))
        final_confidence[dim] = adjusted_conf
        if disagreement_gap >= 2:
            reasons.append(f"{dim}: high tier disagreement (gap={disagreement_gap})")
        threshold = threshold_values[dim]
        if adjusted_conf < threshold:
            reasons.append(f"{dim}: confidence {adjusted_conf:.2f} below threshold {threshold:.2f}")

    escalate = any(final_confidence[dim] < threshold_values[dim] for dim in DIMENSIONS)
    if any(gaps[dim] >= 2 for dim in DIMENSIONS):
        escalate = True
    if any(final_confidence[dim] < escalate_floor for dim in DIMENSIONS):
        reasons.append("Escalated due to low confidence below floor")
        escalate = True

    return AggregateScores(
        final=final_scores,
        confidence=final_confidence,
        per_tier=list(tiers),
        escalate=escalate,
        reasons=reasons,
    )


__all__ = ["aggregate_tiers", "DEFAULT_CONFIG", "build_config"]
