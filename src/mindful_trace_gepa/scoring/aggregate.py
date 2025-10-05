"""Aggregation logic for the wisdom scoring tiers."""

from __future__ import annotations

import itertools

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence


from .schema import DIMENSIONS, AggregateScores, TierScores

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

def build_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return a scoring config merged with :data:`DEFAULT_CONFIG`."""

    cfg = deepcopy(DEFAULT_CONFIG)
    if not isinstance(overrides, Mapping):
        return cfg

    weights_override = overrides.get("weights")
    if isinstance(weights_override, Mapping):
        dest = dict(cfg.get("weights", {}))
        for tier, weight in weights_override.items():
            if weight is None:
                continue
            dest[tier] = weight
        cfg["weights"] = dest

    thresholds_override = overrides.get("abstention_thresholds")
    if isinstance(thresholds_override, Mapping):
        dest = dict(cfg.get("abstention_thresholds", {}))
        for dim, threshold in thresholds_override.items():
            if threshold is None:
                continue
            dest[dim] = threshold
        cfg["abstention_thresholds"] = dest

    for key, value in overrides.items():
        if key in {"weights", "abstention_thresholds"}:
            continue
        if value is None:
            continue
        cfg[key] = value

    return cfg

def _clone_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            cloned[key] = _clone_mapping(value)
        else:
            cloned[key] = value
    return cloned


def _merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged = _clone_mapping(base)
    for key, value in overrides.items():
        if value is None:
            continue
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(base_value, value)
        elif isinstance(value, Mapping):
            merged[key] = _merge_mappings({}, value)
        else:
            merged[key] = value
    return merged


def build_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return a scoring config merged with :data:`DEFAULT_CONFIG`."""

    if not isinstance(overrides, Mapping):
        return _clone_mapping(DEFAULT_CONFIG)
    return _merge_mappings(DEFAULT_CONFIG, overrides)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weight_for(tier: TierScores, config: Mapping[str, Any]) -> float:
    raw = config.get(tier.tier, 0.0) if isinstance(config, Mapping) else 0.0
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
    config: Mapping[str, object] | None = None,
) -> AggregateScores:
    """Combine tier scores with disagreement-aware confidences."""

    if not tiers:
        raise ValueError("At least one tier score is required for aggregation")

    cfg = build_config(config)

    weight_cfg_obj = cfg.get("weights", DEFAULT_CONFIG["weights"])
    if not isinstance(weight_cfg_obj, Mapping):
        weight_cfg_obj = DEFAULT_CONFIG["weights"]
    weight_cfg = dict(weight_cfg_obj)

    thresholds_obj = cfg.get("abstention_thresholds", DEFAULT_CONFIG["abstention_thresholds"])
    if not isinstance(thresholds_obj, Mapping):
        thresholds_obj = DEFAULT_CONFIG["abstention_thresholds"]
    thresholds = dict(thresholds_obj)

    default_thresholds = DEFAULT_CONFIG["abstention_thresholds"]
    penalty = _safe_float(
        cfg.get("disagreement_penalty", DEFAULT_CONFIG["disagreement_penalty"]),
        DEFAULT_CONFIG["disagreement_penalty"],
    )
    escalate_floor = _safe_float(
        cfg.get("escalate_if_any_below", DEFAULT_CONFIG["escalate_if_any_below"]),
        DEFAULT_CONFIG["escalate_if_any_below"],
    )

    threshold_values = {
        dim: _safe_float(thresholds.get(dim, default_thresholds.get(dim, 0.75)), default_thresholds.get(dim, 0.75))
        for dim in DIMENSIONS
    }

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
