from __future__ import annotations

import itertools
from dataclasses import asdict
from typing import Any, Dict, Final, Mapping, Sequence

from ..train.grn import GRNSettings, build_grn, torch  # torch is optional_import result
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
    "confidence_grn": asdict(GRNSettings()),
}


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


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return a scoring config merged with :data:`DEFAULT_CONFIG` and sanitised."""

    if not isinstance(overrides, Mapping):
        cfg = _clone_mapping(DEFAULT_CONFIG)
    else:
        cfg = _merge_mappings(DEFAULT_CONFIG, overrides)

    weight_defaults = DEFAULT_CONFIG["weights"]
    weights_obj = cfg.get("weights", weight_defaults)
    if isinstance(weights_obj, Mapping):
        cfg["weights"] = {
            tier: _safe_float(
                weights_obj.get(tier, weight_defaults.get(tier, 0.0)),
                weight_defaults.get(tier, 0.0),
            )
            for tier in weight_defaults
        }

    threshold_defaults = DEFAULT_CONFIG["abstention_thresholds"]
    thresholds_obj = cfg.get("abstention_thresholds", threshold_defaults)
    if isinstance(thresholds_obj, Mapping):
        cfg["abstention_thresholds"] = {
            dim: _safe_float(
                thresholds_obj.get(dim, threshold_defaults.get(dim, 0.75)),
                threshold_defaults.get(dim, 0.75),
            )
            for dim in threshold_defaults
        }

    cfg["disagreement_penalty"] = _safe_float(
        cfg.get("disagreement_penalty", DEFAULT_CONFIG["disagreement_penalty"]),
        DEFAULT_CONFIG["disagreement_penalty"],
    )
    cfg["escalate_if_any_below"] = _safe_float(
        cfg.get("escalate_if_any_below", DEFAULT_CONFIG["escalate_if_any_below"]),
        DEFAULT_CONFIG["escalate_if_any_below"],
    )

    cfg["confidence_grn"] = asdict(GRNSettings.from_mapping(cfg.get("confidence_grn")))

    return cfg


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

    weight_cfg = cfg.get("weights", DEFAULT_CONFIG["weights"])
    if not isinstance(weight_cfg, Mapping):
        weight_cfg = DEFAULT_CONFIG["weights"]
    thresholds_cfg = cfg.get("abstention_thresholds", DEFAULT_CONFIG["abstention_thresholds"])
    if not isinstance(thresholds_cfg, Mapping):
        thresholds_cfg = DEFAULT_CONFIG["abstention_thresholds"]

    penalty = _safe_float(
        cfg.get("disagreement_penalty"),
        DEFAULT_CONFIG["disagreement_penalty"],
    )
    escalate_floor = _safe_float(
        cfg.get("escalate_if_any_below"),
        DEFAULT_CONFIG["escalate_if_any_below"],
    )
    confidence_grn = GRNSettings.from_mapping(cfg.get("confidence_grn"))

    threshold_values = {
        dim: _safe_float(thresholds_cfg.get(dim), DEFAULT_CONFIG["abstention_thresholds"][dim])
        for dim in DIMENSIONS
    }

    final_scores: Dict[str, int] = {}
    raw_confidence: Dict[str, float] = {}
    reasons: list[str] = []

    gaps = _pairwise_disagreement(tiers)

    for dim in DIMENSIONS:
        weighted = 0.0
        total_weight = 0.0
        for tier in tiers:
            weight = _weight_for(tier, weight_cfg)
            weighted += weight * tier.scores[dim]
            total_weight += weight
        final_scores[dim] = int(round(weighted / max(total_weight, 1e-9)))
        confidence = sum(tier.confidence[dim] for tier in tiers) / max(len(tiers), 1)
        gap = gaps[dim]
        if gap >= 2:
            confidence -= penalty * max(gap - 1, 1)
        raw_confidence[dim] = max(0.0, min(1.0, confidence))

    grn_module = build_grn(confidence_grn)
    if grn_module is not None and torch is not None:
        with torch.no_grad():
            conf_tensor = torch.tensor(
                [[raw_confidence[dim] for dim in DIMENSIONS]], dtype=torch.float32
            )
            normalised = grn_module(conf_tensor).squeeze(0).clamp(0.0, 1.0)
        final_confidence = {
            dim: float(normalised[idx].item()) for idx, dim in enumerate(DIMENSIONS)
        }
    else:
        final_confidence = dict(raw_confidence)

    for dim in DIMENSIONS:
        if final_confidence[dim] < threshold_values[dim]:
            reasons.append(f"Confidence below threshold for {dim}")

    large_gaps = {dim: gap for dim, gap in gaps.items() if gap >= 2}
    escalate = any(value < escalate_floor for value in final_confidence.values())
    if large_gaps:
        escalate = True
        reasons.append("disagreement across tiers detected")

    return AggregateScores(
        final=final_scores,
        confidence=final_confidence,
        per_tier=list(tiers),
        escalate=escalate,
        reasons=reasons,
    )


__all__ = ["build_config", "aggregate_tiers", "DEFAULT_CONFIG"]
