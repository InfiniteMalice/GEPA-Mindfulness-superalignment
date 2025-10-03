"""Aggregation logic for the wisdom scoring tiers."""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Mapping, Sequence

from .schema import AggregateScores, DIMENSIONS, TierScores


DEFAULT_CONFIG = {
    "weights": {
        "heuristic": 0.2,
        "judge": 0.5,
        "classifier": 0.3,
    },
    "abstention_thresholds": {dim: 0.75 for dim in DIMENSIONS},
    "disagreement_penalty": 0.25,
    "escalate_if_any_below": 0.5,
}


def _weight_for(tier: TierScores, config: Mapping[str, float]) -> float:
    return float(config.get(tier.tier, 0.0))


def _pairwise_disagreement(scores: Sequence[TierScores]) -> Dict[str, int]:
    gaps = {dim: 0 for dim in DIMENSIONS}
    for left, right in itertools.combinations(scores, 2):
        for dim in DIMENSIONS:
            diff = abs(int(left.scores[dim]) - int(right.scores[dim]))
            gaps[dim] = max(gaps[dim], diff)
    return gaps


def aggregate_tiers(tiers: Sequence[TierScores], config: Mapping[str, object] | None = None) -> AggregateScores:
    """Combine tier scores with disagreement-aware confidences."""

    if not tiers:
        raise ValueError("At least one tier score is required for aggregation")

    cfg = dict(DEFAULT_CONFIG)
    if config:

        for key, value in config.items():
            if key in {"weights", "abstention_thresholds"} and isinstance(value, Mapping):
                merged = dict(DEFAULT_CONFIG[key])
                merged.update(value)
                cfg[key] = merged
            else:
                cfg[key] = value

        cfg.update(config)

    weight_cfg = cfg.get("weights", DEFAULT_CONFIG["weights"])
    thresholds = cfg.get("abstention_thresholds", DEFAULT_CONFIG["abstention_thresholds"])
    penalty = float(cfg.get("disagreement_penalty", DEFAULT_CONFIG["disagreement_penalty"]))
    escalate_floor = float(cfg.get("escalate_if_any_below", DEFAULT_CONFIG["escalate_if_any_below"]))

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
        threshold = float(thresholds.get(dim, 0.75))
        if adjusted_conf < threshold:
            reasons.append(f"{dim}: confidence {adjusted_conf:.2f} below threshold {threshold:.2f}")

    escalate = any(final_confidence[dim] < float(thresholds.get(dim, 0.75)) for dim in DIMENSIONS)
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


__all__ = ["aggregate_tiers"]