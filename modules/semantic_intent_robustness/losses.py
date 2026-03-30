"""Training utilities for semantic intent robustness objectives."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticBatch:
    """Batch contract for invariance and contrastive training losses."""

    invariant_scores: list[float]
    contrastive_margins: list[float]
    policy_match_scores: list[float]
    abstention_targets: list[float]
    abstention_predictions: list[float]
    auxiliary_errors: list[float]


@dataclass(frozen=True)
class LossBreakdown:
    """Named loss components for logging and weighting."""

    invariance_loss: float
    contrastive_loss: float
    policy_consistency_loss: float
    abstention_calibration_loss: float
    auxiliary_supervision_loss: float
    invariance_weight: float = 1.0
    contrastive_weight: float = 1.0
    policy_consistency_weight: float = 1.0
    abstention_calibration_weight: float = 1.0
    auxiliary_supervision_weight: float = 1.0

    def __post_init__(self) -> None:
        weights = [
            self.invariance_weight,
            self.contrastive_weight,
            self.policy_consistency_weight,
            self.abstention_calibration_weight,
            self.auxiliary_supervision_weight,
        ]
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Loss weights must be non-negative")

    @property
    def total(self) -> float:
        return (
            self.invariance_loss * self.invariance_weight
            + self.contrastive_loss * self.contrastive_weight
            + self.policy_consistency_loss * self.policy_consistency_weight
            + self.abstention_calibration_loss * self.abstention_calibration_weight
            + self.auxiliary_supervision_loss * self.auxiliary_supervision_weight
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "invariance_loss": self.invariance_loss,
            "contrastive_loss": self.contrastive_loss,
            "policy_consistency_loss": self.policy_consistency_loss,
            "abstention_calibration_loss": self.abstention_calibration_loss,
            "auxiliary_supervision_loss": self.auxiliary_supervision_loss,
            "total": self.total,
        }


def invariance_loss(scores: list[float]) -> float:
    """Penalize disagreement among semantically equivalent variants."""

    return _penalize_disagreement(scores)


def contrastive_separation_loss(margins: list[float], *, margin: float = 0.2) -> float:
    """Encourage same-topic different-intent separation."""

    penalties = [max(0.0, margin - observed) for observed in margins]
    return _mean_loss(penalties)


def policy_consistency_loss(scores: list[float]) -> float:
    """Penalize policy instability for equivalent prompts."""

    return _penalize_disagreement(scores)


def abstention_calibration_loss(targets: list[float], predictions: list[float]) -> float:
    """Simple squared-error calibration objective for abstention heads."""

    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    errors = [(target - pred) ** 2 for target, pred in zip(targets, predictions)]
    return _mean_loss(errors)


def auxiliary_supervision_loss(errors: list[float]) -> float:
    """Aggregate optional decomposition supervision errors."""

    return _mean_loss(errors)


def compute_loss_breakdown(
    batch: SemanticBatch,
    *,
    invariance_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    policy_consistency_weight: float = 1.0,
    abstention_calibration_weight: float = 1.0,
    auxiliary_supervision_weight: float = 1.0,
) -> LossBreakdown:
    """Compute all loss components for the current training batch."""

    return LossBreakdown(
        invariance_loss=invariance_loss(batch.invariant_scores),
        contrastive_loss=contrastive_separation_loss(batch.contrastive_margins),
        policy_consistency_loss=policy_consistency_loss(batch.policy_match_scores),
        abstention_calibration_loss=abstention_calibration_loss(
            batch.abstention_targets,
            batch.abstention_predictions,
        ),
        auxiliary_supervision_loss=auxiliary_supervision_loss(batch.auxiliary_errors),
        invariance_weight=invariance_weight,
        contrastive_weight=contrastive_weight,
        policy_consistency_weight=policy_consistency_weight,
        abstention_calibration_weight=abstention_calibration_weight,
        auxiliary_supervision_weight=auxiliary_supervision_weight,
    )


def batch_format_expectations() -> dict[str, Any]:
    """Describe expected batch tensors/arrays for training integration."""

    expectations = {
        "invariant_scores": "per-cluster semantic agreement in [0, 1]",
        "contrastive_margins": "distance between topic-matched but intent-shifted pairs",
        "policy_match_scores": "per-cluster policy agreement in [0, 1]",
        "abstention_targets": "target abstention probabilities for ambiguous prompts",
        "abstention_predictions": "model abstention predictions",
        "auxiliary_errors": "optional decomposition field losses",
    }
    _validate_probability_fields(expectations)
    return expectations


def _penalize_disagreement(scores: list[float]) -> float:
    """Convert agreement scores into disagreement penalties."""

    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("scores passed to _penalize_disagreement must be in [0, 1]")
    return _mean_loss([1.0 - score for score in scores])


def _mean_loss(values: list[float]) -> float:
    """Return zero for empty inputs because an absent term contributes zero loss."""

    if not values:
        return 0.0
    return sum(values) / len(values)


def _validate_probability_fields(expectations: dict[str, Any]) -> None:
    """Validate documented probability fields use [0, 1] interval guidance."""

    probability_fields = (
        "invariant_scores",
        "policy_match_scores",
        "abstention_targets",
        "abstention_predictions",
    )
    for field_name in probability_fields:
        descriptor = str(expectations.get(field_name, ""))
        if "[0, 1]" not in descriptor:
            raise ValueError(
                f"{field_name} must document probabilities in [0, 1], got: {descriptor!r}"
            )


__all__ = [
    "LossBreakdown",
    "SemanticBatch",
    "abstention_calibration_loss",
    "auxiliary_supervision_loss",
    "batch_format_expectations",
    "compute_loss_breakdown",
    "contrastive_separation_loss",
    "invariance_loss",
    "policy_consistency_loss",
]
