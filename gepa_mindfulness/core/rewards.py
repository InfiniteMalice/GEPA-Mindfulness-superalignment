"""Reward calculation utilities shared by PPO and GRPO trainers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .abstention import AbstentionQuality
from .paraconsistent import ParaconsistentTruthValue

_ABSTENTION_MARKERS = (
    "i don't know",
    "i am not sure",
    "i'm not sure",
    "uncertain",
    "not certain",
    "unsure",
    "cannot answer",
    "no answer",
    "abstain",
)


@dataclass(frozen=True)
class RewardWeights:
    """Weighting coefficients used to blend the reward components."""

    alpha: float
    beta: float
    gamma: float
    delta: float

    def validate(self) -> None:
        """Validate that weights form a convex combination."""

        for name, value in {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
        }.items():
            if value < 0.0:
                raise ValueError(f"weight {name} must be non-negative")
        if self.total <= 0.0:
            raise ValueError("reward weights must have positive mass")

    @property
    def total(self) -> float:
        """Return the aggregate weight magnitude."""

        return self.alpha + self.beta + self.gamma + self.delta

    def normalized(self) -> "RewardWeights":
        """Return a normalised copy whose weights sum to one."""

        self.validate()
        if math.isclose(self.total, 1.0, rel_tol=1e-6):
            return self
        total = self.total
        return RewardWeights(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
            delta=self.delta / total,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, float]) -> "RewardWeights":
        return cls(
            alpha=float(payload.get("alpha", payload.get("task_success", 1.0))),
            beta=float(payload.get("beta", payload.get("gepa_alignment", 1.0))),
            gamma=float(payload.get("gamma", payload.get("honesty_trace", 1.0))),
            delta=float(payload.get("delta", payload.get("hallucination_penalty", 1.0))),
        )


@dataclass(frozen=True)
class HallucinationConfig:
    """Configuration describing hallucination penalties and rewards."""

    confidence_threshold: float
    confident_wrong_penalty: float
    uncertain_wrong_penalty: float
    appropriate_abstention_reward: float
    lazy_abstention_penalty: float

    def classify(self, *, confidence: float, is_correct: bool) -> str:
        """Return the outcome label for hallucination handling."""

        if is_correct:
            return "correct"
        if confidence >= self.confidence_threshold:
            return "confident_wrong"
        return "uncertain_wrong"


@dataclass(frozen=True)
class RewardBreakdown:
    """Structured report returned by :class:`GEPARewardCalculator`."""

    task_success: float
    gepa_alignment: float
    honesty: float
    hallucination: float
    paraconsistent_truth: float
    total: float
    abstention_quality: AbstentionQuality | None


@dataclass
class RewardSignal:
    """Compatibility wrapper used by legacy modules and tests."""

    task_success: float
    gepa_score: float
    honesty_reward: float
    hallucination_score: float
    imperatives_truth: ParaconsistentTruthValue

    def combined(self, weights: RewardWeights) -> float:
        paraconsistent_component = self.imperatives_truth.resolve()
        hallucination_component = weights.hallucination_penalty * self.hallucination_score
        reward = (
            weights.task_success * self.task_success
            + weights.gepa_alignment * self.gepa_score
            + weights.honesty_trace * (self.honesty_reward + paraconsistent_component)
            + hallucination_component
        )
        return reward
