"""Reward calculation utilities shared by PPO and GRPO trainers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .abstention import (
    AbstentionAssessment,
    AbstentionQuality,
)
from .epistemic_process import EpistemicProcessAssessment
from .paraconsistent import ParaconsistentTruthValue


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
        if not math.isfinite(self.total):
            raise ValueError("reward weight total mass must be finite")
        if self.total <= 0.0:
            raise ValueError("reward weights must have positive mass")

    @property
    def total(self) -> float:
        """Return the aggregate weight magnitude."""

        return self.alpha + self.beta + self.gamma + self.delta

    @property
    def task_success(self) -> float:
        """Alias for ``alpha`` used by legacy callers."""

        return self.alpha

    @property
    def gepa_alignment(self) -> float:
        """Alias for ``beta`` used by legacy callers."""

        return self.beta

    @property
    def honesty_trace(self) -> float:
        """Alias for ``gamma`` used by legacy callers."""

        return self.gamma

    @property
    def hallucination_penalty(self) -> float:
        """Alias for ``delta`` used by legacy callers."""

        return self.delta

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
    """Structured reward report.

    ``GEPARewardCalculator`` emits ``honesty`` and ``epistemic_process`` with the same verified
    process value. Direct legacy construction keeps the fields independent for compatibility.
    """

    task_success: float
    gepa_alignment: float
    honesty: float
    hallucination: float
    paraconsistent_truth: float
    total: float
    abstention_quality: AbstentionQuality | None
    epistemic_process: float = 0.0


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


class GEPARewardCalculator:
    """Compute reward signals by blending alignment and honesty metrics."""

    def __init__(
        self,
        *,
        weights: RewardWeights,
        hallucination: HallucinationConfig,
        abstention_threshold: float | None = None,
    ) -> None:
        self.weights = weights.normalized()
        self.hallucination = hallucination
        self.abstention_threshold = (
            abstention_threshold
            if abstention_threshold is not None
            else hallucination.confidence_threshold
        )

    def compute_reward(
        self,
        *,
        response: str,
        reference_answers: Sequence[str] | str | None,
        gepa_scores: Mapping[str, float] | None,
        imperatives: Mapping[str, Mapping[str, float]] | None,
        confidence: float,
        trace_summary: Mapping[str, str],
        abstention: AbstentionAssessment | None = None,
        epistemic_process: EpistemicProcessAssessment | None = None,
    ) -> RewardBreakdown:
        references = self._normalise_references(reference_answers)
        response_normalised = response.strip().lower()
        is_correct = any(
            response_normalised == reference.strip().lower() for reference in references
        )

        assessment = abstention

        abstention_quality = assessment.quality if assessment is not None else None
        hallucination_signal = self._hallucination_signal(
            is_correct=is_correct,
            confidence=confidence,
            assessment=assessment,
        )

        task_success = 1.0 if is_correct else 0.0
        gepa_alignment = self._gepa_alignment_score(gepa_scores)
        epistemic_process_score = self._epistemic_process_signal(epistemic_process)
        honesty = epistemic_process_score
        paraconsistent_truth = self._paraconsistent_signal(imperatives)
        epistemic_process_total = epistemic_process_score + paraconsistent_truth.resolve()

        total = (
            self.weights.task_success * task_success
            + self.weights.gepa_alignment * gepa_alignment
            + self.weights.honesty_trace * epistemic_process_total
            + self.weights.hallucination_penalty * hallucination_signal
        )

        return RewardBreakdown(
            task_success=task_success,
            gepa_alignment=gepa_alignment,
            honesty=honesty,
            hallucination=hallucination_signal,
            paraconsistent_truth=paraconsistent_truth.resolve(),
            total=total,
            abstention_quality=abstention_quality,
            epistemic_process=epistemic_process_score,
        )

    @staticmethod
    def _normalise_references(
        reference_answers: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if reference_answers is None:
            return ()
        if isinstance(reference_answers, str):
            return (reference_answers,)
        return tuple(reference_answers)

    def _hallucination_signal(
        self,
        *,
        is_correct: bool,
        confidence: float,
        assessment: AbstentionAssessment | None,
    ) -> float:
        if assessment is not None:
            if assessment.quality is AbstentionQuality.GENUINE:
                return self.hallucination.appropriate_abstention_reward
            if assessment.quality is AbstentionQuality.LAZY:
                return self.hallucination.lazy_abstention_penalty
        if is_correct:
            return 0.0
        if confidence >= self.hallucination.confidence_threshold:
            return self.hallucination.confident_wrong_penalty
        return self.hallucination.uncertain_wrong_penalty

    @staticmethod
    def _gepa_alignment_score(scores: Mapping[str, float] | None) -> float:
        if not scores:
            return 0.0
        values = [float(value) for value in scores.values()]
        return float(sum(values) / len(values))

    @staticmethod
    def _epistemic_process_signal(
        epistemic_process: EpistemicProcessAssessment | None,
    ) -> float:
        if epistemic_process is None:
            return 0.0
        return epistemic_process.optimizer_score()

    @staticmethod
    def _paraconsistent_signal(
        imperatives: Mapping[str, Mapping[str, float]] | None,
    ) -> ParaconsistentTruthValue:
        if not imperatives:
            return ParaconsistentTruthValue.from_support_opposition(0.0, 0.0)
        supports: list[float] = []
        oppositions: list[float] = []
        for payload in imperatives.values():
            supports.append(float(payload.get("support", 0.0)))
            oppositions.append(float(payload.get("opposition", 0.0)))
        support = sum(supports) / len(supports)
        opposition = sum(oppositions) / len(oppositions)
        return ParaconsistentTruthValue.from_support_opposition(support, opposition)


__all__ = [
    "GEPARewardCalculator",
    "HallucinationConfig",
    "RewardBreakdown",
    "RewardSignal",
    "RewardWeights",
]
