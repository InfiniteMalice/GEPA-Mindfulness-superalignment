"""Reward calculation utilities shared by PPO and GRPO trainers."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .abstention import (
    ABSTAIN_OUTPUT,
    AbstentionAssessment,
    AbstentionQuality,
    assess_abstention_quality,
)
from .paraconsistent import ParaconsistentTruthValue, dialetheic_and

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
        try:
            normalized = weights.normalized()
        except ValueError:
            normalized = RewardWeights(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25)
        calculator = GEPARewardCalculator(
            weights=normalized,
            hallucination=HallucinationConfig(
                confidence_threshold=0.75,
                confident_wrong_penalty=self.hallucination_score,
                uncertain_wrong_penalty=self.hallucination_score,
                appropriate_abstention_reward=0.0,
                lazy_abstention_penalty=0.0,
            ),
        )
        breakdown = calculator.compute_reward(
            response="",
            reference_answers=None,
            gepa_scores={"aggregate": self.gepa_score},
            imperatives={
                "aggregate": {
                    "support": max(0.0, min(1.0, self.imperatives_truth.support)),
                    "opposition": max(0.0, min(1.0, self.imperatives_truth.opposition)),
                }
            },
            confidence=1.0,
            trace_summary={},
        )
        base = (
            normalized.alpha * self.task_success
            + normalized.beta * self.gepa_score
            + normalized.gamma * self.honesty_reward
            - normalized.delta * self.hallucination_score
        )
        return base + normalized.gamma * breakdown.paraconsistent_truth


class GEPARewardCalculator:
    """Blend GEPA reward components into a scalar value."""

    def __init__(
        self,
        *,
        weights: RewardWeights,
        hallucination: HallucinationConfig,
        abstention_threshold: float = 0.75,
    ) -> None:
        self.weights = weights.normalized()
        self.hallucination = hallucination
        self.abstention_threshold = abstention_threshold

    def compute_reward(
        self,
        *,
        response: str,
        reference_answers: Sequence[str] | str | None,
        gepa_scores: Mapping[str, float] | None,
        imperatives: Mapping[str, Mapping[str, float]] | None,
        confidence: float,
        trace_summary: Mapping[str, str] | None,
        abstention: AbstentionAssessment | None = None,
    ) -> RewardBreakdown:
        """Compute the GEPA reward for a model response.

        Args:
            response: Generated response string.
            reference_answers: Canonical answers or evaluation targets.
            gepa_scores: Scores for contemplative principles and imperatives.
            imperatives: Mapping of imperative name to support/opposition pairs.
            confidence: Mean token confidence for the response.
            trace_summary: Circuit Tracer summary dictionary.
            abstention: Assessment of abstention quality if available.

        Returns:
            RewardBreakdown: Weighted reward components with metadata.
        """

        normalized_answers = self._normalise_reference(reference_answers)
        is_correct = self._matches_reference(response, normalized_answers)
        abstained = self._detect_abstention(response, confidence)

        if abstained and abstention is None:
            abstention = assess_abstention_quality(trace_summary or {}, [response])

        task_success = self._task_reward(is_correct, abstained)
        gepa_alignment = self._gepa_alignment_score(gepa_scores)
        paraconsistent_truth = self._paraconsistent_truth(imperatives)
        honesty = self._honesty_reward(
            abstained=abstained,
            confidence=confidence,
            trace_summary=trace_summary,
            assessment=abstention,
        )
        hallucination = self._hallucination_reward(
            is_correct=is_correct,
            abstained=abstained,
            confidence=confidence,
            assessment=abstention,
        )

        total = (
            self.weights.alpha * task_success
            + self.weights.beta * gepa_alignment
            + self.weights.gamma * (honesty + paraconsistent_truth)
            + self.weights.delta * hallucination
        )

        return RewardBreakdown(
            task_success=task_success,
            gepa_alignment=gepa_alignment,
            honesty=honesty,
            hallucination=hallucination,
            paraconsistent_truth=paraconsistent_truth,
            total=total,
            abstention_quality=abstention.quality if abstention else None,
        )

    def _task_reward(self, is_correct: bool, abstained: bool) -> float:
        if is_correct:
            return 1.0
        if abstained:
            return 0.0
        return -1.0

    def _gepa_alignment_score(self, scores: Mapping[str, float] | None) -> float:
        if not scores:
            return 0.0
        values = [max(0.0, min(1.0, value)) for value in scores.values()]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _paraconsistent_truth(self, imperatives: Mapping[str, Mapping[str, float]] | None) -> float:
        if not imperatives:
            return 0.0
        aggregate: ParaconsistentTruthValue | None = None
        for payload in imperatives.values():
            support = max(0.0, min(1.0, payload.get("support", 0.0)))
            opposition = max(0.0, min(1.0, payload.get("opposition", 0.0)))
            truth = ParaconsistentTruthValue.from_support_opposition(support, opposition)
            aggregate = truth if aggregate is None else dialetheic_and(aggregate, truth)
        return aggregate.resolve() if aggregate else 0.0

    def _honesty_reward(
        self,
        *,
        abstained: bool,
        confidence: float,
        trace_summary: Mapping[str, str] | None,
        assessment: AbstentionAssessment | None,
    ) -> float:
        if not abstained:
            return max(0.0, 1.0 - abs(confidence - self.abstention_threshold))
        if assessment and assessment.quality is AbstentionQuality.GENUINE:
            return 1.0
        if assessment and assessment.quality is AbstentionQuality.LAZY:
            return -1.0
        evidence = 1.0 if trace_summary and trace_summary.get("evidence") else 0.0
        return evidence - 0.5

    def _hallucination_reward(
        self,
        *,
        is_correct: bool,
        abstained: bool,
        confidence: float,
        assessment: AbstentionAssessment | None,
    ) -> float:
        if abstained:
            if assessment and assessment.quality is AbstentionQuality.GENUINE:
                return self.hallucination.appropriate_abstention_reward
            if assessment and assessment.quality is AbstentionQuality.LAZY:
                return self.hallucination.lazy_abstention_penalty
            return self.hallucination.lazy_abstention_penalty / 2
        outcome = self.hallucination.classify(confidence=confidence, is_correct=is_correct)
        if outcome == "correct":
            return 1.0
        if outcome == "confident_wrong":
            return self.hallucination.confident_wrong_penalty
        return self.hallucination.uncertain_wrong_penalty

    def _detect_abstention(self, response: str, confidence: float) -> bool:
        candidate = response.lower().strip()
        if not candidate:
            return True
        if any(marker in candidate for marker in _ABSTENTION_MARKERS):
            return True
        if ABSTAIN_OUTPUT.lower() in candidate:
            return True
        return confidence < self.abstention_threshold * 0.6

    @staticmethod
    def _normalise_reference(
        answers: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if answers is None:
            return tuple()
        if isinstance(answers, str):
            return (answers.lower().strip(),)
        return tuple(answer.lower().strip() for answer in answers)

    @staticmethod
    def _matches_reference(response: str, answers: Iterable[str]) -> bool:
        normalized = response.lower().strip()
        return any(normalized == answer for answer in answers)


__all__ = [
    "GEPARewardCalculator",
    "HallucinationConfig",
    "RewardBreakdown",
    "RewardSignal",
    "RewardWeights",
]
