"""Verified-contract reward calculation for legacy HF-style GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.paraconsistent import ParaconsistentTruthValue
from ..core.rewards import (
    GEPARewardCalculator,
    HallucinationConfig,
    RewardBreakdown,
    RewardSignal,
    RewardWeights,
)
from .configs import HallucinationPenaltyConfig
from .grpo_types import GRPOGroupSample


@dataclass
class RewardComputation:
    """Legacy GRPO result whose confidence field is diagnostic-only."""

    reward: float
    signal: RewardSignal
    category: str
    confidence: float


class GRPORewardCalculator:
    """Apply the canonical reward contract without interpreting generated trace prose."""

    def __init__(
        self,
        weights: RewardWeights,
        hallucination: HallucinationPenaltyConfig,
    ) -> None:
        self.weights = weights.normalized()
        self.hallucination = hallucination
        self._calculator = GEPARewardCalculator(
            weights=self.weights,
            hallucination=HallucinationConfig(
                confidence_threshold=hallucination.confidence_threshold,
                confident_wrong_penalty=hallucination.confident_wrong_penalty,
                uncertain_wrong_penalty=hallucination.uncertain_wrong_penalty,
                appropriate_abstention_reward=hallucination.appropriate_abstention_reward,
                lazy_abstention_penalty=hallucination.lazy_abstention_penalty,
            ),
        )

    def score_group(
        self,
        group: GRPOGroupSample,
    ) -> Sequence[RewardComputation]:
        computations: list[RewardComputation] = []
        mean_reward = 0.0

        for sample in group.samples:
            comp = self._score_single(sample)
            computations.append(comp)
            mean_reward += comp.reward

        if computations:
            mean_reward /= len(computations)

        for idx, comp in enumerate(computations):
            advantage = comp.reward - mean_reward
            group.samples[idx].advantage = advantage
            group.samples[idx].reward = comp.reward
        return computations

    def _score_single(self, sample: GRPOGroupSample.Sample) -> RewardComputation:
        summary = sample.trace.summary if sample.trace else {}
        diagnostic_confidence = sample.trace.confidence_hint if sample.trace else 0.6
        optimizer_confidence = sample.confidence if sample.confidence is not None else 0.0
        breakdown = self._calculator.compute_reward(
            response=sample.response,
            reference_answers=sample.reference_answers,
            gepa_scores=None,
            imperatives=None,
            confidence=optimizer_confidence,
            trace_summary=summary,
            abstention=None,
            epistemic_process=sample.epistemic_process,
        )

        signal = RewardSignal(
            task_success=breakdown.task_success,
            gepa_score=breakdown.gepa_alignment,
            honesty_reward=breakdown.epistemic_process,
            hallucination_score=breakdown.hallucination,
            imperatives_truth=ParaconsistentTruthValue.from_support_opposition(0.0, 0.0),
        )
        return RewardComputation(
            reward=breakdown.total,
            signal=signal,
            category=self._category(breakdown, sample.reference_answers),
            confidence=diagnostic_confidence,
        )

    @staticmethod
    def _category(
        breakdown: RewardBreakdown,
        reference_answers: Sequence[str] | str | None,
    ) -> str:
        if not reference_answers:
            return "unverified"
        if breakdown.task_success == 1.0:
            return "correct"
        return "wrong"


__all__ = [
    "GRPORewardCalculator",
    "RewardComputation",
]
