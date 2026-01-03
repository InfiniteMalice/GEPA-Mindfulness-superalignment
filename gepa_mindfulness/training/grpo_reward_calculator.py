"""Hallucination-aware reward calculation for GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..core.abstention import AbstentionAssessment
from ..core.paraconsistent import ParaconsistentTruthValue
from ..core.rewards import RewardSignal, RewardWeights
from .configs import HallucinationPenaltyConfig
from .grpo_types import GRPOGroupSample


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _score_contemplative_principles(summary: Mapping[str, str]) -> float:
    path_1 = summary.get("path_1_reasoning", "")
    path_2 = summary.get("path_2_reasoning", "")
    recommendation = summary.get("recommendation", "")

    mindfulness = 1.0 if _contains_any(path_1, ["consider", "mindful", "reflection"]) else 0.6
    empathy = 1.0 if _contains_any(path_1 + path_2, ["care", "harm", "support"]) else 0.5
    comparison = summary.get("comparison", "")
    perspective = 1.0 if (path_1 and path_2) or comparison else 0.4
    agency = 1.0 if _contains_any(recommendation, ["should", "take", "act"]) else 0.5

    return (mindfulness + empathy + perspective + agency) / 4.0


def _imperative_truth(response: str) -> ParaconsistentTruthValue:
    support = 0.5
    opposition = 0.0
    lower = response.lower()

    if "reduce harm" in lower or "stay safe" in lower:
        support = max(support, 0.8)
    if "profit" in lower or "growth" in lower:
        support = max(support, 0.7)
    if "learn" in lower or "research" in lower:
        support = max(support, 0.7)
    if "ignore" in lower or "dangerous" in lower:
        opposition = max(opposition, 0.4)
    return ParaconsistentTruthValue.from_support_opposition(support, opposition)


@dataclass
class RewardComputation:
    reward: float
    signal: RewardSignal
    category: str
    confidence: float


class GRPORewardCalculator:
    """Blend GEPA and hallucination rewards for GRPO."""

    def __init__(
        self,
        weights: RewardWeights,
        hallucination: HallucinationPenaltyConfig,
    ) -> None:
        self.weights = weights
        self.hallucination = hallucination

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
        response = sample.response
        summary = sample.trace.summary if sample.trace else {}
        abstention = sample.trace.abstention if sample.trace else None
        confidence = sample.trace.confidence_hint if sample.trace else 0.6

        if abstention and (abstention.is_genuine or abstention.is_lazy):
            category = "abstention"
        else:
            category = self._classify_response(response)
        hallucination_term = self._hallucination_reward(category, confidence, abstention)
        gepa_score = _score_contemplative_principles(summary)
        honesty_reward = self._honesty_bonus(abstention, confidence)
        task_success = 1.0 if category == "correct" else 0.0

        signal = RewardSignal(
            task_success=task_success,
            gepa_score=gepa_score,
            honesty_reward=honesty_reward,
            hallucination_score=hallucination_term,
            imperatives_truth=_imperative_truth(response),
        )
        reward = signal.combined(self.weights)
        return RewardComputation(
            reward=reward,
            signal=signal,
            category=category,
            confidence=confidence,
        )

    def _classify_response(self, response: str) -> str:
        lowered = response.lower().strip()
        if not lowered:
            return "abstention"
        if any(marker in lowered for marker in ["i don't know", "cannot answer", "abstain"]):
            return "abstention"
        if "correct" in lowered or "yes" in lowered or "no" in lowered:
            # Without explicit answer key, treat decisive statements as correct-ish
            return "correct"
        if _contains_any(lowered, ["unsafe", "wrong", "avoid"]):
            return "correct"
        return "wrong"

    def _hallucination_reward(
        self,
        category: str,
        confidence: float,
        abstention: AbstentionAssessment | None,
    ) -> float:
        cfg = self.hallucination
        if category == "correct":
            return 1.0
        if category == "abstention":
            if abstention and abstention.is_genuine():
                return cfg.appropriate_abstention_reward
            if abstention and abstention.is_lazy():
                return cfg.lazy_abstention_penalty
            return cfg.lazy_abstention_penalty / 2
        assert category == "wrong"
        if confidence > cfg.confidence_threshold:
            return cfg.confident_wrong_penalty
        return cfg.uncertain_wrong_penalty

    def _honesty_bonus(
        self,
        abstention: AbstentionAssessment | None,
        confidence: float,
    ) -> float:
        bonus = 0.0
        if abstention and abstention.is_genuine():
            bonus += 0.3
        if confidence < self.hallucination.confidence_threshold:
            bonus += 0.1
        return bonus


__all__ = [
    "GRPORewardCalculator",
    "RewardComputation",
]
