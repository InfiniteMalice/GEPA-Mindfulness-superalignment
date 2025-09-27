"""Reward shaping utilities for PPO training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .paraconsistent import ParaconsistentTruthValue


@dataclass(frozen=True)
class RewardWeights:
    task_success: float
    gepa_alignment: float
    honesty_trace: float
    hallucination_penalty: float

    @classmethod
    def from_mapping(cls, weights: Dict[str, float]) -> "RewardWeights":
        return cls(
            task_success=weights.get("alpha", weights.get("task_success", 1.0)),
            gepa_alignment=weights.get("beta", weights.get("gepa_alignment", 1.0)),
            honesty_trace=weights.get("gamma", weights.get("honesty_trace", 1.0)),
            hallucination_penalty=weights.get("delta", weights.get("hallucination_penalty", 1.0)),
        )


@dataclass
class RewardSignal:
    task_success: float
    gepa_score: float
    honesty_reward: float
    hallucination_score: float
    imperatives_truth: ParaconsistentTruthValue

    def combined(self, weights: RewardWeights) -> float:
        paraconsistent_component = self.imperatives_truth.resolve()
        reward = (
            weights.task_success * self.task_success
            + weights.gepa_alignment * self.gepa_score
            + weights.honesty_trace * (self.honesty_reward + paraconsistent_component)
            - weights.hallucination_penalty * self.hallucination_score
        )
        return reward
