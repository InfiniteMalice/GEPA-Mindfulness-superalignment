"""Lightweight GRPO trainer used by the test suite."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .base_trainer import BaseTrainer, GeneratedResponse
from .config import GRPOConfig
from .dataloader import DatasetExample


@dataclass
class GRPOTrainingStats:
    """Summary of a single GRPO optimisation step."""

    prompt: str
    rewards: Sequence[float]
    advantages: Sequence[float]
    confidences: Sequence[float]


class GRPOTrainer(BaseTrainer):
    """Minimal trainer implementing the GRPO interface exercised by tests."""

    def __init__(self, config: GRPOConfig, *, seed: int | None = None) -> None:
        super().__init__(config)
        self.random = random.Random(seed or 0)
        self._logits: dict[str, float] = {}
        self.training_history: list[GRPOTrainingStats] = []

    def train(self) -> None:
        """Run a lightweight training loop over the dataset."""

        steps = 0
        for batch in self.dataset.iter_batches(self.config.batch_size):
            prompts = [example.prompt for example in batch]
            groups = [self._generate_group(example) for example in batch]
            references = [example.references for example in batch]
            gepa_scores = [example.gepa_scores for example in batch]
            imperatives = [example.imperatives for example in batch]

            grouped_rewards, breakdowns = self.compute_batch_rewards(
                prompts,
                groups,
                references=references,
                gepa_scores=gepa_scores,
                imperatives=imperatives,
            )
            grouped_advantages = self._compute_advantages(grouped_rewards)
            self._apply_policy_update(batch, groups, grouped_advantages)

            for prompt, rewards, advantages, group in zip(
                prompts, grouped_rewards, grouped_advantages, groups
            ):
                confidences = [response.confidence() for response in group]
                self.training_history.append(
                    GRPOTrainingStats(
                        prompt=prompt,
                        rewards=rewards,
                        advantages=advantages,
                        confidences=confidences,
                    )
                )

            steps += 1
            self.global_step += 1
            if steps >= self.config.max_steps:
                break

        self.save_summary()

    def _compute_advantages(self, grouped_rewards: Sequence[Sequence[float]]) -> list[list[float]]:
        advantages: list[list[float]] = []
        for rewards in grouped_rewards:
            if not rewards:
                advantages.append([])
                continue
            mean_reward = sum(rewards) / len(rewards)
            advantages.append([reward - mean_reward for reward in rewards])
        return advantages

    def _generate_group(self, example: DatasetExample) -> list[GeneratedResponse]:
        group: list[GeneratedResponse] = []
        correct = example.references[0] if example.references else None
        base_probability = 1.0 / max(1, self.config.group_size)
        for idx in range(self.config.group_size):
            if correct is not None and idx == 0:
                text = correct
                probability = max(0.7, 1.0 - base_probability)
            else:
                text = self._fabricate_response(example.prompt, idx)
                probability = max(0.05, base_probability / 2)
            log_prob = math.log(probability)
            response = GeneratedResponse(
                text=text,
                log_probs=[log_prob],
                mask=[1],
                metadata={"index": float(idx)},
            )
            group.append(response)
        return group

    def _apply_policy_update(
        self,
        examples: Iterable[DatasetExample],
        groups: Sequence[Sequence[GeneratedResponse]],
        advantages: Sequence[Sequence[float]],
    ) -> None:
        for example, group_advantages in zip(examples, advantages):
            if not group_advantages:
                continue
            adjustment = max(group_advantages)
            if adjustment == 0.0:
                adjustment = -min(group_advantages)
            self._logits[example.prompt] = self._logits.get(example.prompt, 0.0) + adjustment

    def _fabricate_response(self, prompt: str, index: int) -> str:
        seed = hash((prompt, index, self.random.random()))
        self.random.seed(seed)
        template = [
            "I am not certain.",
            "It might be true.",
            "Further evidence is required.",
            "Consulting additional sources.",
        ]
        choice = self.random.choice(template)
        return f"{choice} ({index + 1})"


__all__ = ["GRPOTrainer", "GRPOTrainingStats"]
