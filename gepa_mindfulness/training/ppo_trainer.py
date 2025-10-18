"""Simplified Proximal Policy Optimisation trainer without external dependencies."""

from __future__ import annotations

import math
import random
from typing import Sequence

from .base_trainer import BaseTrainer, GeneratedResponse
from .config import PPOConfig
from .dataloader import DatasetExample


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class PPOTrainer(BaseTrainer):
    """Optimise Bernoulli policies using a lightweight PPO-style update."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__(config)
        self.config = config
        self._policy_logits: dict[str, float] = {}
        self._value_estimates: dict[str, float] = {}
        self._random = random.Random(0)

    def train(self) -> None:
        for step in range(self.config.max_steps):
            batch = self.dataset.sample_batch(self.config.batch_size)
            prompts = [example.prompt for example in batch]
            references = [example.references for example in batch]
            gepa_scores = [example.gepa_scores for example in batch]
            imperatives = [example.imperatives for example in batch]

            responses = [self._generate_response(example) for example in batch]
            grouped = [[response] for response in responses]
            grouped_rewards, _ = self.compute_batch_rewards(
                prompts,
                grouped,
                references=references,
                gepa_scores=gepa_scores,
                imperatives=imperatives,
            )
            rewards = [group[0] for group in grouped_rewards]
            self._apply_updates(batch, responses, rewards)
            self.global_step = step + 1
            if (step + 1) % self.config.batch_size == 0:
                self.save_checkpoint(step + 1)
        self.save_summary()

    def _compute_advantages(
        self, grouped_rewards: Sequence[Sequence[float]]
    ) -> Sequence[Sequence[float]]:
        """Return identity advantages for compatibility with the base class."""

        return [list(group) for group in grouped_rewards]

    def _generate_response(self, example: DatasetExample) -> GeneratedResponse:
        logit = self._policy_logits.setdefault(example.prompt, 0.0)
        prob = _sigmoid(logit)
        action = 1.0 if self._random.random() < prob else 0.0
        reference_answers = example.references or []
        if action == 1.0 and reference_answers:
            text = reference_answers[0]
        else:
            text = "I don't know"
        log_prob = math.log(prob if action == 1.0 else max(1e-6, 1.0 - prob))
        return GeneratedResponse(
            text=text,
            log_probs=[log_prob],
            mask=[1],
            policy_log_prob=log_prob,
            metadata={
                "action": action,
                "probability": prob,
                "reference_available": bool(reference_answers),
            },
        )

    def _apply_updates(
        self,
        batch: Sequence[DatasetExample],
        responses: Sequence[GeneratedResponse],
        rewards: Sequence[float],
    ) -> None:
        for example, generated, reward in zip(batch, responses, rewards):
            logit = self._policy_logits.setdefault(example.prompt, 0.0)
            prob = _sigmoid(logit)
            metadata = generated.metadata or {}
            action = metadata.get("action", 0.0)
            advantage = reward - self._value_estimates.setdefault(example.prompt, 0.0)
            grad = advantage * (action - prob)
            logit += self.config.learning_rate * grad
            self._policy_logits[example.prompt] = logit
            value = self._value_estimates[example.prompt]
            value += self.config.learning_rate * self.config.value_coef * (reward - value)
            self._value_estimates[example.prompt] = value


__all__ = ["PPOTrainer"]
