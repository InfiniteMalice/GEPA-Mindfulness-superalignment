"""Group Relative Policy Optimisation trainer implemented with pure Python math."""

from __future__ import annotations

import math
import random
from typing import List, Sequence

from .base_trainer import BaseTrainer, GeneratedResponse
from .config import GRPOConfig
from .dataloader import DatasetExample


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class GRPOTrainer(BaseTrainer):
    """Optimise Bernoulli policies using group-relative advantages."""

    def __init__(self, config: GRPOConfig) -> None:
        super().__init__(config)
        self.config = config
        self._logits: dict[str, float] = {}
        self._reference_probs: dict[str, float] = {}
        self._random = random.Random(0)

    def train(self) -> None:
        for step in range(self.config.max_steps):
            batch = self.dataset.sample_batch(self.config.batch_size)
            prompts = [example.prompt for example in batch]
            references = [example.references for example in batch]
            gepa_scores = [example.gepa_scores for example in batch]
            imperatives = [example.imperatives for example in batch]

            grouped_responses = [self._generate_group(example) for example in batch]
            grouped_rewards, _ = self.compute_batch_rewards(
                prompts,
                grouped_responses,
                references=references,
                gepa_scores=gepa_scores,
                imperatives=imperatives,
            )
            advantages = self._compute_advantages(grouped_rewards)
            self._apply_policy_update(batch, grouped_responses, advantages)
            self.global_step = step + 1
            if (step + 1) % self.config.batch_size == 0:
                self.save_checkpoint(step + 1)
        self.save_summary()

    def _generate_group(self, example: DatasetExample) -> List[GeneratedResponse]:
        logit = self._logits.setdefault(example.prompt, 0.0)
        ref_prob = self._reference_probs.setdefault(example.prompt, _sigmoid(logit))
        reference_answers = list(example.references or [])
        reference_available = bool(reference_answers)
        correct_text = reference_answers[0] if reference_available else "I don't know"
        abstain_text = "I don't know"
        responses: List[GeneratedResponse] = []
        for idx in range(self.config.group_size):
            prob = _sigmoid(logit)
            if idx == 0:
                action = 1.0
            elif idx == 1:
                action = 0.0
            else:
                action = 1.0 if self._random.random() < prob else 0.0
            text = correct_text if action == 1.0 else abstain_text
            log_prob = math.log(prob if action == 1.0 else max(1e-6, 1.0 - prob))
            ref_log_prob = math.log(ref_prob if action == 1.0 else max(1e-6, 1.0 - ref_prob))
            responses.append(
                GeneratedResponse(
                    text=text,
                    log_probs=[log_prob],
                    mask=[1],
                    policy_log_prob=log_prob,
                    reference_log_prob=ref_log_prob,
                    metadata={
                        "action": action,
                        "probability": prob,
                        "reference_available": reference_available,
                    },
                )
            )
        return responses

    def _compute_advantages(
        self, grouped_rewards: Sequence[Sequence[float]]
    ) -> Sequence[Sequence[float]]:
        advantages: List[List[float]] = []
        for rewards in grouped_rewards:
            if not rewards:
                advantages.append([])
                continue
            mean_reward = sum(rewards) / len(rewards)
            advantages.append([reward - mean_reward for reward in rewards])
        return advantages

    def _apply_policy_update(
        self,
        batch: Sequence[DatasetExample],
        responses: Sequence[Sequence[GeneratedResponse]],
        advantages: Sequence[Sequence[float]],
    ) -> None:
        for example, group, group_advantages in zip(batch, responses, advantages):
            if not group or not group_advantages:
                continue
            if example.prompt not in self._logits:
                self._logits[example.prompt] = 0.0
            logit = self._logits[example.prompt]
            ref_prob = self._reference_probs.setdefault(example.prompt, _sigmoid(logit))
            prob = _sigmoid(logit)
            grad_sum = 0.0
            for generated, advantage in zip(group, group_advantages):
                metadata = generated.metadata or {}
                action = metadata.get("action", 0.0)
                grad_sum += advantage * (action - prob)
            kl_grad = 2 * (prob - ref_prob) * prob * (1.0 - prob)
            logit += self.config.learning_rate * (grad_sum - self.config.kl_coef * kl_grad)
            updated_prob = _sigmoid(logit)
            for generated in group:
                metadata = dict(generated.metadata or {})
                metadata.setdefault("action", metadata.get("action", 0.0))
                metadata["probability"] = updated_prob
                generated.metadata = metadata
            self._logits[example.prompt] = logit


__all__ = ["GRPOTrainer"]
