"""Simplified Proximal Policy Optimisation trainer without external dependencies."""

from __future__ import annotations

import math
import random
from typing import Sequence

from mindful_trace_gepa.train.grn import build_grn
from mindful_trace_gepa.utils.imports import optional_import

from .base_trainer import BaseTrainer, GeneratedResponse
from .config import PPOConfig
from .dataloader import DatasetExample

torch = optional_import("torch")


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
        self.policy_grn = build_grn(config.policy_grn)

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

    def _compute_advantages(self, grouped_rewards: Sequence[Sequence[float]]) -> list[list[float]]:
        """Estimate advantages using a lightweight GAE-style accumulator."""

        advantages: list[list[float]] = []
        gamma = 1.0
        lam = self.config.gae_lambda

        for rewards in grouped_rewards:
            if not rewards:
                advantages.append([])
                continue

            running_advantage = 0.0
            next_value = 0.0
            group_advantages: list[float] = []
            for reward in reversed(rewards):
                delta = reward - next_value
                running_advantage = delta + gamma * lam * running_advantage
                group_advantages.append(running_advantage)
                next_value = reward
            group_advantages.reverse()
            advantages.append(group_advantages)

        return advantages

    def _generate_response(self, example: DatasetExample) -> GeneratedResponse:
        logit = self._policy_logits.setdefault(example.prompt, 0.0)
        prob = _sigmoid(self._apply_policy_grn(logit))
        action = 1.0 if self._random.random() < prob else 0.0
        references = list(example.references) if example.references else []
        if action == 1.0 and references:
            text = references[0]
            reference_used = True
        elif action == 1.0:
            text = self._synthesise_response(example.prompt)
            reference_used = False
        else:
            text = "I don't know"
            reference_used = False
        log_prob = math.log(prob if action == 1.0 else max(1e-6, 1.0 - prob))
        return GeneratedResponse(
            text=text,
            log_probs=[log_prob],
            mask=[1],
            policy_log_prob=log_prob,
            metadata={
                "action": action,
                "probability": prob,
                "reference_used": reference_used,
            },
        )

    @staticmethod
    def _synthesise_response(prompt: str) -> str:
        """Generate a lightweight fallback response when references are missing."""

        return (
            "Here is a mindful perspective: "
            f"{prompt} Mindfulness encourages thoughtful reflection, steady breathing, "
            "and compassionate awareness of the present moment."
        )

    def _apply_updates(
        self,
        batch: Sequence[DatasetExample],
        responses: Sequence[GeneratedResponse],
        rewards: Sequence[float],
    ) -> None:
        for example, generated, reward in zip(batch, responses, rewards):
            logit = self._policy_logits.setdefault(example.prompt, 0.0)
            prob = _sigmoid(self._apply_policy_grn(logit))
            metadata = generated.metadata or {}
            action = metadata.get("action", 0.0)
            advantage = reward - self._value_estimates.setdefault(example.prompt, 0.0)
            # GRN normalises logits for sampling; updates remain on raw logits heuristically.
            grad = advantage * (action - prob)
            logit += self.config.learning_rate * grad
            self._policy_logits[example.prompt] = logit
            value = self._value_estimates[example.prompt]
            value += self.config.learning_rate * self.config.value_coef * (reward - value)
            self._value_estimates[example.prompt] = value

    def _apply_policy_grn(self, logit: float) -> float:
        if self.policy_grn is None or torch is None:
            return logit
        with torch.no_grad():
            tensor = torch.tensor([[logit]], dtype=torch.float32)
            normalised = self.policy_grn(tensor).squeeze(0)
        return float(normalised.item())


__all__ = ["PPOTrainer"]
