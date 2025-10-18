"""Implementation of Group Relative Policy Optimisation for GEPA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from ..core.circuit_tracer_adapter import CircuitTracerAdapter
from ..core.rewards import RewardWeights
from ..core.tracing import CircuitTracerLogger
from .configs import GRPOConfig
from .grpo_reward_calculator import GRPORewardCalculator
from .grpo_types import GRPOGroupSample

LOGGER = logging.getLogger(__name__)


@dataclass
class GRPOBatchStats:
    prompt: str
    mean_reward: float
    advantages: List[float]
    categories: List[str]
    confidences: List[float]


@dataclass
class GRPOTrainingSummary:
    steps: int = 0
    total_reward: float = 0.0
    batches: List[GRPOBatchStats] = field(default_factory=list)

    def record(self, stats: GRPOBatchStats) -> None:
        self.steps += 1
        self.total_reward += stats.mean_reward
        self.batches.append(stats)

    def mean_reward(self) -> float:
        if not self.batches:
            return 0.0
        return self.total_reward / len(self.batches)


class GRPOTrainer:
    """Trainer performing GRPO updates with GEPA rewards."""

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer,
        config: GRPOConfig,
        reward_weights: RewardWeights,
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_weights = reward_weights
        self.device = device
        self.model.to(device)
        self.ref_model.to(device)
        self.ref_model.eval()

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is None:
                raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
            self.tokenizer.pad_token_id = eos

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.logger = CircuitTracerLogger()
        self.tracer_adapter = CircuitTracerAdapter(
            self.logger,
            trace_frequency=config.circuit_tracer.trace_frequency,
            trace_strategy=config.circuit_tracer.trace_strategy,
        )
        self.reward_calculator = GRPORewardCalculator(reward_weights, config.hallucination)

    def train_epoch(
        self, prompts: Sequence[str], *, batch_size: int | None = None
    ) -> GRPOTrainingSummary:
        summary = GRPOTrainingSummary()
        batch_size = batch_size or self.config.batch_size
        iterator = self._batch_iterator(prompts, batch_size)

        for batch_prompts in iterator:
            batch_stats = self._train_batch(batch_prompts)
            summary.record(batch_stats)
        return summary

    def _batch_iterator(self, prompts: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
        for start in range(0, len(prompts), batch_size):
            yield prompts[start : start + batch_size]

    def _train_batch(self, prompts: Sequence[str]) -> GRPOBatchStats:
        total_reward = 0.0
        total_advantages: list[float] = []
        total_categories: list[str] = []
        total_confidences: list[float] = []

        accumulate = max(1, self.config.gradient_accumulation_steps)
        self.optimizer.zero_grad()

        for idx, prompt in enumerate(prompts):
            group = self._collect_group(prompt)
            computations = self.reward_calculator.score_group(group)
            loss = self._compute_group_loss(group) / accumulate
            loss.backward()

            mean_reward = sum(comp.reward for comp in computations) / max(1, len(computations))
            total_reward += mean_reward
            total_advantages.extend(sample.advantage for sample in group.samples)
            total_categories.extend(comp.category for comp in computations)
            total_confidences.extend(comp.confidence for comp in computations)

            if (idx + 1) % accumulate == 0 or idx == len(prompts) - 1:
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        prompt_list = list(prompts)
        description = GRPOBatchStats(
            prompt=prompt_list[0] if prompt_list else "",
            mean_reward=total_reward / max(1, len(prompts)),
            advantages=total_advantages,
            categories=total_categories,
            confidences=total_confidences,
        )
        LOGGER.debug("GRPO batch mean reward %.3f", description.mean_reward)
        return description

    def _collect_group(self, prompt: str) -> GRPOGroupSample:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self.device)
        prompt_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.sampling_temperature,
                num_return_sequences=self.config.group_size,
            )

        responses = []
        tokens_per_sample: list[torch.Tensor] = []
        for sequence in outputs:
            response_tokens = sequence[prompt_len:]
            tokens_per_sample.append(sequence)
            responses.append(self.tokenizer.decode(response_tokens, skip_special_tokens=True))

        analyses = self.tracer_adapter.analyse_group(prompt, responses)

        policy_log_probs = self._sequence_log_probs(
            self.model, tokens_per_sample, prompt_len, requires_grad=True
        )
        with torch.no_grad():
            ref_log_probs = self._sequence_log_probs(
                self.ref_model, tokens_per_sample, prompt_len, requires_grad=False
            )

        group = GRPOGroupSample(prompt=prompt)
        for idx, response in enumerate(responses):
            sample = GRPOGroupSample.Sample(
                response=response,
                tokens=tokens_per_sample[idx].tolist(),
                log_prob=policy_log_probs[idx],
                ref_log_prob=ref_log_probs[idx],
                trace=analyses[idx],
            )
            group.samples.append(sample)
        return group

    def _sequence_log_probs(
        self,
        model: torch.nn.Module,
        sequences: Sequence[torch.Tensor],
        prompt_len: int,
        *,
        requires_grad: bool,
    ) -> List[torch.Tensor]:
        stacked = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)
        input_tokens = stacked[:, :-1]
        target_tokens = stacked[:, 1:]
        attention_mask = (input_tokens != self.tokenizer.pad_token_id).long()
        outputs = model(
            input_tokens,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        seq_len = gathered.shape[1]
        mask = torch.zeros_like(gathered)
        mask[:, prompt_len - 1 : seq_len] = 1
        masked = gathered * mask
        log_prob_per_seq = masked.sum(dim=1)
        if not requires_grad:
            log_prob_per_seq = log_prob_per_seq.detach()
        return list(log_prob_per_seq)

    def _compute_group_loss(self, group: GRPOGroupSample) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for sample in group.samples:
            advantage = torch.tensor(sample.advantage, device=self.device, dtype=torch.float32)
            policy_log_prob = sample.log_prob
            ref_log_prob = sample.ref_log_prob
            kl = policy_log_prob - ref_log_prob
            losses.append(-advantage.detach() * policy_log_prob + self.config.kl_coef * kl)
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()


__all__ = [
    "GRPOTrainer",
    "GRPOTrainingSummary",
    "GRPOBatchStats",
]
