"""GRPO trainer facade with backward-compatible constructor signatures."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Sequence

from .base_trainer import BaseTrainer, GeneratedResponse
from .config import BaseTrainerConfig
from .dataloader import DatasetExample

LOGGER = logging.getLogger(__name__)


@dataclass
class GRPOTrainingStats:
    """Summary of a single synthetic GRPO optimisation step."""

    prompt: str
    rewards: Sequence[float]
    advantages: Sequence[float]
    confidences: Sequence[float]


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


class _ConfigGRPOTrainer(BaseTrainer):
    """Trainer used by the lightweight config-driven pipeline and CLI."""

    def __init__(self, config: BaseTrainerConfig, *, seed: int | None = None) -> None:
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

            grouped_rewards, _ = self.compute_batch_rewards(
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


class _LegacyGRPOTrainer:
    """Implementation of Group Relative Policy Optimisation for GEPA."""

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config,
        reward_weights,
        *,
        device,
    ) -> None:
        import torch
        from torch.nn.utils import clip_grad_norm_

        from ..core.circuit_tracer_adapter import CircuitTracerAdapter
        from ..core.rewards import RewardWeights
        from ..core.tracing import CircuitTracerLogger
        from .grpo_reward_calculator import GRPORewardCalculator
        from .grpo_types import GRPOGroupSample

        if not isinstance(reward_weights, RewardWeights):
            raise TypeError("reward_weights must be a RewardWeights instance")

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

        self._torch = torch
        self._clip_grad_norm = clip_grad_norm_
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.logger = CircuitTracerLogger()
        strategy = getattr(config.circuit_tracer, "trace_strategy", None) or getattr(
            config.circuit_tracer, "strategy", "all"
        )
        self.tracer_adapter = CircuitTracerAdapter(
            self.logger,
            trace_frequency=config.circuit_tracer.trace_frequency,
            strategy=strategy,
        )
        self.reward_calculator = GRPORewardCalculator(reward_weights, config.hallucination)
        self._GRPOGroupSample = GRPOGroupSample

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
                self._clip_grad_norm(self.model.parameters(), 1.0)
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

    def _collect_group(self, prompt: str):
        torch = self._torch
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

        group = self._GRPOGroupSample(prompt=prompt)
        for idx, response in enumerate(responses):
            sample = self._GRPOGroupSample.Sample(
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
        model,
        sequences: Sequence,
        prompt_len: int,
        *,
        requires_grad: bool,
    ) -> List:
        torch = self._torch
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

    def _compute_group_loss(self, group) -> object:
        torch = self._torch
        losses: list = []
        for sample in group.samples:
            advantage = torch.tensor(sample.advantage, device=self.device, dtype=torch.float32)
            policy_log_prob = sample.log_prob
            ref_log_prob = sample.ref_log_prob
            kl = policy_log_prob - ref_log_prob
            losses.append(-advantage.detach() * policy_log_prob + self.config.kl_coef * kl)
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()


class GRPOTrainer:
    """Facade selecting the appropriate trainer implementation."""

    def __init__(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], BaseTrainerConfig):
            impl = _ConfigGRPOTrainer(*args, **kwargs)
        else:
            impl = _LegacyGRPOTrainer(*args, **kwargs)
        object.__setattr__(self, "_impl", impl)

    def __getattr__(self, name: str):  # pragma: no cover - delegation helper
        return getattr(self._impl, name)

    def __setattr__(self, name: str, value):  # pragma: no cover - delegation helper
        if name == "_impl":
            object.__setattr__(self, name, value)
        else:
            setattr(self._impl, name, value)


__all__ = [
    "GRPOTrainer",
    "GRPOTrainingStats",
    "GRPOTrainingSummary",
    "GRPOBatchStats",
]
