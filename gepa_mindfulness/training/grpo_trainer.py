"""Lightweight GRPO trainer used by the test suite."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # pragma: no cover - torch is optional for lightweight tests
    import torch
except Exception:  # pragma: no cover - fallback when torch is absent
    torch = None  # type: ignore[assignment]

from ..core.circuit_tracer_adapter import TraceResult
from .base_trainer import BaseTrainer, GeneratedResponse
from .config import GRPOConfig as DatasetGRPOConfig
from .configs import GRPOConfig as TransformersGRPOConfig
from .dataloader import DatasetExample
from .grpo_reward_calculator import GRPORewardCalculator
from .grpo_types import GRPOGroupSample


@dataclass
class GRPOTrainingStats:
    """Summary of a single GRPO optimisation step."""

    prompt: str
    rewards: Sequence[float]
    advantages: Sequence[float]
    confidences: Sequence[float]


@dataclass
class GRPOBatchSummary:
    """Aggregate statistics for a single prompt during HF-style training."""

    prompt: str
    mean_reward: float
    advantages: List[float]
    categories: List[str]
    confidences: List[float]


@dataclass
class EpochSummary:
    """Lightweight summary returned by ``train_epoch`` in HF compatibility mode."""

    steps: int
    batches: List[GRPOBatchSummary]

    def mean_reward(self) -> float:
        if not self.batches:
            return 0.0
        return float(
            sum(batch.mean_reward for batch in self.batches) / max(len(self.batches), 1)
        )


RewardsMatrix = Sequence[Sequence[float]]


GRPOEpochSum = EpochSummary


class GRPOTrainer(BaseTrainer):
    """Minimal trainer implementing the GRPO interface exercised by tests."""

    def __init__(self, *args, seed: int | None = None, **kwargs) -> None:
        if args and isinstance(args[0], DatasetGRPOConfig):
            if len(args) != 1:
                raise TypeError(
                    "Dataset-driven GRPOTrainer expects a single GRPOConfig argument"
                )
            if kwargs:
                raise TypeError(
                    "Dataset-driven GRPOTrainer does not accept additional keyword"
                    " arguments"
                )
            dataset_config = args[0]
            super().__init__(dataset_config)
            self.random = random.Random(seed or 0)
            self._logits: dict[str, float] = {}
            self.training_history: list[GRPOTrainingStats] = []
            self._hf_mode = False
            return

        if "config" in kwargs and not isinstance(
            kwargs.get("config"), TransformersGRPOConfig
        ):
            if args:
                raise TypeError(
                    "Dataset-driven GRPOTrainer expects config as the sole argument"
                )
            dataset_config = kwargs.pop("config")
            if not isinstance(dataset_config, DatasetGRPOConfig):
                raise TypeError(
                    "config must be a GRPOConfig when initialising without models"
                )
            if kwargs:
                raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
            super().__init__(dataset_config)
            self.random = random.Random(seed or 0)
            self._logits: dict[str, float] = {}
            self.training_history: list[GRPOTrainingStats] = []
            self._hf_mode = False
            return

        if torch is None:  # pragma: no cover - exercised when torch missing
            raise RuntimeError(
                "PyTorch is required for GRPOTrainer when initialised with models"
            )

        positional = list(args)
        policy_model = positional.pop(0) if positional else kwargs.pop("policy_model", None)
        reference_model = (
            positional.pop(0)
            if positional
            else kwargs.pop("reference_model", None)
        )
        tokenizer = positional.pop(0) if positional else kwargs.pop("tokenizer", None)
        config = positional.pop(0) if positional else kwargs.pop("config", None)
        reward_weights = (
            positional.pop(0)
            if positional
            else kwargs.pop("reward_weights", None)
        )

        if positional:
            raise TypeError(f"Unexpected positional arguments: {positional}")

        missing = [
            name
            for name, value in {
                "policy": policy_model,
                "reference": reference_model,
                "tokenizer": tokenizer,
                "config": config,
                "reward_weights": reward_weights,
            }.items()
            if value is None
        ]
        if missing:
            raise TypeError(
                "Model-driven GRPOTrainer expects policy, reference, tokenizer, config, "
                "and reward weights"
            )

        device = kwargs.pop("device", None)
        output_dir = kwargs.pop("output_dir", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        if isinstance(device, str):
            device = torch.device(device)
        elif device is None:
            device = torch.device("cpu")

        if not isinstance(config, TransformersGRPOConfig):
            raise TypeError(
                "Expected Transformers GRPOConfig when initialising with models"
            )

        if hasattr(policy_model, "to"):
            self.policy_model = policy_model.to(device)
        else:
            self.policy_model = policy_model

        if hasattr(reference_model, "to"):
            self.reference_model = reference_model.to(device)
        else:
            self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.transformers_config = config
        self.reward_weights = reward_weights
        self.device = device
        self._hf_mode = True
        self._hf_steps = 0
        self._reward_calculator = GRPORewardCalculator(
            reward_weights, config.hallucination
        )
        if hasattr(self.policy_model, "eval"):
            self.policy_model.eval()
        if hasattr(self.reference_model, "eval"):
            self.reference_model.eval()

        if output_dir is None:
            output_dir = Path.cwd() / "runs" / "grpo_hf"
        else:
            output_dir = Path(output_dir)
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.dataset = None
        self.output_dir = output_dir
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.reward_calculator = self._reward_calculator
        self.tracer_adapter = None
        self.global_step = 0
        self.logged_metrics = []

    def train(self) -> None:
        """Run a lightweight training loop over the dataset."""

        if getattr(self, "_hf_mode", False):
            raise RuntimeError("train() is not supported in HF compatibility mode")

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

    def _compute_advantages(self, grouped_rewards: RewardsMatrix) -> list[list[float]]:
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

    # ------------------------------------------------------------------
    # HF compatibility helpers

    def train_epoch(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> GRPOEpochSum:
        """Run one HF-style epoch and return the aggregated rollout summary."""
        if not getattr(self, "_hf_mode", False):
            raise RuntimeError("train_epoch() is only available in HF compatibility mode")

        assert torch is not None  # mypy hint: guarded in __init__

        if batch_size is None:
            batch_size = self.transformers_config.batch_size

        batches: list[GRPOBatchSummary] = []
        steps = 0
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            steps += 1
            for prompt in chunk:
                encoded = self.tokenizer(prompt, return_tensors="pt")
                if hasattr(encoded, "to"):
                    encoded = encoded.to(self.device)
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")

                with torch.no_grad():
                    sequences = self.policy_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.transformers_config.max_new_tokens,
                        do_sample=True,
                        temperature=self.transformers_config.sampling_temperature,
                        num_return_sequences=self.transformers_config.group_size,
                        pad_token_id=pad_token_id,
                    )

                if sequences.dim() == 1:
                    sequences = sequences.unsqueeze(0)

                group = GRPOGroupSample(prompt=prompt)
                for seq in sequences:
                    tokens = seq.tolist()
                    log_prob = torch.zeros(len(tokens), device=self.device, dtype=torch.float32)
                    ref_log_prob = torch.zeros_like(log_prob)
                    text = (
                        self.tokenizer.decode(tokens, skip_special_tokens=True)
                        if hasattr(self.tokenizer, "decode")
                        else str(tokens)
                    )
                    trace = TraceResult(
                        summary={},
                        assessment=None,
                        confidence_hint=0.6,
                        traced=False,
                    )
                    group.samples.append(
                        GRPOGroupSample.Sample(
                            response=text,
                            tokens=tokens,
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            trace=trace,
                        )
                    )

                computations = self._reward_calculator.score_group(group)
                rewards = [comp.reward for comp in computations]
                advantages = [sample.advantage for sample in group.samples]
                categories = [comp.category for comp in computations]
                confidences = [comp.confidence for comp in computations]
                mean_reward = sum(rewards) / max(len(rewards), 1)
                batches.append(
                    GRPOBatchSummary(
                        prompt=prompt,
                        mean_reward=mean_reward,
                        advantages=[float(value) for value in advantages],
                        categories=categories,
                        confidences=[float(value) for value in confidences],
                    )
                )

        self._hf_steps += steps
        return GRPOEpochSum(steps=steps, batches=batches)


__all__ = [
    "GRPOTrainer",
    "GRPOTrainingStats",
    "GRPOBatchSummary",
    "EpochSummary",
    "GRPOEpochSum",
]
