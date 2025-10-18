"""Shared infrastructure for PPO and GRPO trainers."""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Sequence

from ..core.circuit_tracer_adapter import CircuitTracerAdapter, TraceResult
from ..core.rewards import GEPARewardCalculator, RewardBreakdown
from .config import BaseTrainerConfig, CircuitTracerConfig
from .dataloader import DatasetBatch


@dataclass
class GeneratedResponse:
    """Container for generated text and associated statistics."""

    text: str
    log_probs: Sequence[float]
    mask: Sequence[int]
    policy_log_prob: object | None = field(default=None)
    reference_log_prob: object | None = field(default=None)
    metadata: dict[str, float] | None = field(default=None)

    def confidence(self) -> float:
        """Return the mean token probability for the generated tokens."""

        selected = [math.exp(value) for value, flag in zip(self.log_probs, self.mask) if flag]
        if not selected:
            return 0.0
        return float(sum(selected) / len(selected))


class BaseTrainer(ABC):
    """Base class providing logging and reward computation helpers."""

    def __init__(
        self,
        config: BaseTrainerConfig,
        *,
        reward_calculator: GEPARewardCalculator | None = None,
        tracer_adapter: CircuitTracerAdapter | None = None,
    ) -> None:
        self.config = config
        self.dataset = DatasetBatch.from_path(Path(config.dataset_path))
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.reward_calculator = reward_calculator or config.create_reward_calculator()
        self.tracer_adapter = tracer_adapter or self._build_tracer_adapter(config.circuit_tracer)
        self.global_step = 0
        self.logged_metrics: List[Mapping[str, object]] = []

    @abstractmethod
    def train(self) -> None:
        """Execute the trainer's optimisation loop."""

    @abstractmethod
    def _compute_advantages(
        self, grouped_rewards: Sequence[Sequence[float]]
    ) -> Sequence[Sequence[float]]:
        """Return advantage values aligned with ``grouped_rewards``."""

    def compute_batch_rewards(
        self,
        prompts: Sequence[str],
        responses: Sequence[Sequence[GeneratedResponse]],
        *,
        references: Sequence[Sequence[str] | str | None],
        gepa_scores: Sequence[Mapping[str, float] | None],
        imperatives: Sequence[Mapping[str, Mapping[str, float]] | None],
    ) -> tuple[list[list[float]], list[list[RewardBreakdown]]]:
        """Compute rewards for a batch of generated responses."""

        trace_groups = self._trace(prompts, responses)
        batch_rewards: list[list[float]] = []
        breakdowns: list[list[RewardBreakdown]] = []
        for prompt, group, reference, gepa, imperative, trace_group in zip(
            prompts, responses, references, gepa_scores, imperatives, trace_groups
        ):
            group_rewards: list[float] = []
            group_breakdowns: list[RewardBreakdown] = []
            for generated, trace_result in zip(group, trace_group):
                summary, assessment = self._resolve_trace(trace_result, generated)
                breakdown = self.reward_calculator.compute_reward(
                    response=generated.text,
                    reference_answers=reference,
                    gepa_scores=gepa,
                    imperatives=imperative,
                    confidence=generated.confidence(),
                    trace_summary=summary,
                    abstention=assessment,
                )
                group_rewards.append(breakdown.total)
                group_breakdowns.append(breakdown)
                self._log_metric(
                    prompt=prompt,
                    response=generated.text,
                    reward=breakdown.total,
                    breakdown=breakdown,
                )
            batch_rewards.append(group_rewards)
            breakdowns.append(group_breakdowns)
        return batch_rewards, breakdowns

    def save_summary(self) -> None:
        """Persist the accumulated metrics summary to disk."""

        payload = {"steps": self.global_step, "metrics": self.logged_metrics}
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_checkpoint(self, step: int) -> None:
        """Persist a lightweight checkpoint marker."""

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        marker = checkpoint_dir / f"checkpoint_{step:05d}.json"
        marker.write_text(json.dumps({"step": step}), encoding="utf-8")

    def _log_metric(
        self,
        *,
        prompt: str,
        response: str,
        reward: float,
        breakdown: RewardBreakdown,
    ) -> None:
        record = {
            "step": self.global_step,
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "task_success": breakdown.task_success,
            "gepa_alignment": breakdown.gepa_alignment,
            "honesty": breakdown.honesty,
            "hallucination": breakdown.hallucination,
        }
        self.logged_metrics.append(record)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + os.linesep)

    def _trace(
        self,
        prompts: Sequence[str],
        responses: Sequence[Sequence[GeneratedResponse]],
    ) -> list[list[TraceResult | None]]:
        if not self.tracer_adapter:
            return [[None for _ in group] for group in responses]
        response_texts = [[item.text for item in group] for group in responses]
        return self.tracer_adapter.trace_responses(prompts, response_texts)

    @staticmethod
    def _resolve_trace(
        trace_result: TraceResult | None,
        generated: GeneratedResponse,
    ) -> tuple[Mapping[str, str], object | None]:
        if trace_result is None:
            return {}, None
        return trace_result.summary, trace_result.assessment

    def _build_tracer_adapter(self, config: CircuitTracerConfig) -> CircuitTracerAdapter | None:
        if not config.enabled:
            return None
        return CircuitTracerAdapter(
            tracer=None,
            strategy=config.strategy,
            trace_frequency=config.trace_frequency,
            seed=config.seed,
        )


__all__ = ["BaseTrainer", "GeneratedResponse"]
