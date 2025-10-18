"""Circuit Tracer integration that works for PPO and GRPO batches."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, MutableSequence, Sequence

from .abstention import AbstentionAssessment, assess_abstention_quality


class TracingStrategy(str, Enum):
    """Strategies that determine which responses should be traced."""

    ALL = "all"
    SINGLE = "single"
    SAMPLE = "sample"
    EXTREMES = "extremes"
    MIXED = "mixed"


@dataclass
class TraceResult:
    """Encapsulates Circuit Tracer output for a single completion."""

    summary: Mapping[str, str]
    trace: object | None
    assessment: AbstentionAssessment | None
    traced: bool


class CircuitTracerAdapter:
    """Run Circuit Tracer according to the requested sampling strategy."""

    def __init__(
        self,
        tracer: object | None,
        *,
        strategy: str = "all",
        trace_frequency: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.tracer = tracer
        try:
            self.strategy = TracingStrategy(strategy.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown tracing strategy: {strategy}") from exc
        self.trace_frequency = max(0.0, min(1.0, trace_frequency))
        self.random = random.Random(seed)

    def trace_responses(
        self,
        prompts: Sequence[str],
        responses: Sequence[Sequence[str]],
        *,
        rewards: Sequence[Sequence[float]] | None = None,
    ) -> list[list[TraceResult | None]]:
        """Trace responses according to the configured strategy.

        Args:
            prompts: Prompts associated with each batch element.
            responses: Generated responses. For PPO batches this will be a
                nested list where each inner list contains a single string.
            rewards: Optional rewards for each response used by EXTREMES and
                MIXED strategies.

        Returns:
            Nested list with :class:`TraceResult` objects or ``None`` for
            untraced completions.
        """

        if not responses:
            return []

        selections = self._select_indices(responses, rewards)
        traced_groups: list[list[TraceResult | None]] = []
        for prompt, group_responses, group_selection in zip(prompts, responses, selections):
            traced_groups.append(self._trace_group(prompt, group_responses, group_selection))
        return traced_groups

    def _select_indices(
        self,
        responses: Sequence[Sequence[str]],
        rewards: Sequence[Sequence[float]] | None,
    ) -> list[list[bool]]:
        selections: list[list[bool]] = []
        for group_index, group in enumerate(responses):
            if self.strategy is TracingStrategy.SINGLE:
                flags = [idx == 0 for idx in range(len(group))]
                selections.append(flags)
                continue
            flags = [False] * len(group)
            if self.strategy is TracingStrategy.ALL or math.isclose(self.trace_frequency, 1.0):
                flags = [True] * len(group)
            else:
                if self.strategy in {TracingStrategy.EXTREMES, TracingStrategy.MIXED}:
                    reward_group = rewards[group_index] if rewards else None
                    extreme_indices = self._extreme_indices(group, reward_group)
                    for idx in extreme_indices:
                        flags[idx] = True
                if self.strategy in {TracingStrategy.SAMPLE, TracingStrategy.MIXED}:
                    self._apply_random_sampling(flags)
            if self.strategy is TracingStrategy.SAMPLE and not any(flags):
                if flags:
                    flags[self.random.randrange(len(flags))] = True
            selections.append(flags)
        return selections

    def _apply_random_sampling(self, flags: MutableSequence[bool]) -> None:
        for idx in range(len(flags)):
            if flags[idx]:
                continue
            if self.random.random() <= self.trace_frequency:
                flags[idx] = True

    def _extreme_indices(self, group: Sequence[str], rewards: Sequence[float] | None) -> list[int]:
        if not group:
            return []
        if rewards and len(rewards) == len(group):
            best = max(range(len(group)), key=lambda idx: rewards[idx])
            worst = min(range(len(group)), key=lambda idx: rewards[idx])
        else:
            lengths = [len(text) for text in group]
            best = max(range(len(group)), key=lengths.__getitem__)
            worst = min(range(len(group)), key=lengths.__getitem__)
        if best == worst:
            return [best]
        return [best, worst]

    def _trace_group(
        self,
        prompt: str,
        responses: Sequence[str],
        selected: Sequence[bool],
    ) -> list[TraceResult | None]:
        traced: list[TraceResult | None] = []
        for index, (response, should_trace) in enumerate(zip(responses, selected)):
            if should_trace and self.tracer is not None:
                traced.append(self._run_tracer(prompt, response, index))
            elif should_trace:
                traced.append(self._heuristic_only(response))
            else:
                traced.append(None)
        return traced

    def _run_tracer(self, prompt: str, response: str, index: int) -> TraceResult:
        if hasattr(self.tracer, "trace"):
            with self.tracer.trace(prompt=prompt, response_index=index) as trace:
                summary = trace.summary() if hasattr(trace, "summary") else {}
        else:
            trace = None
            summary = {}
        assessment = assess_abstention_quality(summary, [response])
        return TraceResult(summary=summary, trace=trace, assessment=assessment, traced=True)

    def _heuristic_only(self, response: str) -> TraceResult:
        assessment = assess_abstention_quality({}, [response])
        return TraceResult(summary={}, trace=None, assessment=assessment, traced=False)


__all__ = [
    "CircuitTracerAdapter",
    "TraceResult",
    "TracingStrategy",
]
