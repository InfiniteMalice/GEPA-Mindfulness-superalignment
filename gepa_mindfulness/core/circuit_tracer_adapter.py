"""Simplified adapter that mimics the public Circuit Tracer interface."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .abstention import (
    AbstentionAssessment,
    AbstentionQuality,
    assess_abstention_quality,
)
from .tracing import CircuitTracerLogger


@dataclass
class TraceResult:
    """Minimal representation of a Circuit Tracer analysis."""

    summary: dict[str, str]
    assessment: AbstentionAssessment | None
    confidence_hint: float
    traced: bool


# Backwards compatibility: older modules import TraceAnalysis directly.
TraceAnalysis = TraceResult


# Backwards compatibility: older modules import TraceAnalysis directly.
TraceAnalysis = TraceResult


# Backwards compatibility: older modules import TraceAnalysis directly.
TraceAnalysis = TraceResult


# Backwards compatibility: older modules import TraceAnalysis directly.
TraceAnalysis = TraceResult


# Backwards compatibility: older modules import TraceAnalysis directly.
TraceAnalysis = TraceResult


# Backwards compatibility: older modules import TraceAnalysis directly.
TraceAnalysis = TraceResult


class CircuitTracerAdapter:
    """Best-effort wrapper that works even when the tracer dependency is absent."""

    def __init__(
        self,
        tracer: CircuitTracerLogger | None,
        *,
        strategy: str = "all",
        trace_frequency: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.logger = tracer or CircuitTracerLogger()
        self.trace_strategy = strategy.lower()
        self.trace_frequency = max(0.0, min(1.0, trace_frequency))
        self.random = random.Random(seed)

    def trace_responses(
        self,
        prompts: Sequence[str],
        responses: Sequence[Sequence[str]],
        *,
        rewards: Sequence[Sequence[float]] | None = None,
    ) -> list[list[TraceResult | None]]:
        traced_batches: list[list[TraceResult | None]] = []
        for index, (prompt, response_group) in enumerate(zip(prompts, responses)):
            reward_group = rewards[index] if rewards and index < len(rewards) else None
            traced_batches.append(self.analyse_group(prompt, response_group, reward_group))
        return traced_batches

    def analyse_group(
        self,
        prompt: str,
        responses: Sequence[str],
        reward_signals: Sequence[float] | None = None,
    ) -> list[TraceResult | None]:
        indices = self._select_indices(responses, reward_signals)
        results: list[TraceResult | None] = []
        for idx, response in enumerate(responses):
            if not indices[idx]:
                results.append(None)
                continue
            if self.logger.circuit_tracer_available:
                results.append(self._trace_single(prompt, response, idx))
            else:
                results.append(self._heuristic_only(response))
        return results

    def _select_indices(
        self, responses: Sequence[str], reward_signals: Sequence[float] | None
    ) -> List[bool]:
        if not responses:
            return []
        strategy = self.trace_strategy
        frequency = self.trace_frequency
        flags = [False] * len(responses)

        if strategy == "all":
            return [True] * len(responses)

        if strategy == "single":
            flags[0] = True
        elif strategy == "extremes":
            if reward_signals and len(reward_signals) == len(responses):
                highest = max(range(len(responses)), key=lambda i: reward_signals[i])
                lowest = min(range(len(responses)), key=lambda i: reward_signals[i])
            else:
                highest = max(range(len(responses)), key=lambda i: len(responses[i]))
                lowest = min(range(len(responses)), key=lambda i: len(responses[i]))
            flags[highest] = True
            flags[lowest] = True
        elif strategy == "mixed":
            if responses:
                flags[0] = True
                flags[-1] = True

        if strategy != "single":
            for idx in range(len(responses)):
                if flags[idx]:
                    continue
                if frequency >= 1.0 or self.random.random() < frequency:
                    flags[idx] = True

        if strategy == "sample" and not any(flags):
            chosen = self.random.randrange(len(responses))
            flags[chosen] = True

        return flags

    def _trace_single(self, prompt: str, response: str, idx: int) -> TraceResult:
        with self.logger.trace(prompt=prompt, response_index=idx) as trace:
            sections = self._split_sections(response)
            for stage, text in sections.items():
                if text:
                    self.logger.log_event(stage, text[:512])
        summary = trace.summary() if trace else {}
        abstention = assess_abstention_quality(summary, sections.values())
        confidence_hint = self._confidence_from_sections(sections.values())
        return TraceResult(
            summary=summary,
            assessment=abstention,
            confidence_hint=confidence_hint,
            traced=True,
        )

    def _heuristic_only(self, response: str) -> TraceResult:
        sections = self._split_sections(response)
        summary = {stage: text for stage, text in sections.items() if text}
        abstention = assess_abstention_quality(summary, sections.values())
        confidence_hint = self._confidence_from_sections(sections.values())
        return TraceResult(
            summary=summary,
            assessment=abstention,
            confidence_hint=confidence_hint,
            traced=False,
        )

    @staticmethod
    def _split_sections(response: str) -> dict[str, str]:
        markers = {
            "[PATH 1 REASONING]": "path_1_reasoning",
            "[PATH 2 REASONING]": "path_2_reasoning",
            "[COMPARISON]": "comparison",
            "[RECOMMENDATION]": "recommendation",
        }
        sections: dict[str, str] = {value: "" for value in markers.values()}
        lower = response
        for marker, key in markers.items():
            start = lower.find(marker)
            if start == -1:
                continue
            content_start = start + len(marker)
            next_positions = [
                lower.find(next_marker, content_start)
                for next_marker in markers
                if lower.find(next_marker, content_start) != -1
            ]
            end = min(next_positions) if next_positions else len(lower)
            sections[key] = response[content_start:end].strip()
        return sections

    @staticmethod
    def _confidence_from_sections(sections: Iterable[str]) -> float:
        joined = " ".join(section.lower() for section in sections)
        if "abstain" in joined or "not confident" in joined:
            return 0.4
        if "certain" in joined or "definitely" in joined:
            return 0.9
        if "probably" in joined or "likely" in joined:
            return 0.7
        return 0.6


__all__ = ["CircuitTracerAdapter", "TraceResult", "AbstentionAssessment", "AbstentionQuality"]
