"""Helpers to run Circuit Tracer across grouped completions efficiently."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence

from .abstention import (
    AbstentionAssessment,
    AbstentionQuality,
    assess_abstention_quality,
)

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .tracing import CircuitTracerLogger, ThoughtTrace
else:  # pragma: no cover - fallback for optional typing
    CircuitTracerLogger = Any  # type: ignore[misc,assignment]
    ThoughtTrace = Any  # type: ignore[misc,assignment]


@dataclass
class TraceAnalysis:
    """Summary of Circuit Tracer output for a single completion."""

    summary: dict[str, str]
    trace: ThoughtTrace | None
    confidence_hint: float
    abstention: AbstentionAssessment | None
    traced: bool

    def genuine_abstention(self) -> bool:
        return bool(self.abstention and self.abstention.is_genuine)

    def lazy_abstention(self) -> bool:
        return bool(self.abstention and self.abstention.is_lazy)


class CircuitTracerAdapter:
    """Manage Circuit Tracer spans for batched GRPO completions."""

    def __init__(
        self,
        logger: CircuitTracerLogger,
        *,
        trace_frequency: float = 1.0,
        trace_strategy: str = "all",
        seed: int | None = None,
    ) -> None:
        self.logger = logger
        self.trace_frequency = max(0.0, min(1.0, trace_frequency))
        self.trace_strategy = trace_strategy.lower()
        self.random = random.Random(seed)

    def analyse_group(self, prompt: str, responses: Sequence[str]) -> List[TraceAnalysis]:
        analyses: list[TraceAnalysis] = []
        should_trace_flags = self._select_indices(responses)

        for idx, response in enumerate(responses):
            if should_trace_flags[idx]:
                analysis = self._trace_single(prompt, response, idx)
            else:
                analysis = self._heuristic_only(response)
            analyses.append(analysis)
        return analyses

    def _select_indices(self, responses: Sequence[str]) -> List[bool]:
        if not responses:
            return []
        strategy = self.trace_strategy
        frequency = self.trace_frequency
        flags = [False] * len(responses)

        if strategy == "all" or frequency >= 1.0:
            return [True] * len(responses)

        if strategy == "extremes":
            longest = max(range(len(responses)), key=lambda i: len(responses[i]))
            shortest = min(range(len(responses)), key=lambda i: len(responses[i]))
            flags[longest] = True
            flags[shortest] = True
        elif strategy == "mixed":
            if responses:
                flags[0] = True
                flags[-1] = True

        for idx in range(len(responses)):
            if flags[idx]:
                continue
            if self.random.random() < frequency:
                flags[idx] = True

        if strategy == "sample" and not any(flags):
            chosen = self.random.randrange(len(responses))
            flags[chosen] = True

        return flags

    def _trace_single(self, prompt: str, response: str, idx: int) -> TraceAnalysis:
        with self.logger.trace(prompt=prompt, response_index=idx) as trace:
            sections = self._split_sections(response)
            for stage, text in sections.items():
                if text:
                    self.logger.log_event(stage, text[:512])
        summary = trace.summary() if trace else {}
        abstention = assess_abstention_quality(summary, sections.values())
        confidence_hint = self._confidence_from_sections(sections.values())
        return TraceAnalysis(
            summary=summary,
            trace=trace,
            confidence_hint=confidence_hint,
            abstention=abstention,
            traced=True,
        )

    def _heuristic_only(self, response: str) -> TraceAnalysis:
        sections = self._split_sections(response)
        summary = {stage: text for stage, text in sections.items() if text}
        abstention = assess_abstention_quality(summary, sections.values())
        confidence_hint = self._confidence_from_sections(sections.values())
        return TraceAnalysis(
            summary=summary,
            trace=None,
            confidence_hint=confidence_hint,
            abstention=abstention,
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
        lower = response
        sections: dict[str, str] = {value: "" for value in markers.values()}
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


__all__ = [
    "CircuitTracerAdapter",
    "TraceAnalysis",
    "AbstentionAssessment",
    "AbstentionQuality",
]
