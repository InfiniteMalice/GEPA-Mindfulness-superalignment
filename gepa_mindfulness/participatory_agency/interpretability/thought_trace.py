"""Thought-trace integration hooks for participatory agency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from ..values import ValueComponents


@dataclass(frozen=True)
class ThoughtTraceRecord:
    """Container for thought-trace attribution metadata."""

    token_index: int
    components: ValueComponents
    metadata: Mapping[str, float]


def build_trace_hook(
    recorder: Callable[[ThoughtTraceRecord], None],
    default_metadata: Mapping[str, float] | None = None,
) -> Callable[[int, ValueComponents], None]:
    """Create a hook that records participatory agency values."""

    base_metadata = dict(default_metadata) if default_metadata else {}

    def _hook(token_index: int, values: ValueComponents) -> None:
        record = ThoughtTraceRecord(
            token_index=token_index,
            components=values,
            metadata=base_metadata,
        )
        recorder(record)

    return _hook
