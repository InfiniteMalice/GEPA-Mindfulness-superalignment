"""Adapters bridging policy rollouts with self-tracing artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from ..core.tracing import ThoughtTrace


@dataclass
class TraceToCheckpoint:
    """Convert traces into GEPA checkpoint dictionaries."""

    def __call__(self, trace: ThoughtTrace) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        for event in trace.events:
            payload[f"trace_{event.stage}"] = event.content
        payload["trace_summary"] = " | ".join(
            f"{event.stage}:{event.content}" for event in trace.events
        )
        return payload


def generate_checkpoints(traces: Iterable[ThoughtTrace]) -> Iterable[Dict[str, str]]:
    adapter = TraceToCheckpoint()
    for trace in traces:
        yield adapter(trace)
