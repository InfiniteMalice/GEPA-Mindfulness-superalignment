"""Integration layer for Anthropic-style self-tracing."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from self_tracing import Tracer  # type: ignore
except Exception:  # pragma: no cover - fallback when package missing
    Tracer = None  # type: ignore


TRACE_STAGES = ["framing", "evidence", "tensions", "decision", "reflection"]


@dataclass
class TraceEvent:
    stage: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtTrace:
    """Container storing self-tracing events for a single rollout."""

    events: List[TraceEvent] = field(default_factory=list)

    def add_event(self, stage: str, content: str, **metadata: Any) -> None:
        if stage not in TRACE_STAGES:
            raise ValueError(f"Unknown trace stage: {stage}")
        self.events.append(TraceEvent(stage=stage, content=content, metadata=metadata))

    def to_payload(self) -> List[Dict[str, Any]]:
        return [
            {
                "stage": event.stage,
                "content": event.content,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata,
            }
            for event in self.events
        ]

    def summary(self) -> Dict[str, str]:
        return {event.stage: event.content for event in self.events}


class SelfTracingLogger:
    """Wrapper that gracefully falls back if the optional dependency is missing."""

    def __init__(self) -> None:
        self._tracer = Tracer() if Tracer is not None else None
        self._active_traces: List[ThoughtTrace] = []

    @contextlib.contextmanager
    def trace(self, **context: Any) -> Iterable[ThoughtTrace]:
        trace = ThoughtTrace()
        self._active_traces.append(trace)
        if self._tracer is not None:
            with self._tracer.span(**context):  # type: ignore[attr-defined]
                yield trace
        else:
            yield trace
        self._active_traces.pop()

    def log_event(self, stage: str, content: str, **metadata: Any) -> None:
        if not self._active_traces:
            raise RuntimeError("log_event called outside of trace context")
        trace = self._active_traces[-1]
        trace.add_event(stage, content, **metadata)
        if self._tracer is not None:
            self._tracer.log(stage=stage, content=content, **metadata)  # type: ignore[attr-defined]

    @property
    def latest_trace(self) -> Optional[ThoughtTrace]:
        return self._active_traces[-1] if self._active_traces else None
