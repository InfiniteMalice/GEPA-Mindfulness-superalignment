"""Integration layer for Anthropic-style self-tracing."""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ContextManager, Iterator, Protocol, cast


def _optional_import(name: str) -> Any:
    """Attempt to import ``name`` but tolerate missing optional deps."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None
    except Exception:
        return None


class TracerProtocol(Protocol):
    """Minimal protocol implemented by ``self_tracing.Tracer``."""

    def span(self, **context: Any) -> ContextManager[Any]: ...

    def log(self, *, stage: str, content: str, **metadata: Any) -> None: ...


_TRACER_MODULE = _optional_import("self_tracing")
TracerFactory: Callable[[], TracerProtocol] | None
if _TRACER_MODULE is not None:
    candidate = getattr(_TRACER_MODULE, "Tracer", None)
    TracerFactory = candidate if callable(candidate) else None
else:  # pragma: no cover - optional dependency missing
    TracerFactory = None


def _create_tracer() -> TracerProtocol | None:
    if TracerFactory is None:
        return None
    tracer = TracerFactory()
    return cast(TracerProtocol, tracer)


TRACE_STAGES = ["framing", "evidence", "tensions", "decision", "reflection"]


@dataclass
class TraceEvent:
    stage: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtTrace:
    """Container storing self-tracing events for a single rollout."""

    events: list[TraceEvent] = field(default_factory=list)

    def add_event(self, stage: str, content: str, **metadata: Any) -> None:
        if stage not in TRACE_STAGES:
            raise ValueError(f"Unknown trace stage: {stage}")
        self.events.append(TraceEvent(stage=stage, content=content, metadata=metadata))

    def to_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": event.stage,
                "content": event.content,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata,
            }
            for event in self.events
        ]

    def summary(self) -> dict[str, str]:
        return {event.stage: event.content for event in self.events}


class SelfTracingLogger:
    """Wrapper that gracefully falls back if the optional dependency is missing."""

    def __init__(self) -> None:
        self._tracer = _create_tracer()
        self._active_traces: list[ThoughtTrace] = []

    @contextlib.contextmanager
    def trace(self, **context: Any) -> Iterator[ThoughtTrace]:
        trace = ThoughtTrace()
        self._active_traces.append(trace)
        try:
            if self._tracer is not None:
                with self._tracer.span(**context):
                    yield trace
            else:
                yield trace
        finally:
            self._active_traces.pop()

    def log_event(self, stage: str, content: str, **metadata: Any) -> None:
        if not self._active_traces:
            raise RuntimeError("log_event called outside of trace context")
        trace = self._active_traces[-1]
        trace.add_event(stage, content, **metadata)
        if self._tracer is not None:
            self._tracer.log(stage=stage, content=content, **metadata)

    @property
    def latest_trace(self) -> ThoughtTrace | None:
        return self._active_traces[-1] if self._active_traces else None
