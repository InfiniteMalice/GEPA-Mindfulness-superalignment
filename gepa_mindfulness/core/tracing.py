"""Integration layer for the optional Circuit Tracer thought logging system."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ContextManager, Iterator, List, Protocol, cast

def _local_optional_import(module_name: str):
    """Gracefully return ``None`` when optional tracing deps are unavailable."""

    import importlib

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


try:  # pragma: no cover - optional dependency missing in most environments
    from mindful_trace_gepa.utils.imports import optional_import as _optional_import
except ModuleNotFoundError:  # pragma: no cover - fallback when package absent
    optional_import = _local_optional_import
else:
    optional_import = _optional_import


class TracerProtocol(Protocol):
    """Minimal protocol implemented by ``circuit_tracer.CircuitTracer``."""

    def span(self, **context: Any) -> ContextManager[Any]: ...

    def log(self, *, stage: str, content: str, **metadata: Any) -> None: ...


_TRACER_MODULE = optional_import("circuit_tracer")


def _resolve_tracer_factory(module: Any) -> Callable[..., TracerProtocol] | None:
    """Best-effort discovery of a Circuit Tracer constructor."""

    candidate_names = [
        "CircuitTracer",
        "Tracer",
        "CircuitThoughtTracer",
        "create_tracer",
    ]
    for name in candidate_names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


TracerFactory: Callable[..., TracerProtocol] | None
if _TRACER_MODULE is not None:
    TracerFactory = _resolve_tracer_factory(_TRACER_MODULE)
else:  # pragma: no cover - optional dependency missing
    TracerFactory = None


def _create_tracer(**factory_kwargs: Any) -> TracerProtocol | None:
    if TracerFactory is None:
        return None
    try:
        tracer = TracerFactory(**factory_kwargs)
    except TypeError:
        tracer = TracerFactory()  # type: ignore[misc]
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
    """Container storing Circuit Tracer events for a single rollout."""

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


class CircuitTracerLogger:
    """Wrapper that gracefully falls back when Circuit Tracer is absent."""

    def __init__(self, **factory_kwargs: Any) -> None:
        self._factory_kwargs = dict(factory_kwargs)
        self._tracer = _create_tracer(**self._factory_kwargs)
        self._active_traces: List[ThoughtTrace] = []

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

    def refresh_backend(self) -> None:
        """Recreate the underlying tracer instance with stored kwargs."""

        self._tracer = _create_tracer(**self._factory_kwargs)
