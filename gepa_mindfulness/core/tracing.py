"""Integration layer for the optional Circuit Tracer thought logging system."""

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ContextManager, Dict, Iterator, Protocol, cast


def _local_optional_import(module_name: str) -> Any:
    """Gracefully return ``None`` when optional tracing deps are unavailable."""

    import importlib

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


optional_import: Callable[[str], Any]

try:  # pragma: no cover - optional dependency missing in most environments
    from mindful_trace_gepa.utils.imports import optional_import as _optional_import
except ModuleNotFoundError:  # pragma: no cover - fallback when package absent
    optional_import = _local_optional_import
else:
    optional_import = cast(Callable[[str], Any], _optional_import)


class TracerProtocol(Protocol):
    """Minimal protocol implemented by ``circuit_tracer.CircuitTracer``."""

    def span(self, **context: Any) -> ContextManager[Any]: ...

    def log(self, *, stage: str, content: str, **metadata: Any) -> None: ...


LOGGER = logging.getLogger(__name__)


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
    factory = TracerFactory
    if factory is None:
        return None
    try:
        tracer = factory(**factory_kwargs)
    except TypeError:
        tracer = factory()
    return cast(TracerProtocol, tracer)


TRACE_STAGES = [
    "framing",
    "evidence",
    "tensions",
    "decision",
    "reflection",
    "path_1_reasoning",
    "path_2_reasoning",
    "comparison",
    "recommendation",
    "deception_analysis",
]


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
        self._active_traces: list[ThoughtTrace] = []
        self.circuit_tracer = None
        self.circuit_tracer_available = False

        # Attempt to instantiate a low-level circuit tracer for activation capture
        tracer_cls = None
        if _TRACER_MODULE is not None:
            tracer_cls = getattr(_TRACER_MODULE, "Tracer", None) or getattr(
                _TRACER_MODULE, "CircuitTracer", None
            )
        if tracer_cls is not None:
            try:
                self.circuit_tracer = tracer_cls()
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Unable to initialise circuit tracer: %s", exc)
            else:
                self.circuit_tracer_available = True
                LOGGER.info("Circuit tracer available")
        else:
            LOGGER.info("Circuit tracer not available - will use heuristic detection")

    @contextmanager
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

    @contextmanager
    def span(self, name: str, **context: Any) -> Iterator["_TraceSpanAdapter"]:
        """Context manager returning a helper that records events."""

        metadata = dict(context)
        metadata.setdefault("span", name)
        with self.trace(**metadata) as trace:
            adapter = _TraceSpanAdapter(self, trace)
            yield adapter

    @contextmanager
    def capture_circuits(self) -> Iterator[Any | None]:
        """Context manager to capture circuit activations during generation."""

        if self.circuit_tracer_available and self.circuit_tracer is not None:
            try:
                with self.circuit_tracer.trace() as trace:
                    yield trace
                    return
            except Exception as exc:  # pragma: no cover - defensive fallback
                LOGGER.warning("Circuit tracer capture failed: %s", exc)
        yield None

    def extract_span_circuits(self, trace: Any, start: int, end: int) -> Dict[str, float]:
        """Extract circuit activations for a specific text span."""

        if trace is None:
            return {
                "uncertainty_circuits": 0.0,
                "confidence_circuits": 0.0,
                "risk_circuits": 0.0,
                "reward_circuits": 0.0,
                "suppression_circuits": 0.0,
            }

        try:
            return {
                "uncertainty_circuits": trace.get_activation("uncertainty", start, end),
                "confidence_circuits": trace.get_activation("confidence", start, end),
                "risk_circuits": trace.get_activation("risk_assessment", start, end),
                "reward_circuits": trace.get_activation("reward_optimization", start, end),
                "suppression_circuits": trace.get_activation("knowledge_suppression", start, end),
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("Failed to extract circuit activations: %s", exc)
            return {
                "uncertainty_circuits": 0.0,
                "confidence_circuits": 0.0,
                "risk_circuits": 0.0,
                "reward_circuits": 0.0,
                "suppression_circuits": 0.0,
            }


class _TraceSpanAdapter:
    """Adapter exposing helper methods while a trace span is active."""

    def __init__(self, logger: CircuitTracerLogger, trace: ThoughtTrace) -> None:
        self._logger = logger
        self._trace = trace
        self._events: list[dict[str, Any]] = []
        self._contradictions: list[dict[str, Any]] = []

    def log_event(self, stage: str, payload: Dict[str, Any] | str) -> None:
        """Record an event on the underlying logger, preserving raw payload."""

        metadata: Dict[str, Any] = {}
        if not isinstance(payload, str):
            metadata["raw"] = payload
            content = json.dumps(payload, ensure_ascii=False)
        else:
            content = payload
        self._logger.log_event(stage, content, **metadata)
        self._events.append({"stage": stage, "payload": payload})

    def add_contradiction(self, detail: Dict[str, Any]) -> None:
        """Store a contradiction detail for later inspection."""

        self._contradictions.append(detail)

    def get_trace_summary(self) -> Dict[str, Any]:
        """Return a dictionary summarising the recorded events."""

        summary_map: Dict[str, Any] = dict(self._trace.summary())
        summary_map["events"] = list(self._events)
        return summary_map

    def get_contradictions(self) -> list[dict[str, Any]]:
        """Return contradictions recorded within the span."""

        return list(self._contradictions)
