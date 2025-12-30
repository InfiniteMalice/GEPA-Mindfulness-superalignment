"""Interpretability helpers for participatory agency."""

from .probes import LinearValueProbe, ProbeResult, run_probe
from .thought_trace import ThoughtTraceRecord, build_trace_hook

__all__ = [
    "LinearValueProbe",
    "ProbeResult",
    "ThoughtTraceRecord",
    "build_trace_hook",
    "run_probe",
]
