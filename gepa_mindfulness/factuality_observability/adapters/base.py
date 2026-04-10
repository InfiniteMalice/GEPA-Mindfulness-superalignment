"""Adapter interfaces for attribution-graph and mech-interp integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..logging import TracePackage


@dataclass(slots=True)
class GraphBuilderInput:
    """Canonical input payload for external attribution-graph builders."""

    trace_package_id: str
    prompt_text: str
    answer_text: str
    atomic_fact_map: dict[int, str]
    evidence_map: dict[int, list[str]]
    critical_span_annotations: dict[str, list[list[int]]]
    case_overlay: str


@dataclass(slots=True)
class GraphBuilderOutput:
    """Returned graph summary to reintegrate into evaluation records."""

    trace_package_id: str
    graph_id: str
    summary: str
    suspected_spurious_path: bool
    status: str


class AttributionGraphAdapter(Protocol):
    """Interface for converting trace packages into graph-builder requests."""

    def to_graph_input(self, trace_package: TracePackage) -> GraphBuilderInput:
        """Convert trace package into graph-builder specific input."""

    def ingest_graph_output(self, output: GraphBuilderOutput) -> dict[str, object]:
        """Convert graph output to evaluator-friendly metadata."""


class CircuitTraceAdapter(Protocol):
    """Interface for optional circuit tracing backend integrations."""

    def queue_trace(self, trace_package: TracePackage) -> str:
        """Queue deep trace request and return backend job id."""

    def summarize_trace(self, job_id: str) -> dict[str, object]:
        """Fetch trace summaries from backend-specific jobs."""


class FeatureInspectionAdapter(Protocol):
    """Interface for SAE and feature-level analysis tools."""

    def inspect_features(self, trace_package: TracePackage) -> dict[str, object]:
        """Return feature-level summaries linked to atomic claims."""


class TraceViewerAdapter(Protocol):
    """Interface for rendering trace packages in viewer applications."""

    def render_payload(self, trace_package: TracePackage) -> dict[str, object]:
        """Export trace package to viewer input schema."""
