"""Default lightweight adapters for environments without deep tracing tools."""

from __future__ import annotations

from .base import GraphBuilderInput, GraphBuilderOutput
from ..logging import TracePackage


class NullAttributionGraphAdapter:
    """No-op adapter that preserves schema compatibility."""

    def to_graph_input(self, trace_package: TracePackage) -> GraphBuilderInput:
        return GraphBuilderInput(
            trace_package_id=trace_package.trace_package_id,
            prompt_text=trace_package.prompt_text,
            answer_text=trace_package.answer_text,
            atomic_fact_map=trace_package.atomic_fact_map,
            evidence_map=trace_package.evidence_map,
            critical_span_annotations=trace_package.critical_span_annotations,
            case_overlay=trace_package.final_case_overlay,
        )

    def ingest_graph_output(self, output: GraphBuilderOutput) -> dict[str, object]:
        return {
            "trace_package_id": output.trace_package_id,
            "graph_id": output.graph_id,
            "graph_summary": output.summary,
            "suspected_spurious_path": output.suspected_spurious_path,
            "status": output.status,
        }


class NullCircuitTraceAdapter:
    """No-op circuit tracing adapter with graceful degradation."""

    def queue_trace(self, trace_package: TracePackage) -> str:
        return f"noop-{trace_package.trace_package_id}"

    def summarize_trace(self, job_id: str) -> dict[str, object]:
        return {"job_id": job_id, "status": "not_supported", "summary": "No backend configured"}


class NullFeatureInspectionAdapter:
    """No-op feature inspection adapter."""

    def inspect_features(self, trace_package: TracePackage) -> dict[str, object]:
        return {
            "trace_package_id": trace_package.trace_package_id,
            "status": "not_supported",
            "features": [],
        }


class NullTraceViewerAdapter:
    """No-op viewer adapter returning a minimal serializable payload."""

    def render_payload(self, trace_package: TracePackage) -> dict[str, object]:
        return {
            "trace_package_id": trace_package.trace_package_id,
            "prompt": trace_package.prompt_text,
            "answer": trace_package.answer_text,
            "case_overlay": trace_package.final_case_overlay,
        }
