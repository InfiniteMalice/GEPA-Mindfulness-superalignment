from __future__ import annotations

from pathlib import Path

from gepa_mindfulness.core.tracing import CircuitTracerLogger
from mindful_trace_gepa.logging_schema import (
    StructuredEventType,
    make_event_envelope,
    normalize_trace_event,
)
from mindful_trace_gepa.tokens import TokenRecorder
from mindful_trace_gepa.viewer.builder import build_viewer_html


def test_legacy_and_event_envelope_traces_load_for_viewer(tmp_path: Path) -> None:
    legacy = {"stage": "framing", "content": "legacy"}
    envelope = make_event_envelope(
        StructuredEventType.MEMORY_LAUNDERING_REPORT,
        {"summary": "memory report", "memory_id": "m1"},
        rollout_id="r1",
    ).to_dict()
    out = build_viewer_html([legacy, envelope], [], tmp_path / "viewer.html")
    html = out.read_text(encoding="utf-8")
    assert "legacy" in html
    assert "memory_laundering_report" in html


def test_normalize_trace_event_preserves_optional_payload_fields() -> None:
    row = make_event_envelope(
        StructuredEventType.SSR_RUN_REPORT,
        {"ssr_run_id": "s1", "review_required": True},
    ).to_dict()
    normalized = normalize_trace_event(row)
    assert normalized["ssr_run_id"] == "s1"
    assert normalized["event_type"] == "ssr_run_report"


def test_synthetic_token_telemetry_is_labeled_and_abstention_propagates() -> None:
    recorder = TokenRecorder()
    recorder.record_text("I do not know", abstained=True)
    record = recorder.records[0]
    assert record.telemetry_mode == "synthetic"
    assert record.telemetry_available is False
    assert record.abstained is True


def test_unavailable_circuit_telemetry_is_not_measured_zero() -> None:
    logger = CircuitTracerLogger()
    telemetry = logger.extract_span_circuits(None, 0, 1)
    assert telemetry["uncertainty_circuits"] is None
    assert telemetry["telemetry_available"] is False


def test_event_references_join_to_rollouts_by_id() -> None:
    for event_type, key in [
        (StructuredEventType.SEMANTIC_ASSESSMENT, "semantic_assessment_id"),
        (StructuredEventType.MEMORY_LAUNDERING_REPORT, "memory_laundering_report_id"),
        (StructuredEventType.CPT_PAIRWISE_EXAMPLE, "pair_id"),
        (StructuredEventType.SSR_RUN_REPORT, "ssr_run_id"),
        (StructuredEventType.ATTRIBUTION_REFERENCE, "attribution_graph_reference"),
    ]:
        event = make_event_envelope(event_type, {key: "ref-1"}, rollout_id="rollout-1").to_dict()
        assert normalize_trace_event(event)[key] == "ref-1"
        assert event["rollout_id"] == "rollout-1"
