"""Tests for structured safety logging schema extensions."""

# Local
from mindful_trace_gepa.logging_schema import (
    StructuredEventType,
    make_event_envelope,
    trainer_metric_optional_fields,
)


def test_logging_schema_accepts_kv_context_events() -> None:
    payload = {
        "conversation_id": "conv",
        "turn_index": 2,
        "single_prompt_risk": 0.18,
        "contextual_risk": 0.74,
        "contextual_uplift": 0.56,
        "contextual_ratio": 4.11,
        "trajectory_flag": True,
        "trajectory_reasons": ["component_assembly"],
        "candidate_response_risk": 0.2,
        "closure_risk": 0.8,
        "cache_mode": "transcript_fallback",
    }

    event = make_event_envelope(StructuredEventType.KV_CONTEXT_RISK_ASSESSMENT, payload)

    assert event.event_type == "kv_context_risk_assessment"
    assert event.payload["trajectory_flag"] is True


def test_logging_schema_accepts_release_and_structured_events() -> None:
    expected = {
        "STRUCTURED_KNOWLEDGE_GRAPH": "structured_knowledge_graph",
        "DISCLOSURE_EVENT": "disclosure_event",
        "RELEASE_GATE_ASSESSMENT": "release_gate_assessment",
        "ADAPTIVE_ATTACK_RUN": "adaptive_attack_run",
    }

    for name, value in expected.items():
        assert getattr(StructuredEventType, name).value == value


def test_trainer_metric_optional_fields_include_new_references() -> None:
    refs = trainer_metric_optional_fields(
        kv_cache_snapshot_reference="kv1",
        kv_context_risk_reference="risk1",
        trajectory_uplift_alert_reference="alert1",
        candidate_response_closure_reference="closure1",
        structured_knowledge_graph_reference="graph1",
        release_gate_assessment_reference="gate1",
        ignored_reference="nope",
    )

    assert refs == {
        "kv_cache_snapshot_reference": "kv1",
        "kv_context_risk_reference": "risk1",
        "trajectory_uplift_alert_reference": "alert1",
        "candidate_response_closure_reference": "closure1",
        "structured_knowledge_graph_reference": "graph1",
        "release_gate_assessment_reference": "gate1",
    }
