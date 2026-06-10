"""Backward-compatible structured event envelopes for trace logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class StructuredEventType(str, Enum):
    REASONING_CHECKPOINT = "reasoning_checkpoint"
    REWARD_BREAKDOWN = "reward_breakdown"
    TOKEN_TELEMETRY = "token_telemetry"
    SEMANTIC_ASSESSMENT = "semantic_assessment"
    PRINCIPLE_ROBUSTNESS_ASSESSMENT = "principle_robustness_assessment"
    MEMORY_WRITE_ASSESSMENT = "memory_write_assessment"
    MEMORY_RETRIEVAL_ASSESSMENT = "memory_retrieval_assessment"
    MEMORY_LAUNDERING_REPORT = "memory_laundering_report"
    CPT_PAIRWISE_EXAMPLE = "cpt_pairwise_example"
    CPT_TEACHER_LABEL = "cpt_teacher_label"
    CPT_TRAINING_METRIC = "cpt_training_metric"
    SSR_REASONING_UNIT = "ssr_reasoning_unit"
    SSR_CONTROLLED_RESOLVE = "ssr_controlled_resolve"
    SSR_REPAIR_EVENT = "ssr_repair_event"
    SSR_RUN_REPORT = "ssr_run_report"
    DECEPTION_PROBE = "deception_probe"
    ATTRIBUTION_REFERENCE = "attribution_reference"
    REVIEW_EVENT = "review_event"
    REPAIR_EVENT = "repair_event"
    OBJECTIVE_SPECIFICATION = "objective_specification"
    VALIDATOR_CAPTURE_ASSESSMENT = "validator_capture_assessment"
    PROXY_OBJECTIVE_ASSESSMENT = "proxy_objective_assessment"
    NOVELTY_ASSESSMENT = "novelty_assessment"
    OBJECTIVE_POSTERIOR_UPDATE = "objective_posterior_update"
    ROBUST_OBJECTIVE_DECISION = "robust_objective_decision"
    PROXY_BREAKDOWN_REPORT = "proxy_breakdown_report"
    OBJECTIVE_VALIDATION_INTERRUPT = "objective_validation_interrupt"
    KV_CACHE_FEATURE_SNAPSHOT = "kv_cache_feature_snapshot"
    KV_CONTEXT_RISK_ASSESSMENT = "kv_context_risk_assessment"
    TRAJECTORY_UPLIFT_ALERT = "trajectory_uplift_alert"
    CANDIDATE_RESPONSE_CLOSURE_ASSESSMENT = "candidate_response_closure_assessment"
    STRUCTURED_KNOWLEDGE_GRAPH = "structured_knowledge_graph"
    CORRELATED_CLAIM_ASSESSMENT = "correlated_claim_assessment"
    INFERENCE_PATH_ASSESSMENT = "inference_path_assessment"
    UNLEARNING_RECONSTRUCTABILITY_REPORT = "unlearning_reconstructability_report"
    DISCLOSURE_EVENT = "disclosure_event"
    CAPABILITY_GRAPH_UPDATE = "capability_graph_update"
    TRAJECTORY_SUMMARY = "trajectory_summary"
    RELEASE_GATE_ASSESSMENT = "release_gate_assessment"
    SAFE_REDACTION_EVENT = "safe_redaction_event"
    INTERNAL_STATE_TRAJECTORY_SNAPSHOT = "internal_state_trajectory_snapshot"
    ADAPTIVE_ATTACK_RUN = "adaptive_attack_run"
    MONITOR_BYPASS_TEST_RESULT = "monitor_bypass_test_result"


@dataclass(frozen=True)
class EventEnvelope:
    schema_version: str
    event_id: str
    event_type: str
    timestamp: str
    run_id: str | None = None
    rollout_id: str | None = None
    trace_id: str | None = None
    sample_id: str | None = None
    conversation_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint_step: int | None = None
    model_id: str | None = None
    model_checkpoint_hash: str | None = None
    dataset_id: str | None = None
    policy_version: str | None = None
    config_hash: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def make_event_envelope(
    event_type: StructuredEventType | str,
    payload: Mapping[str, Any],
    **ids: Any,
) -> EventEnvelope:
    event_value = (
        event_type.value if isinstance(event_type, StructuredEventType) else str(event_type)
    )
    return EventEnvelope(
        schema_version=str(ids.pop("schema_version", "1.0")),
        event_id=str(ids.pop("event_id", uuid4())),
        event_type=event_value,
        timestamp=str(ids.pop("timestamp", datetime.now(UTC).isoformat().replace("+00:00", "Z"))),
        payload=dict(payload),
        **ids,
    )


def normalize_trace_event(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy and envelope-shaped trace rows for viewers/tools."""

    normalized = dict(row)
    if "schema_version" in normalized and "event_type" in normalized and "payload" in normalized:
        payload = normalized.get("payload") or {}
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                normalized.setdefault(key, value)
        normalized.setdefault("stage", normalized.get("event_type"))
        normalized.setdefault("content", _content_from_payload(payload))
        return normalized
    normalized.setdefault("event_type", normalized.get("stage", "legacy_trace_event"))
    normalized.setdefault("payload", {})
    return normalized


def trainer_metric_optional_fields(**refs: Any) -> dict[str, Any]:
    """Return only populated optional trainer metric fields."""

    allowed = {
        "confidence",
        "abstained",
        "schema_case",
        "trace_summary",
        "contradiction_report",
        "abstention_assessment",
        "reward_components",
        "generated_response_metadata",
        "semantic_assessment_reference",
        "memory_laundering_report_reference",
        "cpt_pairwise_reference",
        "ssr_run_report_reference",
        "deception_fingerprint_reference",
        "circuit_trace_reference",
        "attribution_graph_reference",
        "objective_specification_reference",
        "validator_capture_assessment_reference",
        "proxy_objective_assessment_reference",
        "novelty_assessment_reference",
        "objective_posterior_reference",
        "robust_objective_decision_reference",
        "proxy_breakdown_report_reference",
        "objective_validation_interrupt_reference",
        "memory_boundary_reference",
        "value_decomposition_reference",
        "kv_cache_snapshot_reference",
        "kv_context_risk_reference",
        "trajectory_uplift_alert_reference",
        "candidate_response_closure_reference",
        "structured_knowledge_graph_reference",
        "correlated_claim_assessment_reference",
        "inference_path_assessment_reference",
        "unlearning_reconstructability_report_reference",
        "disclosure_event_reference",
        "capability_graph_update_reference",
        "trajectory_summary_reference",
        "release_gate_assessment_reference",
        "safe_redaction_event_reference",
        "internal_state_trajectory_snapshot_reference",
        "adaptive_attack_run_reference",
        "monitor_bypass_test_result_reference",
    }
    return {key: value for key, value in refs.items() if key in allowed and value is not None}


def _content_from_payload(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in (
        "content",
        "summary",
        "content_summary",
        "teacher_rationale_summary",
        "repair_reason",
    ):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


__all__ = [
    "EventEnvelope",
    "StructuredEventType",
    "make_event_envelope",
    "normalize_trace_event",
    "trainer_metric_optional_fields",
]
