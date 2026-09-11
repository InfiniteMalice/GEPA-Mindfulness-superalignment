"""Backward-compatible structured event envelopes for trace logging."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

_RFC3339_OFFSET_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


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
    PREDICTION_COMMIT = "prediction_commit"
    ACTION_PROPOSED = "action_proposed"
    ACTION_EXECUTED = "action_executed"
    OUTCOME_OBSERVED = "outcome_observed"
    VERIFICATION_RESULT = "verification_result"
    EPISTEMIC_ASSESSMENT = "epistemic_assessment"
    CASE_ASSESSMENT = "case_assessment"


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
    action_id: str | None = None
    parent_event_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    model_version: str | None = None
    harness_version: str | None = None
    case_version: str | None = None
    case_id: int | None = None
    stripe_id: str | None = None
    repeat_id: int | None = None
    seed: int | None = None
    authorization_scope: str | None = None
    verifier_refs: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        """Validate action-bound metadata and snapshot reference collections."""

        for field_name in (
            "action_id",
            "model_version",
            "harness_version",
            "case_version",
            "stripe_id",
            "authorization_scope",
            "superseded_by",
        ):
            _validate_optional_nonblank_string(field_name, getattr(self, field_name))

        for field_name in ("parent_event_ids", "evidence_refs", "verifier_refs"):
            object.__setattr__(
                self,
                field_name,
                _coerce_reference_tuple(field_name, getattr(self, field_name)),
            )

        _validate_optional_case_id(self.case_id)
        _validate_optional_nonnegative_int("repeat_id", self.repeat_id)
        _validate_optional_int("seed", self.seed)
        valid_from = _parse_validity_bound("valid_from", self.valid_from)
        valid_until = _parse_validity_bound("valid_until", self.valid_until)
        if valid_from is not None and valid_until is not None and valid_until < valid_from:
            raise ValueError("valid_until must not be earlier than valid_from")

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
        timestamp=str(
            ids.pop("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        ),
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
        "prediction_commit_reference",
        "action_record_reference",
        "outcome_observation_reference",
        "verification_result_reference",
        "epistemic_assessment_reference",
        "case_assessment_reference",
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


def _validate_optional_nonblank_string(field_name: str, value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{field_name} must be a nonblank string when supplied")


def _coerce_reference_tuple(field_name: str, values: object) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ValueError(f"{field_name} must be an iterable of nonblank strings")
    references = tuple(values)
    for reference in references:
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"{field_name} must contain only nonblank strings")
    return references


def _validate_optional_case_id(case_id: object) -> None:
    if case_id is not None and (type(case_id) is not int or not 0 <= case_id <= 17):
        raise ValueError("case_id must be a built-in integer from 0 through 17 when supplied")


def _validate_optional_nonnegative_int(field_name: str, value: object) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{field_name} must be a nonnegative built-in integer when supplied")


def _validate_optional_int(field_name: str, value: object) -> None:
    if value is not None and type(value) is not int:
        raise ValueError(f"{field_name} must be a built-in integer when supplied")


def _parse_validity_bound(field_name: str, value: object) -> datetime | None:
    if value is None:
        return None
    _validate_optional_nonblank_string(field_name, value)
    timestamp = str(value)
    if _RFC3339_OFFSET_DATETIME.fullmatch(timestamp) is None:
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime")
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an explicit UTC offset")
    return parsed


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
