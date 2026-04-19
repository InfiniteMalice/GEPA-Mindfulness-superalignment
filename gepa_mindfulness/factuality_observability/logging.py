"""Logging and trace package export for factuality-observability diagnostics."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .schemas import (
    CaseOverlayV2,
    GuessingAbstentionDiagnostics,
    MechInterpStatus,
    ObservabilityTier,
    RelatedQueryConsistency,
    SuspectedFailureMode,
    TraceCaptureStatus,
)


@dataclass(slots=True)
class SampleLogBundle:
    """Minimum required per-example logging bundle."""

    sample_id: str
    prompt_id: str
    model_id: str
    model_version: str
    domain: str
    task_type: str
    raw_input: str
    raw_answer: str
    repaired_answer: str
    final_case_label: str
    final_case_overlay: str
    observability_tier: ObservabilityTier
    declared_confidence: float
    latent_uncertainty_signal_summary: float | None
    verification_status: str
    provenance_status: str
    recommended_action: str
    routing_path: list[str]
    atomic_fact_list: list[str]
    fact_verdict_per_fact: list[str]
    evidence_per_fact: list[list[str]]
    unsupported_fact_indices: list[int]
    contradiction_fact_indices: list[int]
    abstention_target: str
    routing_target: str
    answer_correctness_score: float
    calibration_score: float
    atomic_fact_support_score: float
    hallucination_axis_intrinsic_extrinsic: str
    hallucination_axis_factuality_faithfulness: str
    hallucination_primary_type: str
    hallucination_secondary_types: list[str]
    task_specific_hallucination_type: str
    guessing_pressure_profile: str
    knowledge_boundary_risk: float
    source_reference_divergence_risk: float
    staleness_risk: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tokenizer_version: str | None = None
    system_prompt_hash: str | None = None
    retrieved_context_ids: list[str] = field(default_factory=list)
    retrieved_context_hashes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TracePackage:
    """Reusable export package for attribution-graph and trace consumers."""

    trace_package_id: str
    sample_metadata: dict[str, str]
    prompt_text: str
    answer_text: str
    tokenization_map: dict[str, list[int]]
    atomic_fact_map: dict[int, str]
    evidence_map: dict[int, list[str]]
    critical_span_annotations: dict[str, list[list[int]]]
    per_token_uncertainty_series: list[float]
    retrieval_and_provenance_links: list[str]
    model_internal_summaries: dict[str, str]
    final_case_overlay: str
    suspected_failure_mode: SuspectedFailureMode
    suspected_spurious_path: bool
    supported_trace_artifacts: list[str]
    downstream_graph_status: str
    graph_candidate_priority: float


def deterministic_trace_package_id(sample_id: str, model_id: str, answer: str) -> str:
    """Compute stable trace package identifiers for joins and replays."""

    digest = hashlib.sha256(f"{sample_id}:{model_id}:{answer}".encode("utf-8")).hexdigest()
    return f"trace-{digest[:16]}"


def build_trace_package(
    sample_id: str,
    model_id: str,
    prompt: str,
    answer: str,
    atomic_fact_list: list[str],
    evidence_per_fact: list[list[str]],
    per_token_uncertainty_series: list[float] | None = None,
    final_case_overlay: str = "Case0-O0",
    suspected_failure_mode: SuspectedFailureMode = SuspectedFailureMode.UNKNOWN,
    suspected_spurious_path: bool = False,
    token_count: int | None = None,
) -> TracePackage:
    """Create a trace package that degrades gracefully when telemetry is absent.
    
    Args:
        sample_id: ...
        model_id: ...
        prompt: ...
        answer: ...
        atomic_fact_list: ...
        evidence_per_fact: ...
        per_token_uncertainty_series: ...
        final_case_overlay: ...
        suspected_failure_mode: ...
        suspected_spurious_path: ...
        token_count: Expected length of per_token_uncertainty_series. If not provided,
            will fall back to a naive whitespace tokenization count.
    """

    trace_id = deterministic_trace_package_id(sample_id, model_id, answer)
    if token_count is None or not isinstance(token_count, int) or token_count < 0:
        if token_count is not None:
            logging.warning(f"Invalid token_count {token_count}, falling back to len(answer.split())")
        resolved_token_count = len(answer.split())
    else:
        resolved_token_count = token_count
    if (
        per_token_uncertainty_series is not None
        and len(per_token_uncertainty_series) != resolved_token_count
    ):
        raise ValueError(
            "per_token_uncertainty_series length must match token_count in build_trace_package: "
            f"{len(per_token_uncertainty_series)} != {resolved_token_count}"
        )
    uncertainty = per_token_uncertainty_series or []
    artifacts = ["sample_metadata", "atomic_fact_map", "evidence_map"]
    if uncertainty:
        artifacts.append("token_uncertainty")

    return TracePackage(
        trace_package_id=trace_id,
        sample_metadata={"sample_id": sample_id, "model_id": model_id},
        prompt_text=prompt,
        answer_text=answer,
        tokenization_map={"tokens": list(range(resolved_token_count))},
        atomic_fact_map={idx: fact for idx, fact in enumerate(atomic_fact_list)},
        evidence_map={idx: evidence for idx, evidence in enumerate(evidence_per_fact)},
        critical_span_annotations={"answer_spans": [[0, max(0, len(answer) - 1)]]},
        per_token_uncertainty_series=uncertainty,
        retrieval_and_provenance_links=[],
        model_internal_summaries={},
        final_case_overlay=final_case_overlay,
        suspected_failure_mode=suspected_failure_mode,
        suspected_spurious_path=suspected_spurious_path,
        supported_trace_artifacts=artifacts,
        downstream_graph_status="queued",
        graph_candidate_priority=1.0 if suspected_spurious_path else 0.3,
    )


def mark_correct_but_wrong_reason(
    answer_correctness_score: float,
    provenance_status: str,
    support_coverage_score: float,
    uncertainty_disagreement: float,
) -> bool:
    """Mark high-value cases where answer is right but evidence pathway is suspicious."""

    return (
        answer_correctness_score >= 1.0
        and (provenance_status in {"none", "textual_only"} or support_coverage_score < 0.5)
        and uncertainty_disagreement > 0.4
    )


def to_jsonl_line(payload: SampleLogBundle | TracePackage | dict[str, object]) -> str:
    """Convert structured bundle to JSONL output line."""

    def _normalize(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(k): _normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    if isinstance(payload, dict):
        body = payload
    else:
        body = asdict(payload)
    return json.dumps(_normalize(body), sort_keys=True)


def build_sample_log_bundle(
    *,
    sample_id: str,
    prompt_id: str,
    model_id: str,
    model_version: str,
    domain: str,
    task_type: str,
    raw_input: str,
    raw_answer: str,
    repaired_answer: str,
    case_overlay: CaseOverlayV2,
    routing_path: list[str],
    routing_target: str,
    atomic_fact_list: list[str],
    fact_verdict_per_fact: list[str | Enum],
    evidence_per_fact: list[list[str]],
    unsupported_fact_indices: list[int],
    contradiction_fact_indices: list[int],
    answer_correctness_score: float,
    calibration_score: float,
    atomic_fact_support_score: float,
    guessing_pressure_profile: str,
    knowledge_boundary_risk: float,
    source_reference_divergence_risk: float,
    staleness_risk: float,
) -> SampleLogBundle:
    """Build minimum schema-complete log bundle for each evaluated sample."""

    return SampleLogBundle(
        sample_id=sample_id,
        prompt_id=prompt_id,
        model_id=model_id,
        model_version=model_version,
        domain=domain,
        task_type=task_type,
        raw_input=raw_input,
        raw_answer=raw_answer,
        repaired_answer=repaired_answer,
        final_case_label=f"Case{case_overlay.base_case_label}",
        final_case_overlay=case_overlay.final_case_overlay,
        observability_tier=case_overlay.observability_tier,
        declared_confidence=case_overlay.declared_confidence,
        latent_uncertainty_signal_summary=case_overlay.latent_uncertainty_signal,
        verification_status=case_overlay.verification_status.value,
        provenance_status=case_overlay.provenance_status.value,
        recommended_action=case_overlay.recommended_action.value,
        routing_path=routing_path,
        atomic_fact_list=atomic_fact_list,
        fact_verdict_per_fact=[
            verdict.value if isinstance(verdict, Enum) else str(verdict)
            for verdict in fact_verdict_per_fact
        ],
        evidence_per_fact=evidence_per_fact,
        unsupported_fact_indices=unsupported_fact_indices,
        contradiction_fact_indices=contradiction_fact_indices,
        abstention_target="abstain" if case_overlay.base_case_label >= 9 else "answer",
        routing_target=routing_target,
        answer_correctness_score=answer_correctness_score,
        calibration_score=calibration_score,
        atomic_fact_support_score=atomic_fact_support_score,
        hallucination_axis_intrinsic_extrinsic=(
            case_overlay.hallucination_axis_intrinsic_extrinsic.value
        ),
        hallucination_axis_factuality_faithfulness=(
            case_overlay.hallucination_axis_factuality_faithfulness.value
        ),
        hallucination_primary_type=case_overlay.hallucination_primary_type.value,
        hallucination_secondary_types=[
            item.value for item in case_overlay.hallucination_secondary_types
        ],
        task_specific_hallucination_type=case_overlay.task_specific_hallucination_type.value,
        guessing_pressure_profile=guessing_pressure_profile,
        knowledge_boundary_risk=knowledge_boundary_risk,
        source_reference_divergence_risk=source_reference_divergence_risk,
        staleness_risk=staleness_risk,
    )


def build_minimal_overlay(base_case_label: int) -> CaseOverlayV2:
    """Create an overlay baseline for quick migration from 13-case schema."""

    return CaseOverlayV2(base_case_label=base_case_label)


__all__ = [
    "GuessingAbstentionDiagnostics",
    "MechInterpStatus",
    "ObservabilityTier",
    "RelatedQueryConsistency",
    "SampleLogBundle",
    "SuspectedFailureMode",
    "TraceCaptureStatus",
    "TracePackage",
    "build_minimal_overlay",
    "build_sample_log_bundle",
    "build_trace_package",
    "deterministic_trace_package_id",
    "mark_correct_but_wrong_reason",
    "to_jsonl_line",
]
