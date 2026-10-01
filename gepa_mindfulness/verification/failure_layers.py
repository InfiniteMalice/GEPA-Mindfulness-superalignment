"""Opt-in, non-authoritative layer hypotheses over a recorded failure graph.

Hosts authenticate event producers and evidence contents. This adapter checks references,
PEO ancestry and numeric reconciliation; it cannot certify a reported diagnosis as true.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .epistemic_reconciliation import EpistemicReconciliation
from .epistemic_state import MismatchStatus
from .failure_graph import FailureGraph
from .interfaces import LocalVerificationResult, RelationalVerificationResult
from .state import parse_rfc3339_datetime


class FailureLayer(str, Enum):
    """Repair surfaces, independent of the failure graph's causal roles."""

    ROUTING = "routing"
    KNOWLEDGE_SKILL = "knowledge_skill"
    EXECUTION = "execution"
    WORLD_MODEL = "world_model"
    EVIDENCE = "evidence"
    CALIBRATION = "calibration"
    VERIFIER_MONITOR = "verifier_monitor"
    REPORTING = "reporting"
    AUTHORITY = "authority"


_REPAIR_TARGETS = {
    FailureLayer.ROUTING: "review_skill_selection_and_routing_description",
    FailureLayer.KNOWLEDGE_SKILL: "review_operational_guidance_and_replay_coverage",
    FailureLayer.EXECUTION: "inspect_executor_arguments_and_runtime",
    FailureLayer.WORLD_MODEL: "reconcile_predictions_and_check_environment_change",
    FailureLayer.EVIDENCE: "retrieve_and_reconcile_observable_sources",
    FailureLayer.CALIBRATION: "evaluate_confidence_against_held_out_outcomes",
    FailureLayer.VERIFIER_MONITOR: "audit_monitor_coverage_and_independent_verifiers",
    FailureLayer.REPORTING: "correct_claims_and_disclose_observed_limits",
    FailureLayer.AUTHORITY: "stop_and_review_authorization_scope",
}
_NEGATIVE_FINDINGS = {
    "executed": FailureLayer.EXECUTION,
    "arguments_valid": FailureLayer.EXECUTION,
    "schema_valid": FailureLayer.EXECUTION,
    "intended_operation_observed": FailureLayer.EXECUTION,
    "authorization_valid": FailureLayer.AUTHORITY,
    "irreversible_action_permitted": FailureLayer.AUTHORITY,
    "authorization_scope_valid": FailureLayer.AUTHORITY,
    "provenance_intact": FailureLayer.EVIDENCE,
    "claimed_outcome_supported": FailureLayer.REPORTING,
}


def _text(value: object, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _typed_verifier(
    event: EventEnvelope,
) -> LocalVerificationResult | RelationalVerificationResult | None:
    if event.event_type != StructuredEventType.VERIFICATION_RESULT:
        return None
    level = event.payload.get("verification_level")
    if level == "local_execution":
        return LocalVerificationResult.from_dict(event.payload["result"])
    if level == "relational_evidence":
        return RelationalVerificationResult.from_dict(event.payload["result"])
    return None


def _typed_evidence(event: EventEnvelope) -> tuple[EvidenceReference, ...] | None:
    result = _typed_verifier(event)
    if result is not None:
        return result.evidence_refs
    if event.event_type != StructuredEventType.EPISTEMIC_RECONCILIATION:
        return None
    record = EpistemicReconciliation.from_dict(event.payload)
    update = record.update
    return (
        update.evidence_refs
        + update.prior_state.evidence_refs
        + update.posterior_state.evidence_refs
        + tuple(ref for item in update.measurements for ref in item.evidence_refs)
        + tuple(ref for binding in record.bindings for ref in binding.innovation.evidence_refs)
    )


@dataclass(frozen=True, slots=True)
class LayerClaim:
    """Caller-reported hypothesis; a reference does not authenticate its conclusion."""

    failure_id: str
    layer: FailureLayer
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        _text(self.failure_id, "failure_id")
        if type(self.layer) is not FailureLayer:
            raise ValueError("layer must be FailureLayer")
        if type(self.evidence_refs) is not tuple or not self.evidence_refs:
            raise ValueError("evidence_refs must be a nonempty tuple")
        refs = []
        for ref in self.evidence_refs:
            if type(ref) is not EvidenceReference:
                raise ValueError("evidence_refs require EvidenceReference")
            snapshot = EvidenceReference.from_dict(ref.to_dict())
            if not snapshot.is_observable:
                raise ValueError("layer claims require observable evidence")
            refs.append(snapshot)
        if len({ref.reference_id for ref in refs}) != len(refs):
            raise ValueError("claim evidence_refs must be unique")
        object.__setattr__(self, "evidence_refs", tuple(refs))


@dataclass(frozen=True, slots=True)
class LayerAnnotation:
    """Diagnostic output only; status never upgrades a graph's causal support."""

    failure_id: str
    layer: FailureLayer
    basis: str
    evidence_refs: tuple[str, ...]
    repair_target: str
    status: str = "hypothesis"


@dataclass(frozen=True, slots=True)
class FailureLayerReport:
    """Historical snapshot ending at the selected reconciliation, not current state."""

    reconciliation_event_id: str
    input_digest: str
    graph_digest: str
    annotations: tuple[LayerAnnotation, ...]
    unlocalized_failure_ids: tuple[str, ...]
    residuals: tuple[tuple[str, float | None], ...]
    world_uncertainty: float | None
    model_uncertainty: float | None
    monitor_uncertainty: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize public diagnostics without creating a reward or repair receipt."""
        return json.loads(
            json.dumps(
                asdict(self)
                | {
                    "schema_version": "failure-layers-v1",
                    "confers_authority": False,
                }
            )
        )


def localize_failure_layers(
    events: tuple[EventEnvelope, ...],
    graph: FailureGraph,
    reconciliation_event_id: str,
    claims: tuple[LayerClaim, ...] = (),
    *,
    enabled: bool = False,
) -> FailureLayerReport:
    """Bind hypotheses to graph nodes in a validated reconciliation's causal ancestry.

    The complete supplied window must end at that reconciliation. Missing evidence, bad
    chronology and cross-unit nodes raise ValueError. High uncertainty and nonzero residuals
    alone leave layers unknown. Claims stay explicitly reported even when references match.
    """
    if enabled is not True:
        raise ValueError("failure localization must be explicitly enabled")
    if type(events) is not tuple or not events or len(events) > 10000:
        raise ValueError("events must be a nonempty tuple of at most 10000 events")
    if any(type(event) is not EventEnvelope for event in events):
        raise ValueError("events require exact EventEnvelope records")
    events = tuple(EventEnvelope(**event.to_dict()) for event in events)
    validate_action_bound_sequence(events)
    if type(graph) is not FailureGraph:
        raise ValueError("graph must be FailureGraph")
    graph = FailureGraph.from_dict(graph.to_dict())
    selected = events[-1]
    if (
        selected.event_id != reconciliation_event_id
        or selected.event_type != StructuredEventType.EPISTEMIC_RECONCILIATION
    ):
        raise ValueError("window must end at the selected epistemic reconciliation")
    unit_fields = (
        "run_id",
        "repeat_id",
        "conversation_id",
        "model_version",
        "harness_version",
        "case_version",
        "case_id",
        "stripe_id",
        "stripe_subtype",
        "seed",
    )
    if any(
        any(getattr(event, field) != getattr(selected, field) for field in unit_fields)
        for event in events
    ):
        raise ValueError("events must share one evaluation unit")
    times = [parse_rfc3339_datetime(event.timestamp, "event.timestamp") for event in events]
    if times != sorted(times):
        raise ValueError("event window must be chronological")
    by_id = {event.event_id: event for event in events}
    ancestry: set[str] = set()
    pending = [selected.event_id]
    while pending:
        identifier = pending.pop()
        if identifier in ancestry:
            continue
        if identifier not in by_id:
            raise ValueError("reconciliation ancestry must be complete")
        ancestry.add(identifier)
        pending.extend(by_id[identifier].parent_event_ids)
    record = EpistemicReconciliation.from_dict(selected.payload)
    nodes = {node.failure_id: node for node in graph.nodes}
    for node in graph.nodes:
        if node.event_id not in ancestry:
            raise ValueError("failure node must occur in reconciliation ancestry")
        event = by_id[node.event_id]
        if parse_rfc3339_datetime(node.observed_at, "node time") != parse_rfc3339_datetime(
            event.timestamp, "event time"
        ):
            raise ValueError("failure node time must match its event")
        if any(
            not ref.is_observable or ref.reference_id not in event.evidence_refs
            for ref in node.evidence_refs
        ):
            raise ValueError("failure node evidence must be observable and recorded on its event")
        typed_refs = _typed_evidence(event)
        if typed_refs is not None and any(
            ref not in typed_refs
            or any(
                other.reference_id == ref.reference_id and other.source_kind != ref.source_kind
                for other in typed_refs
            )
            for ref in node.evidence_refs
        ):
            raise ValueError("failure node evidence must preserve recorded source kinds")
    annotations: list[LayerAnnotation] = []

    def add(failure_id: str, layer: FailureLayer, basis: str, refs: tuple[str, ...]) -> None:
        annotation = LayerAnnotation(failure_id, layer, basis, refs, _REPAIR_TARGETS[layer])
        if annotation not in annotations:
            annotations.append(annotation)

    if type(claims) is not tuple or len(claims) > 10000:
        raise ValueError("claims must be a tuple of at most 10000 claims")
    for claim in claims:
        if type(claim) is not LayerClaim:
            raise ValueError("claims require LayerClaim records")
        claim = LayerClaim(claim.failure_id, claim.layer, claim.evidence_refs)
        if claim.failure_id not in nodes:
            raise ValueError("claim must name a graph failure")
        if any(ref not in nodes[claim.failure_id].evidence_refs for ref in claim.evidence_refs):
            raise ValueError("claim evidence must match its failure node")
        add(
            claim.failure_id,
            claim.layer,
            "reported_claim",
            tuple(ref.reference_id for ref in claim.evidence_refs),
        )
    mismatch = record.update.model_mismatch not in {MismatchStatus.NONE, MismatchStatus.UNASSESSED}
    mismatch |= any(
        binding.innovation.mismatch_status
        not in {
            MismatchStatus.NONE,
            MismatchStatus.UNASSESSED,
        }
        for binding in record.bindings
    )
    for node in graph.nodes:
        event = by_id[node.event_id]
        if mismatch and event.event_id == selected.event_id:
            add(
                node.failure_id,
                FailureLayer.WORLD_MODEL,
                "declared_model_mismatch",
                tuple(ref.reference_id for ref in node.evidence_refs),
            )
        result = _typed_verifier(event)
        if result is None:
            continue
        for binding in result.evidence_bindings:
            field = binding.field_name
            layer = _NEGATIVE_FINDINGS.get(field)
            value = getattr(result, field)
            if field == "contradiction_status" and value == "contradicted":
                layer = FailureLayer.EVIDENCE
            elif value is not False:
                continue
            refs = tuple(
                ref.reference_id
                for ref in binding.evidence_refs
                if ref.is_observable and ref in node.evidence_refs
            )
            if layer is not None and refs:
                add(node.failure_id, layer, f"verifier_finding:{field}", refs)
    state = record.update.posterior_state
    localized = {annotation.failure_id for annotation in annotations}
    return FailureLayerReport(
        selected.event_id,
        _digest([event.to_dict() for event in events]),
        _digest(graph.to_dict()),
        tuple(annotations),
        tuple(node.failure_id for node in graph.nodes if node.failure_id not in localized),
        tuple((binding.measurement_id, binding.innovation.residual) for binding in record.bindings),
        state.world_uncertainty,
        state.model_uncertainty,
        state.monitor_uncertainty,
    )
