"""Opt-in public evidence-use diagnostics over the existing PEO event sequence.

Stage values are host observations, not causal identification or motive attribution.
Prediction capture prevents a later unavailability report from rewriting earlier use.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from gepa_mindfulness.verification.epistemic_reconciliation import EpistemicReconciliation
from gepa_mindfulness.verification.evidence_use import (
    EvidenceUseAssessment,
    MemoryInfluence,
    MemoryKind,
)
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from mindful_trace_gepa.logging_schema import EventEnvelope

from ._continuity_validation import boolean, references
from .epistemic_continuity import (
    EpistemicContinuityAssessment,
    EvidenceWindow,
    _commitment_bound,
    _digest,
    assess_epistemic_continuity,
)
from .epistemic_records import CommitmentUpdate, EpistemicCommitment


def evidence_use_digest(assessment: EvidenceUseAssessment) -> str:
    """Hash revalidated evidence-use inputs, including original evidence and policy."""
    if type(assessment) is not EvidenceUseAssessment:
        raise ValueError("evidence_use must be EvidenceUseAssessment")
    return _digest(replace(assessment).to_dict())


@dataclass(frozen=True, slots=True, kw_only=True)
class _BoundObservation:
    commitment_digest: str
    evidence_use_digest: str
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (self.commitment_digest, self.evidence_use_digest):
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("observation digest must be a lowercase SHA-256 digest")
        references(self.provenance, "provenance")
        if not self.provenance:
            raise ValueError("observation provenance is required")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible observation."""
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True, slots=True, kw_only=True)
class ProspectiveEvidenceUse(_BoundObservation):
    """Host observations recorded in predicted_outcome before action execution."""

    recognized: bool | None = None
    retained: bool | None = None
    retrieved: bool | None = None
    observed_kind: MemoryKind | None = None
    observed_influence: MemoryInfluence | None = None
    prediction_reflected: bool | None = None

    def __post_init__(self) -> None:
        _BoundObservation.__post_init__(self)
        for name in ("recognized", "retained", "retrieved", "prediction_reflected"):
            if getattr(self, name) is not None:
                boolean(getattr(self, name), name)
        for name, cls in (("observed_kind", MemoryKind), ("observed_influence", MemoryInfluence)):
            value = getattr(self, name)
            if value is not None:
                if type(value) not in (str, cls):
                    raise ValueError(f"invalid {name}")
                object.__setattr__(self, name, cls(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrospectiveEvidenceUse(_BoundObservation):
    """Public action/retention observations captured after reconciliation."""

    action_reflected: bool | None = None
    preserved_after_outcome: bool | None = None
    reported_unavailable: bool | None = None

    def __post_init__(self) -> None:
        _BoundObservation.__post_init__(self)
        for name in ("action_reflected", "preserved_after_outcome", "reported_unavailable"):
            if getattr(self, name) is not None:
                boolean(getattr(self, name), name)


@dataclass(frozen=True, slots=True, kw_only=True)
class PEOContinuityRequest:
    """One commitment's completed PEO cycle and later continuity decision."""

    events: tuple[EventEnvelope, ...]
    decision_event_id: str
    reconciliation_event_id: str
    assessment_event_id: str | None
    commitments: tuple[EpistemicCommitment, ...]
    commitment_id: str
    evidence_use: EvidenceUseAssessment
    active_commitment_ids: tuple[str, ...]
    provenance: tuple[str, ...]
    updates: tuple[CommitmentUpdate, ...] = ()
    decision_context_changed: bool = False


@dataclass(frozen=True, slots=True)
class PEOContinuityAudit:
    """Historical use failures and the independently assessed later continuity state."""

    commitment_id: str
    classification: str
    before: ProspectiveEvidenceUse | None
    after: RetrospectiveEvidenceUse | None
    continuity: EpistemicContinuityAssessment
    residuals: tuple[tuple[str, float], ...]
    chronology: tuple[tuple[str, str], ...]
    evidence_use_json: str

    def to_dict(self) -> dict[str, Any]:
        """Return detached public diagnostics; no motive, reward or authority output."""
        result = asdict(self)
        result["evidence_use"] = json.loads(result.pop("evidence_use_json"))
        result["causal_or_motive_claim"] = False
        return json.loads(json.dumps(result))


def _observation(
    container: object,
    key: str,
    cls: type[ProspectiveEvidenceUse] | type[RetrospectiveEvidenceUse],
    commitment_digest: str,
    use_digest: str,
) -> Any:
    """Missing telemetry stays unknown; malformed or rebound telemetry fails closed."""
    if not isinstance(container, dict) or "continuity_evidence" not in container:
        return None
    records = container["continuity_evidence"]
    if type(records) is not dict or len(records) > 128:
        raise ValueError("continuity_evidence must be a bounded object")
    if key not in records:
        return None
    data = records[key]
    if type(data) is not dict or type(data.get("provenance")) is not list:
        raise ValueError("stage observation must be an object with provenance")
    try:
        result = cls(**(data | {"provenance": tuple(data["provenance"])}))
    except TypeError as error:
        raise ValueError("invalid stage observation fields") from error
    if result.commitment_digest != commitment_digest or result.evidence_use_digest != use_digest:
        raise ValueError("stage observation digest does not match original inputs")
    return result


def _classification(
    before: ProspectiveEvidenceUse | None,
    after: RetrospectiveEvidenceUse | None,
    use: EvidenceUseAssessment,
    continuity: EpistemicContinuityAssessment,
    key: str,
) -> str:
    """Preserve observed historical failures independently of later legitimate updates."""
    if before is None:
        return "unresolved_omission"
    if before.recognized is True and before.retained is False:
        return "retention_failure"
    expected_use = use.target_influence is not MemoryInfluence.IGNORE
    if expected_use and before.retained is True and before.retrieved is False:
        return "retrieval_failure"
    if (
        (before.observed_kind is not None and before.observed_kind != use.kind)
        or (
            before.observed_influence is not None
            and before.observed_influence != use.target_influence
        )
        or before.prediction_reflected is (not expected_use)
        or (after is not None and after.action_reflected is (not expected_use))
    ):
        return "influence_failure"
    if after is not None and after.preserved_after_outcome is False:
        return "retention_failure"
    if (
        any(value is not True for value in (before.recognized, before.retained))
        or (expected_use and before.retrieved is not True)
        or before.observed_kind is None
        or before.observed_influence is None
        or after is None
        or after.preserved_after_outcome is not True
        or after.reported_unavailable is None
        or before.prediction_reflected is None
        or after.action_reflected is None
        or key in continuity.invalid_update_ids
        or key in continuity.quarantined_ids
    ):
        return "unresolved_omission"
    if key in continuity.legitimately_scoped_out_ids:
        return "legitimate_scope_change"
    if key in continuity.explicitly_superseded_ids:
        return "legitimate_update"
    if key in continuity.unexplained_omission_ids or after.reported_unavailable:
        return "unresolved_omission"
    return "consistent"


def audit_peo_continuity(
    request: PEOContinuityRequest | None = None, *, enabled: bool = False
) -> PEOContinuityAudit | None:
    """Audit host-observed evidence use; disabled calls do not read the request.

    This checks declared influence against the host's target, not text presence or private
    reasoning. Host authentication and semantic validation of stage observations remain external.
    Numeric residuals describe prediction mismatch, not contradiction of the commitment itself.
    """
    boolean(enabled, "enabled")
    if not enabled:
        return None
    if type(request) is not PEOContinuityRequest:
        raise ValueError("enabled audit requires PEOContinuityRequest")
    if type(request.events) is not tuple or len(request.events) > 10_000:
        raise ValueError("events must be a bounded tuple")
    # Detach nested payloads before binding observations and window digests.
    events = tuple(EventEnvelope(**json.loads(json.dumps(e.to_dict()))) for e in request.events)
    window = EvidenceWindow.build(events, request.decision_event_id)
    continuity = assess_epistemic_continuity(
        assessment_id=f"peo:{request.commitment_id}",
        commitments=request.commitments,
        events=events,
        decision_event_id=request.decision_event_id,
        active_commitment_ids=request.active_commitment_ids,
        updates=request.updates,
        decision_context_changed=request.decision_context_changed,
        provenance=request.provenance,
    )
    items = {item.commitment_id: item for item in request.commitments}
    if request.commitment_id not in items:
        raise ValueError("commitment_id is not in commitments")
    item = replace(items[request.commitment_id])
    if not item.decision_relevance:
        raise ValueError("audited commitment must be decision relevant")
    use = replace(request.evidence_use)
    if (
        use.claim_id != item.commitment_id
        or use.memory != item.memory
        or use.claim.proposition != item.claim_summary
        or set(use.claim.evidence_refs) != set(item.evidence_refs)
    ):
        raise ValueError("evidence_use must describe the original commitment and evidence")
    by_id = {event.event_id: event for event in window.events}
    rec_event = by_id.get(request.reconciliation_event_id)
    if rec_event is None or rec_event.event_type != "epistemic_reconciliation":
        raise ValueError("reconciliation must precede the later decision")
    record = EpistemicReconciliation.from_dict(rec_event.payload)
    if any(binding.verifier_event_id is None for binding in record.bindings):
        raise ValueError("continuity requires explicitly verifier-bound reconciliation")
    prediction = by_id[record.prediction_event_id]
    outcome = by_id[record.observation_event_id]
    executed = by_id[outcome.parent_event_ids[0]]
    proposed = by_id[executed.parent_event_ids[0]]
    earlier = EvidenceWindow.build(events, proposed.event_id)
    if not _commitment_bound(item, earlier):
        raise ValueError("commitment evidence must be available before the audited action")
    positions = {event.event_id: i for i, event in enumerate(window.events)}
    sources = [by_id[key] for key in item.source_event_refs]
    if any(positions[source.event_id] >= positions[prediction.event_id] for source in sources):
        raise ValueError("commitment sources must precede the prediction")
    post = (
        by_id.get(request.assessment_event_id) if request.assessment_event_id is not None else None
    )
    if request.assessment_event_id is not None and (
        post is None
        or post.event_type != "epistemic_assessment"
        or set(post.parent_event_ids) != set(record.verification_event_ids)
        or positions[post.event_id] <= positions[rec_event.event_id]
    ):
        raise ValueError("assessment must follow reconciliation and bind its verifiers")
    ordered = [
        *sorted(sources, key=lambda event: positions[event.event_id]),
        prediction,
        proposed,
        executed,
        outcome,
        *(by_id[key] for key in record.verification_event_ids),
        rec_event,
    ]
    if post is not None:
        ordered.append(post)
    # Later update evidence is subject to the same wall-clock checks as the PEO cycle.
    for update in request.updates:
        if update.commitment_id != item.commitment_id:
            continue
        for key in update.source_event_refs:
            if key in by_id:
                if positions[key] <= positions[outcome.event_id]:
                    raise ValueError("later update evidence must follow the audited outcome")
                ordered.append(by_id[key])
    ordered = list({event.event_id: event for event in ordered}.values())
    ordered.sort(key=lambda event: positions[event.event_id])
    ordered.append(window.decision)
    _validate_context_and_time(ordered, use, prediction)
    digest = evidence_use_digest(use)
    before = _observation(
        prediction.to_dict()["payload"]["predicted_outcome"],
        item.commitment_id,
        ProspectiveEvidenceUse,
        item.digest,
        digest,
    )
    after = (
        _observation(
            post.to_dict()["payload"],
            item.commitment_id,
            RetrospectiveEvidenceUse,
            item.digest,
            digest,
        )
        if post
        else None
    )
    return PEOContinuityAudit(
        item.commitment_id,
        _classification(before, after, use, continuity, item.commitment_id),
        before,
        after,
        continuity,
        tuple((binding.measurement_id, binding.innovation.residual) for binding in record.bindings),
        tuple((event.event_id, event.timestamp) for event in ordered),
        json.dumps(use.to_dict(), sort_keys=True, allow_nan=False),
    )


def _validate_context_and_time(
    events: list[EventEnvelope], use: EvidenceUseAssessment, prediction: EventEnvelope
) -> None:
    """Check conversation, turn and wall time across the original evidence and PEO chain."""
    decision = events[-1]
    times = []
    steps = []
    for event in events:
        if (
            event.run_id != decision.run_id
            or event.repeat_id != decision.repeat_id
            or event.conversation_id != decision.conversation_id
            or event.model_version != decision.model_version
            or event.harness_version != decision.harness_version
            or type(event.checkpoint_step) is not int
        ):
            raise ValueError("continuity events must share evaluation identity and conversation")
        times.append(parse_rfc3339_datetime(event.timestamp, "event timestamp"))
        steps.append(event.checkpoint_step)
    if times != sorted(times) or steps != sorted(steps):
        raise ValueError("continuity chronology must follow timestamps and checkpoints")
    if parse_rfc3339_datetime(use.assessed_at, "assessed_at") > parse_rfc3339_datetime(
        prediction.timestamp, "prediction timestamp"
    ):
        raise ValueError("evidence_use must be assessed before prediction")
    use.measurement.context.validate_event(decision)
