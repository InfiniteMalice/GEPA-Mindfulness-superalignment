"""Bind V5 evaluation records to validated PR-2 action-bound event sequences."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, cast

from mindful_trace_gepa.action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
)
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .v5_records import ScoreRecord, V5EvaluationRecord

_PREDICTION = StructuredEventType.PREDICTION_COMMIT.value
_PROPOSED = StructuredEventType.ACTION_PROPOSED.value
_EXECUTED = StructuredEventType.ACTION_EXECUTED.value
_OBSERVATION = StructuredEventType.OUTCOME_OBSERVED.value
_VERIFICATION = StructuredEventType.VERIFICATION_RESULT.value
_EPISTEMIC = StructuredEventType.EPISTEMIC_ASSESSMENT.value
_CASE = StructuredEventType.CASE_ASSESSMENT.value
_ACTION_TYPES = frozenset({_PROPOSED, _EXECUTED})
_ACTION_BOUND_TYPES = frozenset(
    {_PREDICTION, _PROPOSED, _EXECUTED, _OBSERVATION, _VERIFICATION, _EPISTEMIC, _CASE}
)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedV5Evaluation:
    """An immutable record snapshot minted only after provenance validation."""

    run_id: str
    event_ids: tuple[str, ...]
    _record_json: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Prevent callers from constructing an unvalidated verification wrapper."""

        raise TypeError("VerifiedV5Evaluation is produced by validate_v5_record_provenance")

    @classmethod
    def _from_validated(
        cls,
        record: V5EvaluationRecord,
        run_id: str,
        events: tuple[EventEnvelope, ...],
    ) -> "VerifiedV5Evaluation":
        """Create a wrapper from already snapshotted and validated inputs."""

        instance = object.__new__(cls)
        object.__setattr__(instance, "run_id", run_id)
        object.__setattr__(instance, "event_ids", tuple(event.event_id for event in events))
        object.__setattr__(
            instance,
            "_record_json",
            json.dumps(
                record.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        return instance

    def record_snapshot(self) -> V5EvaluationRecord:
        """Return a fresh validated record detached from the verified internal snapshot."""

        payload = json.loads(self._record_json)
        if not isinstance(payload, Mapping):  # pragma: no cover - guarded by private construction
            raise ValueError("verified V5 record snapshot must decode to a JSON object")
        return V5EvaluationRecord.from_dict(payload)

    def optimizer_scores(self) -> dict[str, object]:
        """Revalidate and return the five optimizer-facing score values."""

        scores = self.record_snapshot().scores
        validated = ScoreRecord(
            correctness=scores.correctness,
            calibration=scores.calibration,
            abstention=scores.abstention,
            epistemic_process=scores.epistemic_process,
            total=scores.total,
        )
        return {
            "correctness": validated.correctness,
            "calibration": validated.calibration,
            "abstention": validated.abstention,
            "epistemic_process": validated.epistemic_process,
            "total": validated.total,
        }


def validate_v5_record_provenance(
    record: V5EvaluationRecord,
    events: Sequence[EventEnvelope],
) -> VerifiedV5Evaluation:
    """Validate and snapshot one record plus its same-cell PR-2 event sequence."""

    if type(record) is not V5EvaluationRecord:
        raise ValueError("record must be an exact V5EvaluationRecord")
    record_snapshot = V5EvaluationRecord.from_dict(record.to_dict())
    event_snapshot = _snapshot_events(events)
    validate_action_bound_sequence(event_snapshot)
    run_id = _validate_cell_identity(record_snapshot, event_snapshot)
    _validate_record_links(record_snapshot, event_snapshot)
    return VerifiedV5Evaluation._from_validated(record_snapshot, run_id, event_snapshot)


def _snapshot_events(events: object) -> tuple[EventEnvelope, ...]:
    """Iterate one event sequence once and detach every exact envelope from its caller."""

    if isinstance(events, (str, bytes, bytearray)) or not isinstance(events, Sequence):
        raise ValueError("events must be a non-string sequence of EventEnvelope instances")
    supplied = tuple(events)
    if not supplied:
        raise ValueError("events must not be empty")
    snapshots: list[EventEnvelope] = []
    for event in supplied:
        if type(event) is not EventEnvelope:
            raise ValueError("events must contain only exact EventEnvelope instances")
        values = {field.name: getattr(event, field.name) for field in fields(EventEnvelope)}
        snapshots.append(EventEnvelope(**cast(Any, values)))
    return tuple(snapshots)


def _validate_cell_identity(
    record: V5EvaluationRecord,
    events: tuple[EventEnvelope, ...],
) -> str:
    """Require every action-bound event to identify the record's exact evaluation cell."""

    action_events = tuple(event for event in events if event.event_type in _ACTION_BOUND_TYPES)
    if not action_events:
        raise ValueError("events must contain an action-bound V5 sequence")
    expected = {
        "repeat_id": record.system.repeat_id,
        "model_version": record.system.model_version,
        "harness_version": record.system.harness_version,
        "case_version": record.case.case_version,
        "case_id": record.case.case_id,
        "stripe_id": record.robustness.stripe_id,
        "seed": record.system.seed,
    }
    run_id: str | None = None
    for event in action_events:
        _require_exact_string(event.event_id, "event_id")
        _require_exact_string(event.event_type, "event_type")
        event_run_id = _require_exact_string(event.run_id, "run_id")
        if run_id is None:
            run_id = event_run_id
        elif event_run_id != run_id:
            raise ValueError("run_id drift within the supplied V5 cell sequence")
        for field_name, expected_value in expected.items():
            actual = getattr(event, field_name)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{field_name} must match the V5 record cell; received {actual!r}")
        for field_name in ("parent_event_ids", "evidence_refs", "verifier_refs"):
            for reference in getattr(event, field_name):
                _require_exact_string(reference, field_name)
    if run_id is None:  # pragma: no cover - action_events is nonempty
        raise ValueError("action-bound V5 sequence must identify one run_id")
    return run_id


def _validate_record_links(
    record: V5EvaluationRecord,
    events: tuple[EventEnvelope, ...],
) -> None:
    """Resolve record references and enforce verified pass/process ancestry."""

    events_by_id = {event.event_id: event for event in events}
    prediction_event = _resolve_one(
        record.epistemics.prediction_ref,
        events_by_id,
        {_PREDICTION},
        "prediction_ref",
    )
    action_events = _resolve_many(
        record.behavior.action_refs,
        events_by_id,
        _ACTION_TYPES,
        "action_refs",
    )
    observation_events = _resolve_many(
        record.outcome.observation_refs,
        events_by_id,
        {_OBSERVATION},
        "observation_refs",
    )
    outcome_verifications = _resolve_many(
        record.outcome.verifier_refs,
        events_by_id,
        {_VERIFICATION},
        "outcome verifier_refs",
    )
    process_verifications = _resolve_many(
        record.epistemics.verifier_refs,
        events_by_id,
        {_VERIFICATION},
        "epistemic verifier_refs",
    )

    prediction = _payload(prediction_event, PredictionCommit)
    if prediction.confidence != record.epistemics.confidence:
        raise ValueError("confidence must match the referenced prediction_commit")
    actions = tuple(_payload(event, ActionRecord) for event in action_events)
    if any(action.prediction_commit_id != prediction.prediction_commit_id for action in actions):
        raise ValueError("action_refs must resolve to the referenced prediction_commit")
    action_ids = {action.action_id for action in actions}

    observations = tuple(_payload(event, OutcomeObservation) for event in observation_events)
    if any(observation.action_id not in action_ids for observation in observations):
        raise ValueError("observation_refs must resolve through record action_refs")
    observation_ids = {observation.observation_id for observation in observations}

    outcome_results = tuple(_payload(event, VerificationResult) for event in outcome_verifications)
    process_results = tuple(_payload(event, VerificationResult) for event in process_verifications)
    if any(result.observation_id not in observation_ids for result in outcome_results):
        raise ValueError("outcome verifier_refs must resolve through observation_refs")
    if any(result.observation_id not in observation_ids for result in process_results):
        raise ValueError("epistemic verifier_refs must resolve through observation_refs")

    evidence_boundary = set(prediction.evidence_refs)
    for observation in observations:
        evidence_boundary.update(observation.evidence_refs)
    unresolved_evidence = set(record.epistemics.evidence_refs) - evidence_boundary
    if unresolved_evidence:
        raise ValueError(
            "epistemic evidence_refs are unresolved by the linked prediction or observations"
        )

    if record.scores.epistemic_process > 0.0:
        if not record.epistemics.evidence_refs:
            raise ValueError("positive epistemic_process requires nonempty evidence_refs")
        if not process_results:
            raise ValueError("positive epistemic_process requires verifier_refs")
        if any(not result.verified for result in process_results):
            raise ValueError("positive epistemic_process requires verified=True results")
        required_verifiers = set(record.epistemics.verifier_refs)
        assessment_found = any(
            event.event_type == _EPISTEMIC and required_verifiers.issubset(event.parent_event_ids)
            for event in events
        )
        if not assessment_found:
            raise ValueError("positive epistemic_process requires a linked epistemic_assessment")

    if record.outcome.passed:
        if not action_events:
            raise ValueError("outcome.passed=True requires nonempty action_refs")
        if not observation_events:
            raise ValueError("outcome.passed=True requires nonempty observation_refs")
        if not outcome_results:
            raise ValueError("outcome.passed=True requires verifier_refs")
        if any(not result.verified for result in outcome_results):
            raise ValueError("outcome.passed=True requires verified=True results")


def _resolve_one(
    reference: str,
    events_by_id: Mapping[str, EventEnvelope],
    allowed_types: set[str] | frozenset[str],
    field_name: str,
) -> EventEnvelope:
    """Resolve one event ID and enforce its semantic event role."""

    resolved = events_by_id.get(reference)
    if resolved is None:
        raise ValueError(f"{field_name} is unresolved: {reference!r}")
    if resolved.event_type not in allowed_types:
        expected = " or ".join(sorted(allowed_types))
        raise ValueError(f"{field_name} must reference {expected}; received {resolved.event_type}")
    return resolved


def _resolve_many(
    references: tuple[str, ...],
    events_by_id: Mapping[str, EventEnvelope],
    allowed_types: set[str] | frozenset[str],
    field_name: str,
) -> tuple[EventEnvelope, ...]:
    """Resolve every event ID in one record reference collection."""

    return tuple(
        _resolve_one(reference, events_by_id, allowed_types, field_name) for reference in references
    )


def _payload(event: EventEnvelope, payload_type: type[Any]) -> Any:
    """Rehydrate the typed payload already accepted by the PR-2 sequence validator."""

    if not isinstance(event.payload, Mapping):  # pragma: no cover - PR-2 validator enforces this
        raise ValueError(f"{payload_type.__name__} payload must be a mapping")
    return payload_type(**dict(event.payload))


def _require_exact_string(value: object, field_name: str) -> str:
    """Return a boundary-trimmed exact built-in string."""

    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be an exact nonblank built-in string")
    return value


__all__ = ["VerifiedV5Evaluation", "validate_v5_record_provenance"]
