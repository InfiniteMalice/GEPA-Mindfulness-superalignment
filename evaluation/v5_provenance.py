"""Bind V5 evaluation records to validated PR-2 action-bound event sequences."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol, cast, runtime_checkable

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
_CASE_ASSESSMENT_VALUES = frozenset({"pass", "fail"})
_EPISTEMIC_ASSESSMENT_VALUES = frozenset({"verified", "unverified"})


@runtime_checkable
class V5ProvenanceResult(Protocol):
    """Read-only shape returned by the V5 provenance validator."""

    @property
    def run_id(self) -> str:
        """Return the one validated run identity."""

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Return the validated event identities in supplied order."""

    def record_snapshot(self) -> V5EvaluationRecord:
        """Return a fresh validated record snapshot."""

    def optimizer_scores(self) -> dict[str, object]:
        """Return the five validated optimizer-facing score values."""


@dataclass(frozen=True, slots=True, init=False)
class _VerifiedV5Evaluation:
    """Private immutable result whose only constructor performs full validation."""

    run_id: str
    event_ids: tuple[str, ...]
    _record_json: str

    def __init__(
        self,
        record: V5EvaluationRecord,
        events: Sequence[EventEnvelope],
    ) -> None:
        """Validate caller inputs before creating any optimizer-capable result."""

        record_snapshot, event_snapshot, run_id = _validated_inputs(record, events)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(
            self,
            "event_ids",
            tuple(event.event_id for event in event_snapshot),
        )
        object.__setattr__(
            self,
            "_record_json",
            json.dumps(
                record_snapshot.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

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
) -> V5ProvenanceResult:
    """Validate and snapshot one record plus its same-cell PR-2 event sequence."""

    return _VerifiedV5Evaluation(record, events)


def _validated_inputs(
    record: V5EvaluationRecord,
    events: Sequence[EventEnvelope],
) -> tuple[V5EvaluationRecord, tuple[EventEnvelope, ...], str]:
    """Return detached inputs only after every V5 provenance check succeeds."""

    if type(record) is not V5EvaluationRecord:
        raise ValueError("record must be an exact V5EvaluationRecord")
    record_snapshot = V5EvaluationRecord.from_dict(record.to_dict())
    event_snapshot = _snapshot_events(events)
    validate_action_bound_sequence(event_snapshot)
    run_id = _validate_cell_identity(record_snapshot, event_snapshot)
    _validate_record_links(record_snapshot, event_snapshot)
    return record_snapshot, event_snapshot, run_id


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
    observed_passes = tuple(_observed_passed(observation) for observation in observations)
    if any(passed is not record.outcome.passed for passed in observed_passes):
        raise ValueError("observed outcome passed value must match record.outcome.passed")

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
        process_verified = _active_epistemic_assessment_verified(
            events,
            frozenset(record.epistemics.verifier_refs),
        )
        if not process_verified:
            raise ValueError("active epistemic_assessment must be verified for positive process")

    if record.outcome.passed:
        if not action_events:
            raise ValueError("outcome.passed=True requires nonempty action_refs")
        if not observation_events:
            raise ValueError("outcome.passed=True requires nonempty observation_refs")
    if not outcome_results:
        raise ValueError("outcome provenance requires verifier_refs")
    if any(not result.verified for result in outcome_results):
        raise ValueError("outcome provenance requires verified=True results")

    case_passed = _active_case_assessment_passed(
        events,
        frozenset(record.outcome.verifier_refs),
    )
    if case_passed is not record.outcome.passed:
        raise ValueError("active case_assessment passed value must match record.outcome.passed")


def _observed_passed(observation: OutcomeObservation) -> bool:
    """Read the exact V5 pass field from one typed observed outcome."""

    outcome = observation.actual_outcome
    if not isinstance(outcome, Mapping) or set(outcome) != {"passed"}:
        raise ValueError("observed outcome must contain exactly the passed field")
    passed = outcome["passed"]
    if type(passed) is not bool:
        raise ValueError("observed outcome passed value must be a built-in bool")
    return passed


def _active_case_assessment_passed(
    events: tuple[EventEnvelope, ...],
    required_verifiers: frozenset[str],
) -> bool:
    """Return the sole active case result for the record's exact verifier ancestry."""

    active_epistemics = {
        event.event_id: event
        for event in events
        if event.event_type == _EPISTEMIC and event.superseded_by is None
    }
    matching: list[EventEnvelope] = []
    for event in events:
        if event.event_type != _CASE or event.superseded_by is not None:
            continue
        try:
            epistemic_parents = tuple(
                active_epistemics[parent] for parent in event.parent_event_ids
            )
        except KeyError:
            continue
        parent_verifiers = frozenset(
            parent_id for parent in epistemic_parents for parent_id in parent.parent_event_ids
        )
        if parent_verifiers == required_verifiers:
            matching.append(event)

    if len(matching) != 1:
        raise ValueError(
            "outcome provenance requires one active case_assessment for its verifier ancestry"
        )
    assessment = _assessment_value(
        matching[0],
        _CASE_ASSESSMENT_VALUES,
        "case_assessment",
    )
    return assessment == "pass"


def _active_epistemic_assessment_verified(
    events: tuple[EventEnvelope, ...],
    required_verifiers: frozenset[str],
) -> bool:
    """Return the sole active process result for the record's exact verifier ancestry."""

    matching = tuple(
        event
        for event in events
        if event.event_type == _EPISTEMIC
        and event.superseded_by is None
        and frozenset(event.parent_event_ids) == required_verifiers
    )
    if len(matching) != 1:
        raise ValueError(
            "positive epistemic_process requires one active epistemic_assessment "
            "for its verifier ancestry"
        )
    assessment = _assessment_value(
        matching[0],
        _EPISTEMIC_ASSESSMENT_VALUES,
        "epistemic_assessment",
    )
    return assessment == "verified"


def _assessment_value(
    event: EventEnvelope,
    allowed_values: frozenset[str],
    event_name: str,
) -> str:
    """Read one exact V5 assessment mapping and reject untyped or ambiguous payloads."""

    if not isinstance(event.payload, Mapping) or set(event.payload) != {"assessment"}:
        raise ValueError(f"{event_name} payload must contain exactly the assessment field")
    assessment = event.payload["assessment"]
    if type(assessment) is not str or assessment not in allowed_values:
        expected = " or ".join(repr(value) for value in sorted(allowed_values))
        raise ValueError(f"{event_name} assessment must be the exact built-in string {expected}")
    return assessment


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


__all__ = ["V5ProvenanceResult", "validate_v5_record_provenance"]
