"""Causal validation for immutable action-bound structured event envelopes.

The validator performs a bounded identity pre-pass because a derived assessment may name a later
supersession target. Causal parent references are always checked while traversing the event stream
and therefore may only point backward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
)
from .logging_schema import EventEnvelope, StructuredEventType

_PREDICTION = StructuredEventType.PREDICTION_COMMIT.value
_PROPOSED = StructuredEventType.ACTION_PROPOSED.value
_EXECUTED = StructuredEventType.ACTION_EXECUTED.value
_OBSERVATION = StructuredEventType.OUTCOME_OBSERVED.value
_VERIFICATION = StructuredEventType.VERIFICATION_RESULT.value
_EPISTEMIC = StructuredEventType.EPISTEMIC_ASSESSMENT.value
_CASE = StructuredEventType.CASE_ASSESSMENT.value

_ACTION_BOUND_TYPES = frozenset(
    {_PREDICTION, _PROPOSED, _EXECUTED, _OBSERVATION, _VERIFICATION, _EPISTEMIC, _CASE}
)
_DERIVED_TYPES = frozenset({_EPISTEMIC, _CASE})
_Payload = TypeVar(
    "_Payload",
    PredictionCommit,
    ActionRecord,
    OutcomeObservation,
    VerificationResult,
)


@dataclass(frozen=True)
class EvaluatedSystemVersion:
    """The immutable model and harness identity evaluated in one repetition."""

    model_version: str
    harness_version: str

    def __post_init__(self) -> None:
        """Reject version identifiers that cannot be audited."""

        _require_nonblank("model_version", self.model_version)
        _require_nonblank("harness_version", self.harness_version)


def validate_action_bound_sequence(events: Sequence[EventEnvelope]) -> None:
    """Validate causal, payload, version, and supersession contracts for action-bound events.

    Legacy and unrelated event types are accepted without interpretation. Action-bound records fail
    closed: each one must supply a stable evaluation identity plus typed payload and causal links.
    A ``repeat_id`` of ``None`` is a valid, distinct evaluation unit for its ``run_id``.
    """

    event_list = tuple(events)
    event_positions = _index_event_ids(event_list)
    _validate_supersession_targets(event_list, event_positions)

    versions: dict[tuple[str, int | None], EvaluatedSystemVersion] = {}
    prediction_events: dict[str, EventEnvelope] = {}
    proposed_actions: dict[str, tuple[ActionRecord, EventEnvelope]] = {}
    executed_actions: dict[str, EventEnvelope] = {}
    observations: dict[str, EventEnvelope] = {}
    verifications: set[str] = set()
    seen_events: dict[str, EventEnvelope] = {}
    resolved_action_ids: dict[str, str] = {}

    for event in event_list:
        if event.event_type not in _ACTION_BOUND_TYPES:
            seen_events[event.event_id] = event
            continue

        unit = _validate_evaluation_identity(event, versions)
        _validate_parent_references(event, seen_events, event_positions)
        _reject_raw_supersession(event)

        if event.event_type == _PREDICTION:
            prediction = _parse_payload(event, PredictionCommit)
            _require_matching_refs(event, prediction.evidence_refs, "evidence_refs")
            if prediction.prediction_commit_id in prediction_events:
                raise ValueError(
                    "duplicate prediction_commit_id is an immutable prediction rewrite"
                )
            prediction_events[prediction.prediction_commit_id] = event
        elif event.event_type == _PROPOSED:
            action = _parse_payload(event, ActionRecord)
            _require_action_metadata(event, action)
            prediction_event = prediction_events.get(action.prediction_commit_id)
            if prediction_event is None:
                raise ValueError("action_proposed requires an earlier prediction_commit")
            _require_direct_parent(event, prediction_event, _PREDICTION)
            _require_same_unit(event, prediction_event, unit)
            if action.action_id in proposed_actions:
                raise ValueError("duplicate proposed action_id")
            proposed_actions[action.action_id] = (action, event)
            resolved_action_ids[event.event_id] = action.action_id
        elif event.event_type == _EXECUTED:
            action = _parse_payload(event, ActionRecord)
            _require_action_metadata(event, action)
            proposal = proposed_actions.get(action.action_id)
            if proposal is None:
                raise ValueError("action_executed requires an earlier action_proposed")
            proposal_payload, proposal_event = proposal
            if action != proposal_payload:
                raise ValueError("action_id may not rewrite the proposed action payload")
            _require_direct_parent(event, proposal_event, _PROPOSED)
            _require_same_unit(event, proposal_event, unit)
            if action.action_id in executed_actions:
                raise ValueError("duplicate executed action_id")
            executed_actions[action.action_id] = event
            resolved_action_ids[event.event_id] = action.action_id
        elif event.event_type == _OBSERVATION:
            observation = _parse_payload(event, OutcomeObservation)
            _require_matching_value(event, "action_id", observation.action_id)
            _require_matching_refs(event, observation.evidence_refs, "evidence_refs")
            execution_event = executed_actions.get(observation.action_id)
            if execution_event is None:
                raise ValueError("outcome_observed requires an earlier action_executed")
            _require_direct_parent(event, execution_event, _EXECUTED)
            _require_same_unit(event, execution_event, unit)
            if observation.observation_id in observations:
                raise ValueError("duplicate observation_id")
            observations[observation.observation_id] = event
            resolved_action_ids[event.event_id] = observation.action_id
        elif event.event_type == _VERIFICATION:
            result = _parse_payload(event, VerificationResult)
            _require_matching_refs(event, result.verifier_refs, "verifier_refs")
            observation_event = observations.get(result.observation_id)
            if observation_event is None:
                raise ValueError("verification_result requires an earlier outcome_observed")
            _require_direct_parent(event, observation_event, _OBSERVATION)
            _require_same_unit(event, observation_event, unit)
            action_id = _resolve_parent_action_id(
                event,
                (observation_event,),
                resolved_action_ids,
            )
            if result.verifier_id in verifications:
                raise ValueError("duplicate verifier_id")
            verifications.add(result.verifier_id)
            resolved_action_ids[event.event_id] = action_id
        elif event.event_type == _EPISTEMIC:
            parents = _require_derived_parents(event, seen_events, _VERIFICATION)
            for parent in parents:
                _require_same_unit(event, parent, unit)
            resolved_action_ids[event.event_id] = _resolve_parent_action_id(
                event,
                parents,
                resolved_action_ids,
            )
        else:
            parents = _require_derived_parents(event, seen_events, _EPISTEMIC)
            for parent in parents:
                _require_same_unit(event, parent, unit)
            resolved_action_ids[event.event_id] = _resolve_parent_action_id(
                event,
                parents,
                resolved_action_ids,
            )

        seen_events[event.event_id] = event

    _validate_supersession_action_identity(event_list, resolved_action_ids)


def _index_event_ids(events: tuple[EventEnvelope, ...]) -> dict[str, int]:
    """Build the bounded global ID index needed for unique identities and supersession targets."""

    positions: dict[str, int] = {}
    for index, event in enumerate(events):
        if not isinstance(event, EventEnvelope):
            raise TypeError("events must contain EventEnvelope instances")
        _require_nonblank("event_id", event.event_id)
        if event.event_id in positions:
            raise ValueError("duplicate event_id")
        positions[event.event_id] = index
    return positions


def _validate_evaluation_identity(
    event: EventEnvelope,
    versions: dict[tuple[str, int | None], EvaluatedSystemVersion],
) -> tuple[str, int | None]:
    """Require and freeze one model/harness pair for each run and repeat evaluation unit."""

    run_id = _required_string("run_id", event.run_id)
    version = EvaluatedSystemVersion(
        _required_string("model_version", event.model_version),
        _required_string("harness_version", event.harness_version),
    )
    unit = (run_id, event.repeat_id)
    previous = versions.setdefault(unit, version)
    if previous != version:
        if previous.model_version != version.model_version:
            raise ValueError("model_version drift within one run_id and repeat_id")
        raise ValueError("harness_version drift within one run_id and repeat_id")
    return unit


def _validate_parent_references(
    event: EventEnvelope,
    seen_events: Mapping[str, EventEnvelope],
    event_positions: Mapping[str, int],
) -> None:
    """Require every action-bound parent reference to name one distinct earlier event."""

    parents = event.parent_event_ids
    if len(parents) != len(set(parents)):
        raise ValueError("duplicate parent_event_ids are not allowed")
    for parent_id in parents:
        if parent_id not in seen_events:
            if parent_id in event_positions:
                raise ValueError(f"{event.event_type} has a forward parent reference")
            raise ValueError(f"{event.event_type} has an unknown parent reference")


def _reject_raw_supersession(event: EventEnvelope) -> None:
    """Keep raw evidence and action records append-only."""

    if event.event_type not in _DERIVED_TYPES and event.superseded_by is not None:
        raise ValueError("superseded_by is allowed only for derived assessment events")


def _parse_payload(event: EventEnvelope, payload_type: type[_Payload]) -> _Payload:
    """Reconstruct and validate one typed payload without trusting caller-owned mappings."""

    if not isinstance(event.payload, Mapping):
        raise ValueError(f"{payload_type.__name__} payload must be a mapping")
    expected_fields = set(payload_type.__dataclass_fields__)
    actual_fields = set(event.payload)
    if actual_fields != expected_fields:
        raise ValueError(f"{payload_type.__name__} payload fields do not match its typed contract")
    try:
        return payload_type(**dict(event.payload))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {payload_type.__name__} payload: {error}") from error


def _require_action_metadata(event: EventEnvelope, action: ActionRecord) -> None:
    """Bind action semantic fields in the envelope to their typed immutable payload."""

    _require_matching_value(event, "action_id", action.action_id)
    _require_matching_value(event, "authorization_scope", action.authorization_scope)


def _require_matching_value(event: EventEnvelope, field_name: str, expected: str) -> None:
    """Reject an absent or conflicting envelope semantic value."""

    if getattr(event, field_name) != expected:
        raise ValueError(f"{field_name} must match the typed payload")


def _require_matching_refs(
    event: EventEnvelope,
    expected: tuple[str, ...],
    field_name: str,
) -> None:
    """Reject absent or conflicting immutable payload provenance references."""

    if getattr(event, field_name) != expected:
        raise ValueError(f"{field_name} must match the typed payload")


def _require_direct_parent(
    event: EventEnvelope,
    parent: EventEnvelope | None,
    expected_type: str,
) -> None:
    """Require the one direct earlier causal parent of the expected structured event type."""

    if parent is None:
        raise ValueError(f"{event.event_type} requires an earlier {expected_type}")
    if event.parent_event_ids != (parent.event_id,):
        raise ValueError(f"{event.event_type} parent_event_ids must be the earlier {expected_type}")


def _require_derived_parents(
    event: EventEnvelope,
    seen_events: Mapping[str, EventEnvelope],
    expected_type: str,
) -> tuple[EventEnvelope, ...]:
    """Resolve one or more direct parents of the required derived/evidence event type."""

    if not event.parent_event_ids:
        raise ValueError(f"{event.event_type} requires at least one parent_event_ids reference")
    parents: list[EventEnvelope] = []
    for parent_id in event.parent_event_ids:
        parent = seen_events.get(parent_id)
        if parent is None or parent.event_type != expected_type:
            raise ValueError(f"{event.event_type} requires earlier {expected_type} parents")
        parents.append(parent)
    return tuple(parents)


def _resolve_parent_action_id(
    event: EventEnvelope,
    parents: tuple[EventEnvelope, ...],
    resolved_action_ids: Mapping[str, str],
) -> str:
    """Resolve one unambiguous action from causal parents and check optional metadata."""

    try:
        action_ids = {resolved_action_ids[parent.event_id] for parent in parents}
    except KeyError as error:
        raise ValueError(f"{event.event_type} parent lacks resolved action ancestry") from error
    if len(action_ids) != 1:
        raise ValueError(f"{event.event_type} parents must resolve to one action")
    action_id = next(iter(action_ids))
    if event.action_id is not None and event.action_id != action_id:
        raise ValueError("action_id must match resolved causal ancestry")
    return action_id


def _require_same_unit(
    event: EventEnvelope,
    parent: EventEnvelope,
    unit: tuple[str, int | None],
) -> None:
    """Prevent an action-bound causal edge from crossing run or repeat identity."""

    if (parent.run_id, parent.repeat_id) != unit:
        raise ValueError("causal parent must belong to the same evaluation unit")


def _validate_supersession_targets(
    events: tuple[EventEnvelope, ...],
    positions: dict[str, int],
) -> None:
    """Validate forward-only, single-source supersession of derived assessments in O(n)."""

    sources_by_target: dict[str, str] = {}
    for index, event in enumerate(events):
        if event.event_type not in _ACTION_BOUND_TYPES or event.superseded_by is None:
            continue
        if event.event_type not in _DERIVED_TYPES:
            raise ValueError("superseded_by is allowed only for derived assessment events")
        target_index = positions.get(event.superseded_by)
        if target_index is None:
            raise ValueError("superseded_by references an unknown event")
        if target_index <= index:
            raise ValueError("superseded_by target must be later in the sequence")
        target = events[target_index]
        if target.event_type != event.event_type:
            raise ValueError("superseded_by target must have the same event type")
        if (target.run_id, target.repeat_id) != (event.run_id, event.repeat_id):
            raise ValueError("superseded_by target must be in the same evaluation unit")
        if target.superseded_by is not None:
            raise ValueError("supersession chains are not allowed")
        if event.superseded_by in sources_by_target:
            raise ValueError("multiple superseders for one target are not allowed")
        sources_by_target[event.superseded_by] = event.event_id


def _validate_supersession_action_identity(
    events: tuple[EventEnvelope, ...],
    resolved_action_ids: Mapping[str, str],
) -> None:
    """Keep each derived replacement within its source event's resolved action ancestry."""

    for event in events:
        if event.event_type not in _DERIVED_TYPES or event.superseded_by is None:
            continue
        if resolved_action_ids[event.event_id] != resolved_action_ids[event.superseded_by]:
            raise ValueError("superseded_by target must resolve to the same action ancestry")


def _required_string(field_name: str, value: object) -> str:
    """Return a required nonblank string after validating it."""

    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")
    return value


def _require_nonblank(field_name: str, value: object) -> None:
    """Require a stable nonblank string identity."""

    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


__all__ = ["EvaluatedSystemVersion", "validate_action_bound_sequence"]
