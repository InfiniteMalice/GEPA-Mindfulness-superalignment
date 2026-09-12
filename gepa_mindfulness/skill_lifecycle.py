"""Authoritative, evidence-bound transitions for the verified skill lifecycle."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, cast
from weakref import WeakKeyDictionary

from evaluation.v5_records import V5EvaluationRecord
from mindful_trace_gepa.action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    VerificationResult,
)
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .core.evidence import EvidenceReference, EvidenceSourceKind
from .learning_surfaces import (
    EvaluationEpochHistory,
    evaluation_record_id,
    validate_epoch_record,
)
from .verification.state import WorldStateChange


class SkillLifecycleState(str, Enum):
    """A reviewable state in the verified skill lifecycle."""

    SOURCE_EXPERIENCE = "source_experience"
    VERIFIED_SKILL = "verified_skill"
    PROCEDURAL_FAMILY = "procedural_family"
    TASK_LOCAL = "task_local"
    EXECUTED = "executed"
    CREDITED = "credited"
    REFINED = "refined"
    HELD_OUT_VALIDATED = "held_out_validated"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


_ALLOWED_TRANSITIONS: Mapping[SkillLifecycleState, frozenset[SkillLifecycleState]] = (
    MappingProxyType(
        {
            SkillLifecycleState.SOURCE_EXPERIENCE: frozenset({SkillLifecycleState.VERIFIED_SKILL}),
            SkillLifecycleState.VERIFIED_SKILL: frozenset(
                {
                    SkillLifecycleState.PROCEDURAL_FAMILY,
                    SkillLifecycleState.TASK_LOCAL,
                    SkillLifecycleState.ROLLED_BACK,
                }
            ),
            SkillLifecycleState.PROCEDURAL_FAMILY: frozenset(
                {SkillLifecycleState.TASK_LOCAL, SkillLifecycleState.ROLLED_BACK}
            ),
            SkillLifecycleState.TASK_LOCAL: frozenset(
                {SkillLifecycleState.EXECUTED, SkillLifecycleState.ROLLED_BACK}
            ),
            SkillLifecycleState.EXECUTED: frozenset(
                {SkillLifecycleState.CREDITED, SkillLifecycleState.ROLLED_BACK}
            ),
            SkillLifecycleState.CREDITED: frozenset(
                {SkillLifecycleState.REFINED, SkillLifecycleState.ROLLED_BACK}
            ),
            SkillLifecycleState.REFINED: frozenset(
                {SkillLifecycleState.HELD_OUT_VALIDATED, SkillLifecycleState.ROLLED_BACK}
            ),
            SkillLifecycleState.HELD_OUT_VALIDATED: frozenset(
                {SkillLifecycleState.COMMITTED, SkillLifecycleState.ROLLED_BACK}
            ),
            SkillLifecycleState.COMMITTED: frozenset({SkillLifecycleState.ROLLED_BACK}),
            SkillLifecycleState.ROLLED_BACK: frozenset(),
        }
    )
)


@dataclass(frozen=True, slots=True)
class SkillArtifact:
    """One immutable, serializable snapshot of a skill lifecycle transition."""

    skill_id: str
    version: str
    state: SkillLifecycleState
    source_refs: tuple[EvidenceReference, ...]
    execution_event_ids: tuple[str, ...] = ()
    validation_record_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    _construction_binding: tuple[object, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate exact fields and detach caller-owned evidence references."""

        _require_token(self.skill_id, "skill_id")
        _require_token(self.version, "version")
        if type(self.state) is not SkillLifecycleState:
            raise ValueError("state must be an exact SkillLifecycleState")
        references = _snapshot_source_refs(self.source_refs)
        execution_ids = _snapshot_ids(self.execution_event_ids, "execution_event_ids")
        validation_ids = _snapshot_record_ids(self.validation_record_ids)
        if self.supersedes is not None:
            _require_token(self.supersedes, "supersedes")
        _validate_state_provenance(
            self.state,
            execution_ids,
            validation_ids,
            self.supersedes,
        )
        object.__setattr__(self, "source_refs", references)
        object.__setattr__(self, "execution_event_ids", execution_ids)
        object.__setattr__(self, "validation_record_ids", validation_ids)
        object.__setattr__(self, "_construction_binding", _artifact_binding(self))

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible detached snapshot."""

        snapshot = _snapshot_artifact(self)
        return {
            "skill_id": snapshot.skill_id,
            "version": snapshot.version,
            "state": snapshot.state.value,
            "source_refs": [reference.to_dict() for reference in snapshot.source_refs],
            "execution_event_ids": list(snapshot.execution_event_ids),
            "validation_record_ids": list(snapshot.validation_record_ids),
            "supersedes": snapshot.supersedes,
        }

    @classmethod
    def from_dict(cls, data: object) -> SkillArtifact:
        """Restore an artifact from its exact JSON-compatible shape."""

        values = _require_exact_mapping(
            data,
            {
                "skill_id",
                "version",
                "state",
                "source_refs",
                "execution_event_ids",
                "validation_record_ids",
                "supersedes",
            },
            "SkillArtifact",
        )
        raw_refs = _require_list(values["source_refs"], "source_refs")
        raw_execution = _require_list(values["execution_event_ids"], "execution_event_ids")
        raw_validation = _require_list(values["validation_record_ids"], "validation_record_ids")
        return cls(
            skill_id=cast(str, values["skill_id"]),
            version=cast(str, values["version"]),
            state=_parse_state(values["state"]),
            source_refs=tuple(EvidenceReference.from_dict(item) for item in raw_refs),
            execution_event_ids=tuple(cast(str, item) for item in raw_execution),
            validation_record_ids=tuple(cast(str, item) for item in raw_validation),
            supersedes=cast(str | None, values["supersedes"]),
        )


@dataclass(frozen=True, slots=True)
class SkillExecutionEvidence:
    """A bound action, observed outcome/world change, and positive verifier event."""

    action_event: EventEnvelope
    outcome_event: EventEnvelope
    verification_event: EventEnvelope
    world_change: WorldStateChange

    def __post_init__(self) -> None:
        action, outcome, verification, change = _snapshot_execution_evidence_fields(self)
        object.__setattr__(self, "action_event", action)
        object.__setattr__(self, "outcome_event", outcome)
        object.__setattr__(self, "verification_event", verification)
        object.__setattr__(self, "world_change", change)
        _validate_execution_evidence(self)

    @property
    def event_ids(self) -> tuple[str, str, str]:
        """Return the ordered action, outcome, and verification event identities."""

        snapshot = _snapshot_execution_evidence(self)
        return (
            snapshot.action_event.event_id,
            snapshot.outcome_event.event_id,
            snapshot.verification_event.event_id,
        )


@dataclass(frozen=True, slots=True)
class _LifecycleEntry:
    artifacts: tuple[SkillArtifact, ...]


@dataclass(frozen=True, slots=True)
class _HistoryLocator:
    store: SkillLifecycleStore
    skill_id: str


class SkillLifecycleStore:
    """Injected process-local authority for skill lineages and version nonreuse.

    The store rejects caller-created artifacts as transition authority. It is intentionally not
    durable or authenticated; a deployment needing cross-process authority must replace it with
    a protected transactional service.
    """

    __slots__ = ("__weakref__",)

    def __init__(self, authority_domain: str) -> None:
        domain = _require_token(authority_domain, "authority_domain")
        with _LIFECYCLE_LOCK:
            _LIFECYCLE_STORES[self] = (domain, {})

    def create_source(
        self,
        skill_id: str,
        version: str,
        source_refs: tuple[EvidenceReference, ...],
    ) -> SkillLifecycleHistory:
        """Create one exclusively owned source lineage."""

        _require_lifecycle_process()
        if type(self) is not SkillLifecycleStore or self not in _LIFECYCLE_STORES:
            raise ValueError("store must be an exact enrolled SkillLifecycleStore")
        source = SkillArtifact(
            skill_id,
            version,
            SkillLifecycleState.SOURCE_EXPERIENCE,
            source_refs,
        )
        with _LIFECYCLE_LOCK:
            domain, entries = _LIFECYCLE_STORES[self]
            del domain
            if source.skill_id in entries:
                raise ValueError("skill_id is already claimed in this authority domain")
            entries[source.skill_id] = _LifecycleEntry((source,))
            history = object.__new__(SkillLifecycleHistory)
            _LIFECYCLE_HISTORIES[history] = _HistoryLocator(self, source.skill_id)
        return history


class SkillLifecycleHistory:
    """Opaque handle to a complete store-owned lifecycle lineage."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("history handles are created by SkillLifecycleStore")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("SkillLifecycleHistory is read-only")

    def snapshot(self) -> tuple[SkillArtifact, ...]:
        """Return a detached non-authoritative lineage snapshot."""

        with _LIFECYCLE_LOCK:
            entry = _history_entry(self)
            return tuple(_snapshot_artifact(item) for item in entry.artifacts)

    def current(self) -> SkillArtifact:
        """Return the detached current artifact."""

        return self.snapshot()[-1]


_LIFECYCLE_PROCESS_ID = os.getpid()
_LIFECYCLE_LOCK = RLock()
_LIFECYCLE_STORES: WeakKeyDictionary[
    SkillLifecycleStore,
    tuple[str, dict[str, _LifecycleEntry]],
] = WeakKeyDictionary()
_LIFECYCLE_HISTORIES: WeakKeyDictionary[SkillLifecycleHistory, _HistoryLocator] = (
    WeakKeyDictionary()
)


def _reset_lifecycle_after_fork() -> None:
    global _LIFECYCLE_PROCESS_ID, _LIFECYCLE_LOCK
    global _LIFECYCLE_STORES, _LIFECYCLE_HISTORIES
    _LIFECYCLE_PROCESS_ID = os.getpid()
    _LIFECYCLE_LOCK = RLock()
    _LIFECYCLE_STORES = WeakKeyDictionary()
    _LIFECYCLE_HISTORIES = WeakKeyDictionary()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lifecycle_after_fork)


def transition_skill(
    history: SkillLifecycleHistory,
    target_state: SkillLifecycleState,
    *,
    execution_evidence: SkillExecutionEvidence | None = None,
    execution_event_ids: object = None,
    version: str | None = None,
    supersedes: str | None = None,
    evaluation_history: EvaluationEpochHistory | None = None,
    evaluation_epoch_id: str | None = None,
    validation_records: tuple[V5EvaluationRecord, ...] = (),
    rollback_target: SkillArtifact | None = None,
) -> SkillArtifact:
    """Apply one allowed transition to a store-owned lifecycle history."""

    if type(history) is not SkillLifecycleHistory:
        raise ValueError("transition authority requires an exact SkillLifecycleHistory")
    if type(target_state) is not SkillLifecycleState:
        raise ValueError("target_state must be an exact SkillLifecycleState")
    if execution_event_ids is not None:
        raise ValueError("execution_event_ids cannot replace structured execution evidence")
    with _LIFECYCLE_LOCK:
        entry = _history_entry(history)
        current = _snapshot_artifact(entry.artifacts[-1])
        if target_state not in _ALLOWED_TRANSITIONS[current.state]:
            raise ValueError(
                f"transition from {current.state.value} to {target_state.value} is forbidden"
            )
        _reject_irrelevant_inputs(
            target_state,
            execution_evidence,
            version,
            supersedes,
            evaluation_history,
            evaluation_epoch_id,
            validation_records,
            rollback_target,
        )
        execution_ids = current.execution_event_ids
        validation_ids = current.validation_record_ids
        next_version = current.version
        next_supersedes = current.supersedes
        if target_state is SkillLifecycleState.EXECUTED:
            if execution_evidence is None:
                raise ValueError("EXECUTED requires structured execution evidence")
            execution_ids = _snapshot_execution_evidence(execution_evidence).event_ids
        elif target_state is SkillLifecycleState.CREDITED:
            if len(execution_ids) != 3:
                raise ValueError("CREDITED requires retained structured execution evidence")
        elif target_state is SkillLifecycleState.REFINED:
            next_version, next_supersedes = _validate_refinement(
                entry,
                current,
                version,
                supersedes,
            )
        elif target_state is SkillLifecycleState.HELD_OUT_VALIDATED:
            validation_ids = _validate_held_out_records(
                evaluation_history,
                evaluation_epoch_id,
                validation_records,
            )
        elif target_state is SkillLifecycleState.COMMITTED:
            target = _validated_rollback_target(entry, current, rollback_target)
            if current.supersedes != target.version:
                raise ValueError("rollback target must match the refined supersedes version")
        elif target_state is SkillLifecycleState.ROLLED_BACK:
            target = _validated_rollback_target(entry, current, rollback_target)
            next_supersedes = target.version
        updated = SkillArtifact(
            current.skill_id,
            next_version,
            target_state,
            current.source_refs,
            execution_ids,
            validation_ids,
            next_supersedes,
        )
        locator = _LIFECYCLE_HISTORIES[history]
        _LIFECYCLE_STORES[locator.store][1][locator.skill_id] = _LifecycleEntry(
            (*entry.artifacts, updated)
        )
        return _snapshot_artifact(updated)


def _reject_irrelevant_inputs(
    target: SkillLifecycleState,
    execution: object,
    version: object,
    supersedes: object,
    evaluation_history: object,
    evaluation_epoch_id: object,
    validation_records: object,
    rollback_target: object,
) -> None:
    if target is not SkillLifecycleState.EXECUTED and execution is not None:
        raise ValueError("execution_evidence is accepted only for EXECUTED")
    if target is not SkillLifecycleState.REFINED and (
        version is not None or supersedes is not None
    ):
        raise ValueError("version and supersedes are accepted only for REFINED")
    held_out_values = (evaluation_history, evaluation_epoch_id)
    if target is not SkillLifecycleState.HELD_OUT_VALIDATED and (
        any(value is not None for value in held_out_values) or validation_records != ()
    ):
        raise ValueError("evaluation evidence is accepted only for HELD_OUT_VALIDATED")
    if target not in {SkillLifecycleState.COMMITTED, SkillLifecycleState.ROLLED_BACK}:
        if rollback_target is not None:
            raise ValueError("rollback_target is accepted only for COMMITTED or ROLLED_BACK")


def _validate_refinement(
    entry: _LifecycleEntry,
    current: SkillArtifact,
    version: object,
    supersedes: object,
) -> tuple[str, str]:
    next_version = _require_token(version, "version")
    prior = _require_token(supersedes, "supersedes")
    if prior != current.version:
        raise ValueError("supersedes must identify the credited source version")
    if next_version == prior or next_version in {item.version for item in entry.artifacts}:
        raise ValueError("refined version must be new within the lifecycle history")
    return next_version, prior


def _validate_held_out_records(
    history: object,
    epoch_id: object,
    records: object,
) -> tuple[str, ...]:
    if type(history) is not EvaluationEpochHistory:
        raise ValueError("held-out validation requires an authoritative EvaluationEpochHistory")
    identifier = _require_token(epoch_id, "evaluation_epoch_id")
    if type(records) is not tuple or not records:
        raise ValueError("validation_records must be a nonempty exact tuple")
    epochs = cast(EvaluationEpochHistory, history).snapshot()
    matches = tuple(epoch for epoch in epochs if epoch.epoch_id == identifier)
    if len(matches) != 1:
        raise ValueError("evaluation_epoch_id is absent from the authoritative history")
    epoch = matches[0]
    if not epoch.closed:
        raise ValueError("held-out evaluation epoch must be closed")
    record_ids: list[str] = []
    for record in cast(tuple[object, ...], records):
        if type(record) is not V5EvaluationRecord:
            raise ValueError("validation_records require exact V5EvaluationRecord values")
        checked = cast(V5EvaluationRecord, record)
        validate_epoch_record(epoch, checked)
        record_ids.append(evaluation_record_id(checked))
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("validation_records must not contain duplicates")
    return tuple(record_ids)


def _validated_rollback_target(
    entry: _LifecycleEntry,
    current: SkillArtifact,
    target: object,
) -> SkillArtifact:
    snapshot = _snapshot_artifact(target)
    if snapshot.skill_id != current.skill_id:
        raise ValueError("rollback target must belong to the same skill_id")
    if snapshot.state is SkillLifecycleState.SOURCE_EXPERIENCE:
        raise ValueError("rollback target must be a verified lifecycle artifact")
    if not any(snapshot == item for item in entry.artifacts):
        raise ValueError("rollback target must be an exact prior authoritative artifact")
    return snapshot


def _history_entry(history: object) -> _LifecycleEntry:
    _require_lifecycle_process()
    if type(history) is not SkillLifecycleHistory or history not in _LIFECYCLE_HISTORIES:
        raise ValueError("transition authority requires an enrolled SkillLifecycleHistory")
    locator = _LIFECYCLE_HISTORIES[cast(SkillLifecycleHistory, history)]
    store_state = _LIFECYCLE_STORES.get(locator.store)
    if store_state is None or locator.skill_id not in store_state[1]:
        raise ValueError("authoritative skill lifecycle no longer exists")
    entry = store_state[1][locator.skill_id]
    _validate_lineage(entry)
    return entry


def _validate_lineage(entry: object) -> None:
    if type(entry) is not _LifecycleEntry or type(entry.artifacts) is not tuple:
        raise ValueError("authoritative skill lifecycle entry is invalid")
    if not entry.artifacts:
        raise ValueError("authoritative skill lifecycle cannot be empty")
    artifacts = tuple(_snapshot_artifact(item) for item in entry.artifacts)
    first = artifacts[0]
    if first.state is not SkillLifecycleState.SOURCE_EXPERIENCE:
        raise ValueError("authoritative skill lifecycle must begin with source experience")
    for previous, current in zip(artifacts, artifacts[1:], strict=False):
        if current.state not in _ALLOWED_TRANSITIONS[previous.state]:
            raise ValueError("authoritative skill lifecycle contains an illegal transition")
        if current.skill_id != first.skill_id or current.source_refs != first.source_refs:
            raise ValueError("authoritative skill lifecycle identity or source evidence changed")


def _snapshot_execution_evidence(value: object) -> SkillExecutionEvidence:
    if type(value) is not SkillExecutionEvidence:
        raise ValueError("EXECUTED requires exact structured execution evidence")
    action, outcome, verification, change = _snapshot_execution_evidence_fields(value)
    return SkillExecutionEvidence(action, outcome, verification, change)


def _snapshot_execution_evidence_fields(
    value: object,
) -> tuple[EventEnvelope, EventEnvelope, EventEnvelope, WorldStateChange]:
    if type(value) is not SkillExecutionEvidence:
        raise ValueError("execution_evidence must be an exact SkillExecutionEvidence")
    checked = cast(SkillExecutionEvidence, value)
    return (
        _snapshot_event(checked.action_event),
        _snapshot_event(checked.outcome_event),
        _snapshot_event(checked.verification_event),
        _snapshot_world_change(checked.world_change),
    )


def _validate_execution_evidence(value: SkillExecutionEvidence) -> None:
    action_event = value.action_event
    outcome_event = value.outcome_event
    verification_event = value.verification_event
    for event in (action_event, outcome_event, verification_event):
        _require_token(event.event_id, "event_id")
        _require_token(event.schema_version, "schema_version")
        _require_token(event.timestamp, "timestamp")
    if action_event.event_type != StructuredEventType.ACTION_EXECUTED.value:
        raise ValueError("action_event must be an executed action")
    action = _action_from_event(action_event)
    outcome = _outcome_from_event(outcome_event)
    verification = _verification_from_event(verification_event)
    if outcome_event.event_type != StructuredEventType.OUTCOME_OBSERVED.value:
        raise ValueError("outcome_event must be an observed outcome")
    if verification_event.event_type != StructuredEventType.VERIFICATION_RESULT.value:
        raise ValueError("verification_event must be a verification result")
    if not verification.verified:
        raise ValueError("verification_event must positively verify the observed outcome")
    if action.action_id != outcome.action_id or action.action_id != value.world_change.action_id:
        raise ValueError("execution evidence must bind one exact action_id")
    if verification.observation_id != outcome.observation_id:
        raise ValueError("verification evidence must bind the exact observed outcome")
    if action_event.action_id != action.action_id or outcome_event.action_id != action.action_id:
        raise ValueError("event metadata must bind the exact action_id")
    if verification_event.action_id != action.action_id:
        raise ValueError("verification event metadata must bind the exact action_id")
    if action_event.authorization_scope != action.authorization_scope:
        raise ValueError("action event metadata must bind the exact authorization_scope")
    if outcome_event.evidence_refs != outcome.evidence_refs:
        raise ValueError("outcome event metadata must bind the exact evidence_refs")
    if verification_event.verifier_refs != verification.verifier_refs:
        raise ValueError("verification event metadata must bind the exact verifier_refs")
    if action_event.event_id not in outcome_event.parent_event_ids:
        raise ValueError("outcome event must descend from the executed action event")
    if outcome_event.event_id not in verification_event.parent_event_ids:
        raise ValueError("verification event must descend from the observed outcome event")
    if len({action_event.event_id, outcome_event.event_id, verification_event.event_id}) != 3:
        raise ValueError("execution evidence event identities must be unique")
    world_refs = {item.reference_id for item in value.world_change.after_observation.evidence_refs}
    if not world_refs.intersection(outcome.evidence_refs):
        raise ValueError("world change and outcome must share observable evidence identity")
    observed_outcome = _require_exact_mapping(
        outcome.actual_outcome,
        {"artifact_ref", "digest"},
        "observed artifact outcome",
    )
    if (
        observed_outcome["artifact_ref"] != value.world_change.artifact_ref
        or observed_outcome["digest"] != value.world_change.after_digest
    ):
        raise ValueError("outcome must match the exact world-change artifact observation")
    _validate_event_context(action_event, outcome_event, verification_event)


def _validate_event_context(*events: EventEnvelope) -> None:
    token_fields = (
        "run_id",
        "model_version",
        "harness_version",
        "case_version",
        "stripe_id",
    )
    for field_name in token_fields:
        first = getattr(events[0], field_name)
        _require_token(first, field_name)
        if any(getattr(event, field_name) != first for event in events[1:]):
            raise ValueError(f"execution evidence must share exact nonempty {field_name}")
    for field_name in ("case_id", "repeat_id", "seed"):
        first = getattr(events[0], field_name)
        if type(first) is not int:
            raise ValueError(f"execution evidence {field_name} must be an exact integer")
        if field_name != "seed" and first < 0:
            raise ValueError(f"execution evidence {field_name} must be nonnegative")
        if any(type(getattr(event, field_name)) is not int for event in events[1:]):
            raise ValueError(f"execution evidence {field_name} must be an exact integer")
        if any(getattr(event, field_name) != first for event in events[1:]):
            raise ValueError(f"execution evidence must share exact {field_name}")


def _action_from_event(event: EventEnvelope) -> ActionRecord:
    values = _require_exact_mapping(
        event.payload,
        {"action_id", "action_class", "reversible", "authorization_scope", "prediction_commit_id"},
        "action_event payload",
    )
    return ActionRecord(**cast(Any, dict(values)))


def _outcome_from_event(event: EventEnvelope) -> OutcomeObservation:
    values = _require_exact_mapping(
        event.payload,
        {"observation_id", "action_id", "actual_outcome", "evidence_refs"},
        "outcome_event payload",
    )
    return OutcomeObservation(
        cast(str, values["observation_id"]),
        cast(str, values["action_id"]),
        values["actual_outcome"],
        tuple(cast(list[str], values["evidence_refs"])),
    )


def _verification_from_event(event: EventEnvelope) -> VerificationResult:
    values = _require_exact_mapping(
        event.payload,
        {"verifier_id", "verifier_version", "observation_id", "verified", "verifier_refs"},
        "verification_event payload",
    )
    return VerificationResult(
        cast(str, values["verifier_id"]),
        cast(str, values["verifier_version"]),
        cast(str, values["observation_id"]),
        cast(bool, values["verified"]),
        tuple(cast(list[str], values["verifier_refs"])),
    )


def _snapshot_event(value: object) -> EventEnvelope:
    if type(value) is not EventEnvelope:
        raise ValueError("execution evidence events must be exact EventEnvelope values")
    try:
        return EventEnvelope(**cast(Any, cast(EventEnvelope, value).to_dict()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"execution evidence contains an invalid event: {exc}") from exc


def _snapshot_world_change(value: object) -> WorldStateChange:
    if type(value) is not WorldStateChange:
        raise ValueError("world_change must be an exact WorldStateChange")
    try:
        world_change_type = cast(Any, WorldStateChange)
        return cast(WorldStateChange, world_change_type.from_dict(cast(Any, value).to_dict()))
    except (TypeError, ValueError) as exc:
        raise ValueError("world_change must remain valid") from exc


def _snapshot_artifact(value: object) -> SkillArtifact:
    if type(value) is not SkillArtifact:
        raise ValueError("rollback target must be an exact SkillArtifact")
    checked = cast(SkillArtifact, value)
    if checked._construction_binding != _artifact_binding(checked):
        raise ValueError("SkillArtifact construction binding changed after construction")
    return SkillArtifact(
        checked.skill_id,
        checked.version,
        checked.state,
        checked.source_refs,
        checked.execution_event_ids,
        checked.validation_record_ids,
        checked.supersedes,
    )


def _artifact_binding(value: SkillArtifact) -> tuple[object, ...]:
    references = tuple(
        (reference.reference_id, reference.source_kind.value) for reference in value.source_refs
    )
    return (
        value.skill_id,
        value.version,
        value.state.value if type(value.state) is SkillLifecycleState else value.state,
        references,
        value.execution_event_ids,
        value.validation_record_ids,
        value.supersedes,
    )


def _validate_state_provenance(
    state: SkillLifecycleState,
    execution_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    supersedes: str | None,
) -> None:
    execution_states = {
        SkillLifecycleState.EXECUTED,
        SkillLifecycleState.CREDITED,
        SkillLifecycleState.REFINED,
        SkillLifecycleState.HELD_OUT_VALIDATED,
        SkillLifecycleState.COMMITTED,
        SkillLifecycleState.ROLLED_BACK,
    }
    if state not in execution_states and execution_ids:
        raise ValueError("execution_event_ids are forbidden before EXECUTED")
    if state in execution_states and state is not SkillLifecycleState.ROLLED_BACK:
        if len(execution_ids) != 3:
            raise ValueError("execution_event_ids require the exact three-event execution chain")
    validated_states = {
        SkillLifecycleState.HELD_OUT_VALIDATED,
        SkillLifecycleState.COMMITTED,
    }
    if state not in validated_states | {SkillLifecycleState.ROLLED_BACK} and validation_ids:
        raise ValueError("validation_record_ids are forbidden before HELD_OUT_VALIDATED")
    if state in validated_states and not validation_ids:
        raise ValueError("validation_record_ids are required after held-out validation")
    if state in validated_states | {SkillLifecycleState.REFINED} and supersedes is None:
        raise ValueError("refined and validated artifacts require supersedes provenance")


def _snapshot_source_refs(value: object) -> tuple[EvidenceReference, ...]:
    if type(value) is not tuple:
        raise ValueError("source_refs must be an exact tuple")
    references: list[EvidenceReference] = []
    for reference in cast(tuple[object, ...], value):
        if type(reference) is not EvidenceReference:
            raise ValueError("source_refs require exact EvidenceReference values")
        checked = cast(EvidenceReference, reference)
        _require_token(checked.reference_id, "source_refs reference_id")
        if type(checked.source_kind) is not EvidenceSourceKind:
            raise ValueError("source_refs source_kind must be an exact EvidenceSourceKind")
        evidence_type = cast(Any, EvidenceReference)
        references.append(
            cast(EvidenceReference, evidence_type(checked.reference_id, checked.source_kind))
        )
    if not references or not any(reference.is_observable for reference in references):
        raise ValueError("source_refs require at least one observable reference")
    identities = tuple((item.reference_id, item.source_kind) for item in references)
    if len(set(identities)) != len(identities):
        raise ValueError("source_refs must be unique and ordered")
    return tuple(references)


def _snapshot_ids(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be an exact tuple")
    values = tuple(_require_token(item, field_name) for item in cast(tuple[object, ...], value))
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique ordered identities")
    return values


def _snapshot_record_ids(value: object) -> tuple[str, ...]:
    values = _snapshot_ids(value, "validation_record_ids")
    for item in values:
        if (
            len(item) != 71
            or not item.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in item[7:])
        ):
            raise ValueError("validation_record_ids must contain canonical SHA-256 identities")
    return values


def _require_token(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a nonblank exact string without outer whitespace")
    return value


def _parse_state(value: object) -> SkillLifecycleState:
    if type(value) is not str:
        raise ValueError("state must be a built-in string enum value")
    try:
        return SkillLifecycleState(value)
    except ValueError as exc:
        raise ValueError(f"unknown skill lifecycle state {value!r}") from exc


def _require_exact_mapping(
    value: object,
    fields: set[str],
    record_name: str,
) -> Mapping[str, object]:
    if type(value) not in {dict, MappingProxyType}:
        raise ValueError(f"{record_name} must be an exact object")
    checked = cast(Mapping[str, object], value)
    if set(checked) != fields:
        raise ValueError(f"{record_name} requires exactly {sorted(fields)!r}")
    return checked


def _require_list(value: object, field_name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be an array")
    return cast(list[object], value)


def _require_lifecycle_process() -> None:
    if os.getpid() != _LIFECYCLE_PROCESS_ID:
        raise RuntimeError("skill lifecycle authority cannot cross a process boundary")


__all__ = [
    "SkillArtifact",
    "SkillExecutionEvidence",
    "SkillLifecycleHistory",
    "SkillLifecycleState",
    "SkillLifecycleStore",
    "transition_skill",
]
