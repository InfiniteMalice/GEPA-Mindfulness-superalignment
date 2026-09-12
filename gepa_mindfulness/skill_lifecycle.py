"""Durable, evidence-bound authority for the verified skill lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, TypeAlias, cast
from uuid import uuid4
from weakref import WeakKeyDictionary

from mindful_trace_gepa.action_bound_events import ActionRecord, OutcomeObservation
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .core.evidence import EvidenceReference, EvidenceSourceKind
from .learning_surfaces import EvaluationEpochStore, ValidationReceipt
from .verification.interfaces import LocalVerificationResult, RelationalVerificationResult
from .verification.state import WorldStateChange

_RFC3339_OFFSET_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
    r"(?:Z|[+-](?P<offset_hour>\d{2}):(?P<offset_minute>\d{2}))$"
)


class SkillLifecycleState(str, Enum):
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
class PruningProvenance:
    target_artifact_id: str
    reason: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        _require_token(self.target_artifact_id, "target_artifact_id")
        _require_token(self.reason, "reason")
        object.__setattr__(self, "evidence_refs", _snapshot_refs(self.evidence_refs))

    def to_dict(self) -> dict[str, object]:
        checked = _snapshot_pruning(self)
        return {
            "kind": "pruning",
            "target_artifact_id": checked.target_artifact_id,
            "reason": checked.reason,
            "evidence_refs": [item.to_dict() for item in checked.evidence_refs],
        }


@dataclass(frozen=True, slots=True)
class ConsolidationProvenance:
    constituent_task_local_artifact_ids: tuple[str, ...]
    pruning_decisions: tuple[PruningProvenance, ...] = ()

    def __post_init__(self) -> None:
        identifiers = _snapshot_ids(
            self.constituent_task_local_artifact_ids,
            "constituent_task_local_artifact_ids",
            required=True,
        )
        if type(self.pruning_decisions) is not tuple:
            raise ValueError("pruning_decisions must be an exact tuple")
        decisions = tuple(_snapshot_pruning(item) for item in self.pruning_decisions)
        targets = tuple(item.target_artifact_id for item in decisions)
        if len(set(targets)) != len(targets):
            raise ValueError("pruning_decisions must have unique targets")
        object.__setattr__(self, "constituent_task_local_artifact_ids", identifiers)
        object.__setattr__(self, "pruning_decisions", decisions)

    def to_dict(self) -> dict[str, object]:
        checked = _snapshot_consolidation(self)
        return {
            "kind": "consolidation",
            "constituent_task_local_artifact_ids": list(
                checked.constituent_task_local_artifact_ids
            ),
            "pruning_decisions": [item.to_dict() for item in checked.pruning_decisions],
        }


@dataclass(frozen=True, slots=True)
class InstantiationProvenance:
    family_parent_artifact_id: str
    task_context: str

    def __post_init__(self) -> None:
        _require_token(self.family_parent_artifact_id, "family_parent_artifact_id")
        _require_token(self.task_context, "task_context")

    def to_dict(self) -> dict[str, object]:
        checked = _snapshot_instantiation(self)
        return {
            "kind": "instantiation",
            "family_parent_artifact_id": checked.family_parent_artifact_id,
            "task_context": checked.task_context,
        }


@dataclass(frozen=True, slots=True)
class RefinementProvenance:
    change_rationale: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        _require_token(self.change_rationale, "change_rationale")
        object.__setattr__(self, "evidence_refs", _snapshot_refs(self.evidence_refs))

    def to_dict(self) -> dict[str, object]:
        checked = _snapshot_refinement(self)
        return {
            "kind": "refinement",
            "change_rationale": checked.change_rationale,
            "evidence_refs": [item.to_dict() for item in checked.evidence_refs],
        }


TransitionProvenance: TypeAlias = (
    ConsolidationProvenance | InstantiationProvenance | RefinementProvenance | PruningProvenance
)


@dataclass(frozen=True, slots=True)
class ExecutionEvidenceBundle:
    """Exact action, outcome/world change, and both PR-5 verifier levels."""

    action_event: EventEnvelope
    outcome_event: EventEnvelope
    world_change: WorldStateChange
    local_verification_event: EventEnvelope
    relational_verification_event: EventEnvelope

    def __post_init__(self) -> None:
        fields = _snapshot_bundle_fields(self)
        for name, value in zip(self.__dataclass_fields__, fields, strict=True):
            object.__setattr__(self, name, value)
        _validated_bundle(self)


@dataclass(frozen=True, slots=True)
class ExecutionEvidenceReceipt:
    receipt_id: str
    bundle_digest: str
    action_id: str
    observation_id: str
    change_id: str
    artifact_ref: str
    artifact_digest: str
    event_ids: tuple[str, ...]
    evidence_ref_ids: tuple[str, ...]
    verifier_ref_ids: tuple[str, ...]
    _construction_binding: tuple[object, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "receipt_id",
            "action_id",
            "observation_id",
            "change_id",
            "artifact_ref",
        ):
            _require_token(getattr(self, field_name), field_name)
        _require_sha256(self.bundle_digest, "bundle_digest")
        _require_digest(self.artifact_digest, "artifact_digest")
        object.__setattr__(
            self, "event_ids", _snapshot_ids(self.event_ids, "event_ids", required=True)
        )
        object.__setattr__(
            self,
            "evidence_ref_ids",
            _snapshot_ids(self.evidence_ref_ids, "evidence_ref_ids", required=True),
        )
        object.__setattr__(
            self,
            "verifier_ref_ids",
            _snapshot_ids(self.verifier_ref_ids, "verifier_ref_ids", required=True),
        )
        object.__setattr__(self, "_construction_binding", _execution_receipt_binding(self))

    def to_dict(self) -> dict[str, object]:
        checked = _snapshot_execution_receipt(self)
        return {
            "receipt_id": checked.receipt_id,
            "bundle_digest": checked.bundle_digest,
            "action_id": checked.action_id,
            "observation_id": checked.observation_id,
            "change_id": checked.change_id,
            "artifact_ref": checked.artifact_ref,
            "artifact_digest": checked.artifact_digest,
            "event_ids": list(checked.event_ids),
            "evidence_ref_ids": list(checked.evidence_ref_ids),
            "verifier_ref_ids": list(checked.verifier_ref_ids),
        }

    @classmethod
    def from_dict(cls, data: object) -> ExecutionEvidenceReceipt:
        values = _exact_mapping(
            data,
            {
                "receipt_id",
                "bundle_digest",
                "action_id",
                "observation_id",
                "change_id",
                "artifact_ref",
                "artifact_digest",
                "event_ids",
                "evidence_ref_ids",
                "verifier_ref_ids",
            },
            "ExecutionEvidenceReceipt",
        )
        return cls(
            cast(str, values["receipt_id"]),
            cast(str, values["bundle_digest"]),
            cast(str, values["action_id"]),
            cast(str, values["observation_id"]),
            cast(str, values["change_id"]),
            cast(str, values["artifact_ref"]),
            cast(str, values["artifact_digest"]),
            tuple(cast(str, item) for item in _exact_list(values["event_ids"], "event_ids")),
            tuple(
                cast(str, item)
                for item in _exact_list(values["evidence_ref_ids"], "evidence_ref_ids")
            ),
            tuple(
                cast(str, item)
                for item in _exact_list(values["verifier_ref_ids"], "verifier_ref_ids")
            ),
        )


@dataclass(frozen=True, slots=True)
class SkillArtifact:
    skill_id: str
    version: str
    state: SkillLifecycleState
    source_refs: tuple[EvidenceReference, ...]
    artifact_id: str
    revision: int
    execution_receipt: ExecutionEvidenceReceipt | None = None
    validation_receipt: ValidationReceipt | None = None
    supersedes: str | None = None
    rollback_target_id: str | None = None
    transition_provenance: TransitionProvenance | None = None
    _construction_binding: tuple[object, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_token(self.skill_id, "skill_id")
        _require_token(self.version, "version")
        if type(self.state) is not SkillLifecycleState:
            raise ValueError("state must be an exact SkillLifecycleState")
        object.__setattr__(self, "source_refs", _snapshot_refs(self.source_refs))
        _require_token(self.artifact_id, "artifact_id")
        _require_nonnegative_int(self.revision, "revision")
        if self.execution_receipt is not None:
            object.__setattr__(
                self,
                "execution_receipt",
                _snapshot_execution_receipt(self.execution_receipt),
            )
        if self.validation_receipt is not None:
            receipt_type = cast(Any, ValidationReceipt)
            object.__setattr__(
                self,
                "validation_receipt",
                cast(
                    ValidationReceipt,
                    receipt_type.from_dict(cast(Any, self.validation_receipt).to_dict()),
                ),
            )
        if self.supersedes is not None:
            _require_token(self.supersedes, "supersedes")
        if self.rollback_target_id is not None:
            _require_token(self.rollback_target_id, "rollback_target_id")
        object.__setattr__(
            self,
            "transition_provenance",
            _snapshot_optional_provenance(self.transition_provenance),
        )
        _validate_artifact_state(self)
        object.__setattr__(self, "_construction_binding", _artifact_binding(self))

    @property
    def execution_event_ids(self) -> tuple[str, ...]:
        return () if self.execution_receipt is None else self.execution_receipt.event_ids

    @property
    def validation_record_ids(self) -> tuple[str, ...]:
        return () if self.validation_receipt is None else self.validation_receipt.record_ids

    def to_dict(self) -> dict[str, object]:
        checked = _snapshot_artifact(self)
        return {
            "skill_id": checked.skill_id,
            "version": checked.version,
            "state": checked.state.value,
            "source_refs": [item.to_dict() for item in checked.source_refs],
            "artifact_id": checked.artifact_id,
            "revision": checked.revision,
            "execution_receipt": (
                None if checked.execution_receipt is None else checked.execution_receipt.to_dict()
            ),
            "validation_receipt": (
                None if checked.validation_receipt is None else checked.validation_receipt.to_dict()
            ),
            "supersedes": checked.supersedes,
            "rollback_target_id": checked.rollback_target_id,
            "transition_provenance": _provenance_to_dict(checked.transition_provenance),
        }

    @classmethod
    def from_dict(cls, data: object) -> SkillArtifact:
        values = _exact_mapping(
            data,
            {
                "skill_id",
                "version",
                "state",
                "source_refs",
                "artifact_id",
                "revision",
                "execution_receipt",
                "validation_receipt",
                "supersedes",
                "rollback_target_id",
                "transition_provenance",
            },
            "SkillArtifact",
        )
        state = _parse_state(values["state"])
        raw_execution = values["execution_receipt"]
        raw_validation = values["validation_receipt"]
        validation_type = cast(Any, ValidationReceipt)
        return cls(
            cast(str, values["skill_id"]),
            cast(str, values["version"]),
            state,
            tuple(
                EvidenceReference.from_dict(item)
                for item in _exact_list(values["source_refs"], "source_refs")
            ),
            cast(str, values["artifact_id"]),
            cast(int, values["revision"]),
            None if raw_execution is None else ExecutionEvidenceReceipt.from_dict(raw_execution),
            (
                None
                if raw_validation is None
                else cast(ValidationReceipt, validation_type.from_dict(raw_validation))
            ),
            cast(str | None, values["supersedes"]),
            cast(str | None, values["rollback_target_id"]),
            _provenance_from_dict(values["transition_provenance"]),
        )


@dataclass(frozen=True, slots=True)
class _HistoryState:
    database_path: str
    authority_domain: str
    catalog_id: str
    skill_id: str
    revision: int


class SkillLifecycleStore:
    """SQLite-backed lifecycle authority scoped to one catalog and authority domain.

    Transactions and revisions protect against duplicate or stale runtime-owner operations.
    Database file permissions, backups, and operator authentication remain deployment concerns.
    The same authority-domain text in another catalog is intentionally a different trust scope.
    """

    __slots__ = ("_authority_domain", "_binding", "_catalog_id", "_database_path")
    _authority_domain: str
    _binding: tuple[str, str, str]
    _catalog_id: str
    _database_path: str

    def __init__(self, database_path: str | os.PathLike[str], authority_domain: str) -> None:
        path = os.path.abspath(os.fspath(database_path))
        domain = _require_token(authority_domain, "authority_domain")
        if not os.path.isdir(os.path.dirname(path)):
            raise ValueError("skill lifecycle store parent directory must already exist")
        with _open_database(path) as connection:
            _initialize_database(connection)
            catalog_id = _catalog_id(connection)
        object.__setattr__(self, "_database_path", path)
        object.__setattr__(self, "_authority_domain", domain)
        object.__setattr__(self, "_catalog_id", catalog_id)
        object.__setattr__(self, "_binding", (path, domain, catalog_id))

    def create_source(
        self,
        skill_id: str,
        version: str,
        source_refs: tuple[EvidenceReference, ...],
    ) -> SkillLifecycleHistory:
        path, domain, catalog = _validated_store(self)
        source = SkillArtifact(
            skill_id,
            version,
            SkillLifecycleState.SOURCE_EXPERIENCE,
            source_refs,
            str(uuid4()),
            0,
        )
        payload = _serialize_entry((source,))
        with _open_database(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO skill_lineages VALUES (?, ?, 0, ?)",
                    (domain, source.skill_id, payload),
                )
                _claim_artifact(connection, domain, source)
                connection.execute(
                    "INSERT INTO skill_versions VALUES (?, ?, ?)",
                    (domain, source.skill_id, source.version),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError("skill_id or version is already claimed in this catalog") from exc
        return _new_history(path, domain, catalog, source.skill_id, 0)

    def open(self, skill_id: str) -> SkillLifecycleHistory:
        path, domain, catalog = _validated_store(self)
        identifier = _require_token(skill_id, "skill_id")
        with _open_database(path) as connection:
            row = connection.execute(
                "SELECT revision, payload FROM skill_lineages "
                "WHERE authority_domain = ? AND skill_id = ?",
                (domain, identifier),
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            revision = _require_nonnegative_int(row[0], "revision")
            _validated_entry(connection, domain, identifier, row[1])
        return _new_history(path, domain, catalog, identifier, revision)


class SkillLifecycleHistory:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("history handles are created by SkillLifecycleStore")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("SkillLifecycleHistory is read-only")

    def snapshot(self) -> tuple[SkillArtifact, ...]:
        state, connection, entry = _begin_read(self)
        del state
        connection.close()
        return tuple(_snapshot_artifact(item) for item in entry)

    def current(self) -> SkillArtifact:
        return self.snapshot()[-1]


_PROCESS_ID = os.getpid()
_HISTORY_LOCK = RLock()
_HISTORY_STATE: WeakKeyDictionary[SkillLifecycleHistory, _HistoryState] = WeakKeyDictionary()


def _reset_after_fork() -> None:
    global _PROCESS_ID, _HISTORY_LOCK, _HISTORY_STATE
    _PROCESS_ID = os.getpid()
    _HISTORY_LOCK = RLock()
    _HISTORY_STATE = WeakKeyDictionary()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def transition_skill(
    history: SkillLifecycleHistory,
    target_state: SkillLifecycleState,
    *,
    provenance: TransitionProvenance | None = None,
    execution_evidence: ExecutionEvidenceBundle | None = None,
    version: str | None = None,
    supersedes: str | None = None,
    validation_store: EvaluationEpochStore | None = None,
    validation_receipt: ValidationReceipt | None = None,
    rollback_target_id: str | None = None,
) -> SkillArtifact:
    """Atomically apply the sole allowed transition to an authoritative history."""

    if type(history) is not SkillLifecycleHistory:
        raise ValueError("transition authority requires an exact SkillLifecycleHistory")
    if type(target_state) is not SkillLifecycleState:
        raise ValueError("target_state must be an exact SkillLifecycleState")
    with _HISTORY_LOCK:
        state, connection, entry = _begin_transition(history)
        current = entry[-1]
        try:
            if target_state not in _ALLOWED_TRANSITIONS[current.state]:
                raise ValueError(
                    f"transition from {current.state.value} to {target_state.value} is forbidden"
                )
            checked_provenance = _validate_transition_provenance(
                connection,
                state.authority_domain,
                current,
                target_state,
                provenance,
            )
            next_version = current.version
            next_supersedes = current.supersedes
            execution_receipt = current.execution_receipt
            held_out_receipt = current.validation_receipt
            target_id: str | None = None
            if target_state is SkillLifecycleState.EXECUTED:
                if type(execution_evidence) is not ExecutionEvidenceBundle:
                    raise ValueError("EXECUTED requires an exact ExecutionEvidenceBundle")
                execution_receipt = _issue_execution_receipt(execution_evidence)
            elif execution_evidence is not None:
                raise ValueError("execution_evidence is accepted only for EXECUTED")
            if target_state is SkillLifecycleState.CREDITED:
                _revalidate_execution_receipt(execution_receipt)
            if target_state is SkillLifecycleState.REFINED:
                next_version = _require_token(version, "version")
                next_supersedes = _require_token(supersedes, "supersedes")
                if next_supersedes != current.version:
                    raise ValueError("supersedes must identify the credited source version")
                if connection.execute(
                    "SELECT 1 FROM skill_versions WHERE authority_domain = ? "
                    "AND skill_id = ? AND version = ?",
                    (state.authority_domain, current.skill_id, next_version),
                ).fetchone():
                    raise ValueError("refined version must be new in the authoritative lineage")
                connection.execute(
                    "INSERT INTO skill_versions VALUES (?, ?, ?)",
                    (state.authority_domain, current.skill_id, next_version),
                )
            elif version is not None or supersedes is not None:
                raise ValueError("version and supersedes are accepted only for REFINED")
            if target_state is SkillLifecycleState.HELD_OUT_VALIDATED:
                if type(validation_store) is not EvaluationEpochStore:
                    raise ValueError("validation_store must be an exact EvaluationEpochStore")
                if type(validation_receipt) is not ValidationReceipt:
                    raise ValueError("validation_receipt must be an exact ValidationReceipt")
                held_out_receipt = validation_store.validate_validation_receipt(validation_receipt)
                if held_out_receipt.candidate_version != current.version:
                    raise ValueError(
                        "validation receipt candidate_version must match skill version"
                    )
            elif validation_store is not None or validation_receipt is not None:
                raise ValueError("validation receipt is accepted only for HELD_OUT_VALIDATED")
            if target_state in {
                SkillLifecycleState.COMMITTED,
                SkillLifecycleState.ROLLED_BACK,
            }:
                target = _resolve_rollback_target(
                    entry,
                    current,
                    rollback_target_id,
                    target_state,
                )
                target_id = target.artifact_id
                if target_state is SkillLifecycleState.COMMITTED:
                    if current.supersedes != target.version:
                        raise ValueError("commit rollback target must match supersedes version")
                else:
                    next_supersedes = target.version
            elif rollback_target_id is not None:
                raise ValueError("rollback_target_id is accepted only for commit or rollback")
            updated = SkillArtifact(
                current.skill_id,
                next_version,
                target_state,
                current.source_refs,
                str(uuid4()),
                current.revision + 1,
                execution_receipt,
                held_out_receipt,
                next_supersedes,
                target_id,
                checked_provenance,
            )
            updated_entry = (*entry, updated)
            payload = _serialize_entry(updated_entry)
            result = connection.execute(
                "UPDATE skill_lineages SET revision = ?, payload = ? "
                "WHERE authority_domain = ? AND skill_id = ? AND revision = ?",
                (
                    updated.revision,
                    payload,
                    state.authority_domain,
                    state.skill_id,
                    state.revision,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("skill lifecycle transition failed a stale revision check")
            _claim_artifact(connection, state.authority_domain, updated)
            connection.commit()
        except Exception:
            connection.rollback()
            connection.close()
            raise
        connection.close()
        _HISTORY_STATE[history] = _HistoryState(
            state.database_path,
            state.authority_domain,
            state.catalog_id,
            state.skill_id,
            updated.revision,
        )
        return _snapshot_artifact(updated)


def _validate_transition_provenance(
    connection: sqlite3.Connection,
    domain: str,
    current: SkillArtifact,
    target: SkillLifecycleState,
    provenance: object,
) -> TransitionProvenance | None:
    if target is SkillLifecycleState.PROCEDURAL_FAMILY:
        checked = _snapshot_consolidation(provenance)
        for identifier in checked.constituent_task_local_artifact_ids:
            artifact = _load_artifact(connection, domain, identifier)
            if artifact.state is not SkillLifecycleState.TASK_LOCAL:
                raise ValueError("consolidation constituents must be task-local artifacts")
        constituent_ids = set(checked.constituent_task_local_artifact_ids)
        if any(
            item.target_artifact_id not in constituent_ids for item in checked.pruning_decisions
        ):
            raise ValueError("consolidation pruning targets must be declared constituents")
        return checked
    if target is SkillLifecycleState.TASK_LOCAL:
        instantiation = _snapshot_instantiation(provenance)
        if instantiation.family_parent_artifact_id != current.artifact_id:
            raise ValueError("instantiation must bind the current parent artifact")
        return instantiation
    if target is SkillLifecycleState.REFINED:
        return _snapshot_refinement(provenance)
    if target is SkillLifecycleState.ROLLED_BACK:
        pruning = _snapshot_pruning(provenance)
        if pruning.target_artifact_id != current.artifact_id:
            raise ValueError("pruning provenance must retire the current artifact")
        return pruning
    if provenance is not None:
        raise ValueError(f"{target.value} does not accept transition provenance")
    return None


def _resolve_rollback_target(
    entry: tuple[SkillArtifact, ...],
    current: SkillArtifact,
    target_id: object,
    target_state: SkillLifecycleState,
) -> SkillArtifact:
    identifier = _require_token(target_id, "rollback_target_id")
    matches = tuple(item for item in entry[:-1] if item.artifact_id == identifier)
    if len(matches) != 1:
        raise ValueError("rollback target must be a strict predecessor artifact ID")
    target = matches[0]
    if target.artifact_id == current.artifact_id or target.state is SkillLifecycleState.ROLLED_BACK:
        raise ValueError("rollback target state is not permitted")
    if target_state is SkillLifecycleState.COMMITTED and target.state not in {
        SkillLifecycleState.CREDITED,
        SkillLifecycleState.HELD_OUT_VALIDATED,
        SkillLifecycleState.COMMITTED,
    }:
        raise ValueError("commit rollback target is not a stable predecessor")
    return target


def _validated_bundle(bundle: object) -> tuple[ActionRecord, OutcomeObservation]:
    if type(bundle) is not ExecutionEvidenceBundle:
        raise ValueError("execution evidence must be an exact ExecutionEvidenceBundle")
    checked = cast(ExecutionEvidenceBundle, bundle)
    action_event, outcome_event, world, local_event, relational_event = _snapshot_bundle_fields(
        checked
    )
    for event in (action_event, outcome_event, local_event, relational_event):
        _validate_exact_event_scalars(event)
    if action_event.event_type != StructuredEventType.ACTION_EXECUTED.value:
        raise ValueError("action_event must be an exact executed action event")
    if outcome_event.event_type != StructuredEventType.OUTCOME_OBSERVED.value:
        raise ValueError("outcome_event must be an exact observed outcome event")
    action = _parse_action(action_event)
    outcome = _parse_outcome(outcome_event)
    local = _parse_local_event(local_event)
    relational = _parse_relational_event(relational_event)
    if any(item.action_id != action.action_id for item in (outcome, world, local, relational)):
        raise ValueError("execution bundle must bind one exact action_id")
    if outcome.observation_id != world.after_observation.observation_id:
        raise ValueError("outcome observation_id must match the world-change observation")
    actual = _exact_mapping(
        outcome.actual_outcome,
        {"artifact_ref", "digest"},
        "observed artifact outcome",
    )
    _require_token(actual["artifact_ref"], "observed artifact_ref")
    _require_digest(actual["digest"], "observed artifact digest")
    if actual["artifact_ref"] != world.artifact_ref or actual["digest"] != world.after_digest:
        raise ValueError("outcome must bind the exact world-change artifact identity and digest")
    if action_event.action_id != action.action_id or outcome_event.action_id != action.action_id:
        raise ValueError("action/outcome envelope linkage does not match the action")
    if local_event.action_id != action.action_id or relational_event.action_id != action.action_id:
        raise ValueError("verification envelope linkage does not match the action")
    if action_event.event_id not in outcome_event.parent_event_ids:
        raise ValueError("outcome event must descend from the action event")
    if outcome_event.event_id not in local_event.parent_event_ids:
        raise ValueError("local verification must descend from the outcome event")
    if outcome_event.event_id not in relational_event.parent_event_ids:
        raise ValueError("relational verification must descend from the outcome event")
    event_ids = (
        action_event.event_id,
        outcome_event.event_id,
        local_event.event_id,
        relational_event.event_id,
    )
    if len(set(event_ids)) != 4:
        raise ValueError("execution bundle event IDs must be unique")
    _validate_shared_context(action_event, outcome_event, local_event, relational_event)
    action_time = _timestamp(action_event.timestamp, "action_event timestamp")
    outcome_time = _timestamp(outcome_event.timestamp, "outcome_event timestamp")
    observed_time = _timestamp(world.observed_at, "world observation timestamp")
    local_time = _timestamp(local_event.timestamp, "local verification timestamp")
    relational_time = _timestamp(
        relational_event.timestamp,
        "relational verification timestamp",
    )
    if outcome_time != observed_time:
        raise ValueError("outcome and world observation timestamps must be identical")
    if not action_time <= outcome_time <= local_time or not outcome_time <= relational_time:
        raise ValueError("execution evidence timestamps violate causal order")
    local_flags = (
        local.executed,
        local.arguments_valid,
        local.schema_valid,
        local.authorization_valid,
        local.intended_operation_observed,
    )
    if any(value is not True for value in local_flags):
        raise ValueError("local verification must affirm every required execution finding")
    if not action.reversible and local.irreversible_action_permitted is not True:
        raise ValueError("irreversible execution requires an affirmative permission finding")
    relational_flags = (
        relational.task_fit,
        relational.dependencies_satisfied,
        relational.provenance_intact,
        relational.authorization_scope_valid,
        relational.claimed_outcome_supported,
    )
    if any(value is not True for value in relational_flags):
        raise ValueError("relational verification must affirm every required outcome finding")
    if relational.contradiction_status != "none" or relational.repeated_failed_route is not False:
        raise ValueError(
            "relational verification must be uncontradicted and not a repeated failure"
        )
    outcome_refs = set(outcome.evidence_refs)
    world_refs = {item.reference_id for item in world.after_observation.evidence_refs}
    if not outcome_refs.intersection(world_refs):
        raise ValueError("outcome and world observation must share observable evidence")
    return action, outcome


def _issue_execution_receipt(bundle: ExecutionEvidenceBundle) -> ExecutionEvidenceReceipt:
    action, outcome = _validated_bundle(bundle)
    snapshot = ExecutionEvidenceBundle(*_snapshot_bundle_fields(bundle))
    events = (
        snapshot.action_event,
        snapshot.outcome_event,
        snapshot.local_verification_event,
        snapshot.relational_verification_event,
    )
    payload = {
        "events": [item.to_dict() for item in events],
        "world_change": snapshot.world_change.to_dict(),
    }
    evidence_ids = tuple(
        dict.fromkeys(
            (
                *outcome.evidence_refs,
                *(
                    item.reference_id
                    for item in snapshot.world_change.after_observation.evidence_refs
                ),
                *snapshot.local_verification_event.evidence_refs,
                *snapshot.relational_verification_event.evidence_refs,
            )
        )
    )
    verifier_ids = tuple(
        dict.fromkeys(
            (
                *snapshot.local_verification_event.verifier_refs,
                *snapshot.relational_verification_event.verifier_refs,
            )
        )
    )
    return ExecutionEvidenceReceipt(
        str(uuid4()),
        _sha256_json(payload),
        action.action_id,
        outcome.observation_id,
        snapshot.world_change.change_id,
        snapshot.world_change.artifact_ref,
        snapshot.world_change.after_digest,
        tuple(item.event_id for item in events),
        evidence_ids,
        verifier_ids,
    )


def _revalidate_execution_receipt(receipt: object) -> ExecutionEvidenceReceipt:
    if type(receipt) is not ExecutionEvidenceReceipt:
        raise ValueError("CREDITED requires a store-issued execution evidence receipt")
    return _snapshot_execution_receipt(receipt)


def _snapshot_bundle_fields(
    value: object,
) -> tuple[EventEnvelope, EventEnvelope, WorldStateChange, EventEnvelope, EventEnvelope]:
    if type(value) is not ExecutionEvidenceBundle:
        raise ValueError("execution evidence must be an exact ExecutionEvidenceBundle")
    checked = cast(ExecutionEvidenceBundle, value)
    return (
        _snapshot_event(checked.action_event),
        _snapshot_event(checked.outcome_event),
        _snapshot_world_change(checked.world_change),
        _snapshot_event(checked.local_verification_event),
        _snapshot_event(checked.relational_verification_event),
    )


def _parse_action(event: EventEnvelope) -> ActionRecord:
    values = _exact_mapping(
        event.payload,
        {"action_id", "action_class", "reversible", "authorization_scope", "prediction_commit_id"},
        "action event payload",
    )
    return ActionRecord(**cast(Any, dict(values)))


def _parse_outcome(event: EventEnvelope) -> OutcomeObservation:
    values = _exact_mapping(
        event.payload,
        {"observation_id", "action_id", "actual_outcome", "evidence_refs"},
        "outcome event payload",
    )
    return OutcomeObservation(
        cast(str, values["observation_id"]),
        cast(str, values["action_id"]),
        values["actual_outcome"],
        tuple(cast(list[str], values["evidence_refs"])),
    )


def _parse_local_event(event: EventEnvelope) -> LocalVerificationResult:
    values = _verification_payload(event, "local_execution")
    result_type = cast(Any, LocalVerificationResult)
    return cast(LocalVerificationResult, result_type.from_dict(values["result"]))


def _parse_relational_event(event: EventEnvelope) -> RelationalVerificationResult:
    values = _verification_payload(event, "relational_evidence")
    result_type = cast(Any, RelationalVerificationResult)
    return cast(RelationalVerificationResult, result_type.from_dict(values["result"]))


def _verification_payload(event: EventEnvelope, expected_level: str) -> Mapping[str, object]:
    if event.event_type != StructuredEventType.VERIFICATION_RESULT.value:
        raise ValueError("verification event must have VERIFICATION_RESULT type")
    values = _exact_mapping(
        event.payload,
        {"verification_level", "result", "verifier_refs"},
        "leveled verification payload",
    )
    if (
        type(values["verification_level"]) is not str
        or values["verification_level"] != expected_level
    ):
        raise ValueError(f"verification event must have {expected_level!r} level")
    payload_refs = tuple(
        cast(str, item)
        for item in _event_array(values["verifier_refs"], "verification verifier_refs")
    )
    if payload_refs != event.verifier_refs:
        raise ValueError("verification verifier_refs must match envelope metadata")
    result_value = cast(object, values["result"])
    result = _exact_mapping(
        result_value,
        set(cast(Mapping[Any, Any], result_value)),
        "verification result",
    )
    raw_evidence = result.get("evidence_refs")
    evidence_ids = tuple(
        cast(str, cast(Mapping[str, object], item)["reference_id"])
        for item in _event_array(raw_evidence, "verification evidence_refs")
    )
    if evidence_ids != event.evidence_refs:
        raise ValueError("verification evidence_refs must match envelope metadata")
    return values


def _validate_exact_event_scalars(event: EventEnvelope) -> None:
    for field_name in ("schema_version", "event_id", "event_type", "timestamp"):
        _require_token(getattr(event, field_name), field_name)
    for field_name in (
        "run_id",
        "rollout_id",
        "trace_id",
        "sample_id",
        "conversation_id",
        "checkpoint_id",
        "model_id",
        "model_checkpoint_hash",
        "dataset_id",
        "policy_version",
        "config_hash",
        "action_id",
        "model_version",
        "harness_version",
        "case_version",
        "stripe_id",
        "authorization_scope",
        "valid_from",
        "valid_until",
        "superseded_by",
    ):
        value = getattr(event, field_name)
        if value is not None:
            _require_token(value, field_name)
    for field_name in ("checkpoint_step", "case_id", "repeat_id", "seed"):
        value = getattr(event, field_name)
        if value is not None and type(value) is not int:
            raise ValueError(f"{field_name} must be an exact built-in integer")
    for field_name in ("parent_event_ids", "evidence_refs", "verifier_refs"):
        _snapshot_ids(getattr(event, field_name), field_name)
    _timestamp(event.timestamp, "timestamp")


def _validate_shared_context(*events: EventEnvelope) -> None:
    for field_name in (
        "run_id",
        "model_version",
        "harness_version",
        "case_version",
        "case_id",
        "stripe_id",
        "repeat_id",
        "seed",
    ):
        expected = getattr(events[0], field_name)
        if expected is None:
            raise ValueError(f"execution evidence requires {field_name}")
        if any(type(getattr(item, field_name)) is not type(expected) for item in events[1:]):
            raise ValueError(f"execution evidence {field_name} types must match exactly")
        if any(getattr(item, field_name) != expected for item in events[1:]):
            raise ValueError(f"execution evidence must share exact {field_name}")


def _timestamp(value: object, field_name: str) -> datetime:
    text = _require_token(value, field_name)
    match = _RFC3339_OFFSET_DATETIME.fullmatch(text)
    if match is None:
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime")
    hour = match.group("offset_hour")
    minute = match.group("offset_minute")
    if hour is not None and (int(hour) > 23 or int(minute) > 59):
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime")
    return parsed


def _validate_artifact_state(artifact: SkillArtifact) -> None:
    receipt_states = {
        SkillLifecycleState.EXECUTED,
        SkillLifecycleState.CREDITED,
        SkillLifecycleState.REFINED,
        SkillLifecycleState.HELD_OUT_VALIDATED,
        SkillLifecycleState.COMMITTED,
        SkillLifecycleState.ROLLED_BACK,
    }
    if artifact.state in receipt_states and artifact.state is not SkillLifecycleState.ROLLED_BACK:
        if artifact.execution_receipt is None:
            raise ValueError("execution receipt is required from EXECUTED onward")
    if artifact.state not in receipt_states and artifact.execution_receipt is not None:
        raise ValueError("execution receipt is forbidden before EXECUTED")
    validation_states = {
        SkillLifecycleState.HELD_OUT_VALIDATED,
        SkillLifecycleState.COMMITTED,
    }
    if artifact.state in validation_states and artifact.validation_receipt is None:
        raise ValueError("validation receipt is required after held-out validation")
    if artifact.state not in validation_states | {SkillLifecycleState.ROLLED_BACK}:
        if artifact.validation_receipt is not None:
            raise ValueError("validation receipt is forbidden before held-out validation")
    expected_provenance: dict[SkillLifecycleState, type[object]] = {
        SkillLifecycleState.PROCEDURAL_FAMILY: ConsolidationProvenance,
        SkillLifecycleState.TASK_LOCAL: InstantiationProvenance,
        SkillLifecycleState.REFINED: RefinementProvenance,
        SkillLifecycleState.ROLLED_BACK: PruningProvenance,
    }
    expected = expected_provenance.get(artifact.state)
    if expected is None and artifact.transition_provenance is not None:
        raise ValueError("transition provenance is not valid for this state")
    if expected is not None and type(artifact.transition_provenance) is not expected:
        raise ValueError(f"{artifact.state.value} requires {expected.__name__}")
    if artifact.state in {SkillLifecycleState.REFINED, *validation_states}:
        if artifact.supersedes is None:
            raise ValueError("refined and validated artifacts require supersedes")
    if artifact.state in {SkillLifecycleState.COMMITTED, SkillLifecycleState.ROLLED_BACK}:
        if artifact.rollback_target_id is None:
            raise ValueError("commit and rollback require a rollback target ID")


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS catalog_metadata (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            catalog_id TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS skill_lineages (
            authority_domain TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, skill_id)
        );
        CREATE TABLE IF NOT EXISTS skill_versions (
            authority_domain TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            version TEXT NOT NULL,
            PRIMARY KEY (authority_domain, skill_id, version)
        );
        CREATE TABLE IF NOT EXISTS skill_artifacts (
            authority_domain TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, artifact_id)
        );
        """)
    connection.execute(
        "INSERT OR IGNORE INTO catalog_metadata VALUES (1, ?)",
        (str(uuid4()),),
    )
    connection.commit()


def _open_database(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _catalog_id(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT catalog_id FROM catalog_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise ValueError("skill lifecycle catalog identity is missing")
    return _require_token(row[0], "catalog_id")


def _validated_store(store: object) -> tuple[str, str, str]:
    if type(store) is not SkillLifecycleStore:
        raise ValueError("store must be an exact SkillLifecycleStore")
    checked = cast(SkillLifecycleStore, store)
    binding = (checked._database_path, checked._authority_domain, checked._catalog_id)
    if checked._binding != binding:
        raise ValueError("SkillLifecycleStore binding changed after construction")
    with _open_database(checked._database_path) as connection:
        if _catalog_id(connection) != checked._catalog_id:
            raise ValueError("SkillLifecycleStore catalog identity changed")
    return binding


def _new_history(
    path: str,
    domain: str,
    catalog: str,
    skill_id: str,
    revision: int,
) -> SkillLifecycleHistory:
    handle = object.__new__(SkillLifecycleHistory)
    _HISTORY_STATE[handle] = _HistoryState(path, domain, catalog, skill_id, revision)
    return handle


def _begin_read(
    history: SkillLifecycleHistory,
) -> tuple[_HistoryState, sqlite3.Connection, tuple[SkillArtifact, ...]]:
    _require_process()
    if type(history) is not SkillLifecycleHistory or history not in _HISTORY_STATE:
        raise ValueError("operation requires a store-derived SkillLifecycleHistory")
    state = _HISTORY_STATE[history]
    connection = _open_database(state.database_path)
    try:
        if _catalog_id(connection) != state.catalog_id:
            raise ValueError("skill lifecycle history catalog identity changed")
        row = connection.execute(
            "SELECT revision, payload FROM skill_lineages "
            "WHERE authority_domain = ? AND skill_id = ?",
            (state.authority_domain, state.skill_id),
        ).fetchone()
        if row is None:
            raise ValueError("authoritative skill lifecycle no longer exists")
        revision = _require_nonnegative_int(row[0], "revision")
        if revision != state.revision:
            raise RuntimeError("skill lifecycle history handle has a stale revision")
        entry = _validated_entry(connection, state.authority_domain, state.skill_id, row[1])
        return state, connection, entry
    except Exception:
        connection.close()
        raise


def _begin_transition(
    history: SkillLifecycleHistory,
) -> tuple[_HistoryState, sqlite3.Connection, tuple[SkillArtifact, ...]]:
    if type(history) is not SkillLifecycleHistory:
        raise ValueError("operation requires a store-derived SkillLifecycleHistory")
    _require_process()
    if history not in _HISTORY_STATE:
        raise ValueError("operation requires a store-derived SkillLifecycleHistory")
    state = _HISTORY_STATE[history]
    connection = _open_database(state.database_path)
    connection.execute("BEGIN IMMEDIATE")
    try:
        if _catalog_id(connection) != state.catalog_id:
            raise ValueError("skill lifecycle history catalog identity changed")
        row = connection.execute(
            "SELECT revision, payload FROM skill_lineages "
            "WHERE authority_domain = ? AND skill_id = ?",
            (state.authority_domain, state.skill_id),
        ).fetchone()
        if row is None:
            raise ValueError("authoritative skill lifecycle no longer exists")
        if _require_nonnegative_int(row[0], "revision") != state.revision:
            raise RuntimeError("skill lifecycle history handle has a stale revision")
        return (
            state,
            connection,
            _validated_entry(connection, state.authority_domain, state.skill_id, row[1]),
        )
    except Exception:
        connection.rollback()
        connection.close()
        raise


def _serialize_entry(entry: tuple[SkillArtifact, ...]) -> str:
    return json.dumps(
        [item.to_dict() for item in entry],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validated_entry(
    connection: sqlite3.Connection,
    domain: str,
    skill_id: str,
    payload: object,
) -> tuple[SkillArtifact, ...]:
    if type(payload) is not str:
        raise ValueError("skill lifecycle payload must be exact JSON text")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("skill lifecycle payload is not valid JSON") from exc
    if type(raw) is not list or not raw:
        raise ValueError("skill lifecycle payload must be a nonempty array")
    artifacts = tuple(SkillArtifact.from_dict(item) for item in raw)
    first = artifacts[0]
    if first.skill_id != skill_id or first.state is not SkillLifecycleState.SOURCE_EXPERIENCE:
        raise ValueError("skill lifecycle root identity or state is invalid")
    if first.revision != 0:
        raise ValueError("skill lifecycle root revision must be zero")
    seen_versions = {first.version}
    seen_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if artifact.skill_id != skill_id or artifact.source_refs != first.source_refs:
            raise ValueError("skill lifecycle identity or source evidence changed")
        if artifact.revision != index:
            raise ValueError("skill lifecycle revisions must be contiguous")
        if artifact.artifact_id in seen_ids:
            raise ValueError("skill lifecycle artifact IDs must be unique")
        seen_ids.add(artifact.artifact_id)
        row = connection.execute(
            "SELECT skill_id, payload FROM skill_artifacts "
            "WHERE authority_domain = ? AND artifact_id = ?",
            (domain, artifact.artifact_id),
        ).fetchone()
        if row is None or row[0] != skill_id or json.loads(row[1]) != artifact.to_dict():
            raise ValueError("skill artifact differs from its catalog claim")
        if index == 0:
            continue
        previous = artifacts[index - 1]
        if artifact.state not in _ALLOWED_TRANSITIONS[previous.state]:
            raise ValueError("skill lifecycle contains an illegal transition")
        if artifact.state is SkillLifecycleState.REFINED:
            if artifact.version in seen_versions or artifact.supersedes != previous.version:
                raise ValueError("skill refinement version lineage is invalid")
            seen_versions.add(artifact.version)
        elif artifact.version != previous.version:
            raise ValueError("skill version changed outside refinement")
        if artifact.state is SkillLifecycleState.EXECUTED:
            if artifact.execution_receipt is None:
                raise ValueError("executed transition lost its evidence receipt")
        elif artifact.execution_receipt != previous.execution_receipt:
            raise ValueError("execution receipt changed outside execution")
        if artifact.state is SkillLifecycleState.HELD_OUT_VALIDATED:
            if artifact.validation_receipt is None:
                raise ValueError("held-out transition lost its validation receipt")
        elif artifact.validation_receipt != previous.validation_receipt:
            raise ValueError("validation receipt changed outside held-out validation")
        if artifact.rollback_target_id is not None:
            if artifact.rollback_target_id not in {item.artifact_id for item in artifacts[:index]}:
                raise ValueError("rollback target is not a strict predecessor")
    claimed_versions = {
        row[0]
        for row in connection.execute(
            "SELECT version FROM skill_versions WHERE authority_domain = ? AND skill_id = ?",
            (domain, skill_id),
        ).fetchall()
    }
    if claimed_versions != seen_versions:
        raise ValueError("skill version claims do not match lifecycle history")
    return artifacts


def _claim_artifact(
    connection: sqlite3.Connection,
    domain: str,
    artifact: SkillArtifact,
) -> None:
    connection.execute(
        "INSERT INTO skill_artifacts VALUES (?, ?, ?, ?)",
        (
            domain,
            artifact.artifact_id,
            artifact.skill_id,
            json.dumps(artifact.to_dict(), separators=(",", ":"), sort_keys=True),
        ),
    )


def _load_artifact(
    connection: sqlite3.Connection,
    domain: str,
    artifact_id: str,
) -> SkillArtifact:
    row = connection.execute(
        "SELECT payload FROM skill_artifacts WHERE authority_domain = ? AND artifact_id = ?",
        (domain, artifact_id),
    ).fetchone()
    if row is None:
        raise ValueError("provenance artifact ID is absent from this authority catalog")
    return SkillArtifact.from_dict(json.loads(row[0]))


def _snapshot_artifact(value: object) -> SkillArtifact:
    if type(value) is not SkillArtifact:
        raise ValueError("artifact must be an exact SkillArtifact")
    checked = cast(SkillArtifact, value)
    if checked._construction_binding != _artifact_binding(checked):
        raise ValueError("SkillArtifact construction binding changed after construction")
    return SkillArtifact(
        checked.skill_id,
        checked.version,
        checked.state,
        checked.source_refs,
        checked.artifact_id,
        checked.revision,
        checked.execution_receipt,
        checked.validation_receipt,
        checked.supersedes,
        checked.rollback_target_id,
        checked.transition_provenance,
    )


def _snapshot_execution_receipt(value: object) -> ExecutionEvidenceReceipt:
    if type(value) is not ExecutionEvidenceReceipt:
        raise ValueError("execution receipt must be an exact ExecutionEvidenceReceipt")
    checked = cast(ExecutionEvidenceReceipt, value)
    if checked._construction_binding != _execution_receipt_binding(checked):
        raise ValueError("ExecutionEvidenceReceipt construction binding changed after construction")
    return ExecutionEvidenceReceipt(
        checked.receipt_id,
        checked.bundle_digest,
        checked.action_id,
        checked.observation_id,
        checked.change_id,
        checked.artifact_ref,
        checked.artifact_digest,
        checked.event_ids,
        checked.evidence_ref_ids,
        checked.verifier_ref_ids,
    )


def _execution_receipt_binding(value: ExecutionEvidenceReceipt) -> tuple[object, ...]:
    return (
        value.receipt_id,
        value.bundle_digest,
        value.action_id,
        value.observation_id,
        value.change_id,
        value.artifact_ref,
        value.artifact_digest,
        value.event_ids,
        value.evidence_ref_ids,
        value.verifier_ref_ids,
    )


def _artifact_binding(value: SkillArtifact) -> tuple[object, ...]:
    source = tuple((item.reference_id, item.source_kind.value) for item in value.source_refs)
    execution = (
        None
        if value.execution_receipt is None
        else _execution_receipt_binding(value.execution_receipt)
    )
    validation = None if value.validation_receipt is None else value.validation_receipt.to_dict()
    provenance = _provenance_to_dict(value.transition_provenance)
    return (
        value.skill_id,
        value.version,
        value.state.value if type(value.state) is SkillLifecycleState else value.state,
        source,
        value.artifact_id,
        value.revision,
        execution,
        validation,
        value.supersedes,
        value.rollback_target_id,
        provenance,
    )


def _snapshot_event(value: object) -> EventEnvelope:
    if type(value) is not EventEnvelope:
        raise ValueError("execution evidence requires exact EventEnvelope values")
    try:
        return EventEnvelope(**cast(Any, cast(EventEnvelope, value).to_dict()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"execution evidence contains an invalid event: {exc}") from exc


def _snapshot_world_change(value: object) -> WorldStateChange:
    if type(value) is not WorldStateChange:
        raise ValueError("world_change must be an exact WorldStateChange")
    value_type = cast(Any, WorldStateChange)
    return cast(WorldStateChange, value_type.from_dict(cast(Any, value).to_dict()))


def _snapshot_refs(value: object) -> tuple[EvidenceReference, ...]:
    if type(value) is not tuple:
        raise ValueError("evidence_refs must be an exact tuple")
    references: list[EvidenceReference] = []
    reference_type = cast(Any, EvidenceReference)
    for item in cast(tuple[object, ...], value):
        if type(item) is not EvidenceReference:
            raise ValueError("evidence_refs must contain exact EvidenceReference values")
        checked = cast(EvidenceReference, item)
        _require_token(checked.reference_id, "evidence reference_id")
        if type(checked.source_kind) is not EvidenceSourceKind:
            raise ValueError("evidence source_kind must be an exact EvidenceSourceKind")
        references.append(
            cast(EvidenceReference, reference_type(checked.reference_id, checked.source_kind))
        )
    if not references or not any(item.is_observable for item in references):
        raise ValueError("evidence_refs require at least one observable reference")
    identities = tuple((item.reference_id, item.source_kind) for item in references)
    if len(set(identities)) != len(identities):
        raise ValueError("evidence_refs must be unique")
    return tuple(references)


def _snapshot_optional_provenance(value: object) -> TransitionProvenance | None:
    if value is None:
        return None
    if type(value) is ConsolidationProvenance:
        return _snapshot_consolidation(value)
    if type(value) is InstantiationProvenance:
        return _snapshot_instantiation(value)
    if type(value) is RefinementProvenance:
        return _snapshot_refinement(value)
    if type(value) is PruningProvenance:
        return _snapshot_pruning(value)
    raise ValueError("transition_provenance has an unsupported exact type")


def _snapshot_pruning(value: object) -> PruningProvenance:
    if type(value) is not PruningProvenance:
        raise ValueError("transition requires exact PruningProvenance")
    checked = cast(PruningProvenance, value)
    return PruningProvenance(
        checked.target_artifact_id,
        checked.reason,
        checked.evidence_refs,
    )


def _snapshot_consolidation(value: object) -> ConsolidationProvenance:
    if type(value) is not ConsolidationProvenance:
        raise ValueError("transition requires exact ConsolidationProvenance")
    checked = cast(ConsolidationProvenance, value)
    return ConsolidationProvenance(
        checked.constituent_task_local_artifact_ids,
        checked.pruning_decisions,
    )


def _snapshot_instantiation(value: object) -> InstantiationProvenance:
    if type(value) is not InstantiationProvenance:
        raise ValueError("transition requires exact InstantiationProvenance")
    checked = cast(InstantiationProvenance, value)
    return InstantiationProvenance(
        checked.family_parent_artifact_id,
        checked.task_context,
    )


def _snapshot_refinement(value: object) -> RefinementProvenance:
    if type(value) is not RefinementProvenance:
        raise ValueError("transition requires exact RefinementProvenance")
    checked = cast(RefinementProvenance, value)
    return RefinementProvenance(checked.change_rationale, checked.evidence_refs)


def _provenance_to_dict(value: TransitionProvenance | None) -> dict[str, object] | None:
    return None if value is None else value.to_dict()


def _provenance_from_dict(value: object) -> TransitionProvenance | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("transition_provenance must be an exact object or null")
    values = cast(dict[str, object], value)
    kind = values.get("kind")
    if kind == "pruning":
        checked = _exact_mapping(
            value,
            {"kind", "target_artifact_id", "reason", "evidence_refs"},
            "PruningProvenance",
        )
        return PruningProvenance(
            cast(str, checked["target_artifact_id"]),
            cast(str, checked["reason"]),
            _refs_from_json(checked["evidence_refs"]),
        )
    if kind == "consolidation":
        checked = _exact_mapping(
            value,
            {"kind", "constituent_task_local_artifact_ids", "pruning_decisions"},
            "ConsolidationProvenance",
        )
        decisions = tuple(
            cast(
                PruningProvenance,
                _provenance_from_dict(item),
            )
            for item in _exact_list(checked["pruning_decisions"], "pruning_decisions")
        )
        return ConsolidationProvenance(
            tuple(
                cast(str, item)
                for item in _exact_list(
                    checked["constituent_task_local_artifact_ids"],
                    "constituent_task_local_artifact_ids",
                )
            ),
            decisions,
        )
    if kind == "instantiation":
        checked = _exact_mapping(
            value,
            {"kind", "family_parent_artifact_id", "task_context"},
            "InstantiationProvenance",
        )
        return InstantiationProvenance(
            cast(str, checked["family_parent_artifact_id"]),
            cast(str, checked["task_context"]),
        )
    if kind == "refinement":
        checked = _exact_mapping(
            value,
            {"kind", "change_rationale", "evidence_refs"},
            "RefinementProvenance",
        )
        return RefinementProvenance(
            cast(str, checked["change_rationale"]),
            _refs_from_json(checked["evidence_refs"]),
        )
    raise ValueError("transition_provenance kind is unknown")


def _refs_from_json(value: object) -> tuple[EvidenceReference, ...]:
    return tuple(EvidenceReference.from_dict(item) for item in _exact_list(value, "evidence_refs"))


def _parse_state(value: object) -> SkillLifecycleState:
    if type(value) is not str:
        raise ValueError("state must be an exact string enum value")
    try:
        return SkillLifecycleState(value)
    except ValueError as exc:
        raise ValueError(f"unknown lifecycle state {value!r}") from exc


def _snapshot_ids(value: object, field_name: str, *, required: bool = False) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be an exact tuple")
    identifiers = tuple(
        _require_token(item, field_name) for item in cast(tuple[object, ...], value)
    )
    if required and not identifiers:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field_name} must contain unique ordered IDs")
    return identifiers


def _require_token(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a nonblank exact string without outer whitespace")
    return value


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative exact integer")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    text = _require_token(value, field_name)
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise ValueError(f"{field_name} must be sha256:<64 lowercase hex>")
    return text


def _require_digest(value: object, field_name: str) -> str:
    text = _require_token(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a canonical lowercase SHA-256 digest")
    return text


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _exact_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, object]:
    if type(value) not in {dict, MappingProxyType}:
        raise ValueError(f"{name} must be an exact object")
    checked = cast(Mapping[str, object], value)
    if set(checked) != fields:
        raise ValueError(f"{name} requires exactly {sorted(fields)!r}")
    return checked


def _exact_list(value: object, field_name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{field_name} must be an exact array")
    return cast(list[object], value)


def _event_array(value: object, field_name: str) -> tuple[object, ...] | list[object]:
    if type(value) is not list and not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be an exact frozen or JSON array")
    return cast(tuple[object, ...] | list[object], value)


def _require_process() -> None:
    if os.getpid() != _PROCESS_ID:
        raise RuntimeError("skill lifecycle authority cannot cross a process boundary")


__all__ = [
    "ConsolidationProvenance",
    "ExecutionEvidenceBundle",
    "ExecutionEvidenceReceipt",
    "InstantiationProvenance",
    "PruningProvenance",
    "RefinementProvenance",
    "SkillArtifact",
    "SkillLifecycleHistory",
    "SkillLifecycleState",
    "SkillLifecycleStore",
    "transition_skill",
]
