"""Typed destinations and immutable proposals for controlled learning."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TypeVar, cast
from weakref import WeakKeyDictionary

from evaluation.v5_records import V5EvaluationRecord
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

EnumT = TypeVar("EnumT", bound=Enum)


class LearningSurface(str, Enum):
    """The single primary destination selected for a lesson."""

    TRACE_ONLY = "trace_only"
    MEMORY = "memory"
    HARNESS = "harness"
    SKILL_GRAPH = "skill_graph"
    MODEL = "model"
    HUMAN = "human"


class LessonKind(str, Enum):
    """Evidence characteristics that determine an automatic learning destination."""

    ONE_OFF_OBSERVATION = "one_off_observation"
    EPISODE_FACT = "episode_fact"
    STABLE_PROCEDURAL_CONVENTION = "stable_procedural_convention"
    REUSABLE_DEPENDENCY = "reusable_dependency"
    PERSISTENT_INTRINSIC_BEHAVIOR = "persistent_intrinsic_behavior"


class LessonReviewStatus(str, Enum):
    """The explicit human review decision attached to a proposal.

    Pending proposals have no review decision, approved proposals may be considered for their
    declared destination, and rejected proposals must not be applied.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class EvaluationEpoch:
    """One immutable model-and-harness boundary for declared V5 records."""

    epoch_id: str
    model_version: str
    harness_version: str
    record_ids: tuple[str, ...]
    record_cell_ids: tuple[str, ...] = ()
    closed: bool = False
    _construction_binding: tuple[object, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_epoch_fields(self)
        object.__setattr__(self, "_construction_binding", _epoch_binding(self))

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible epoch snapshot."""

        snapshot = _snapshot_epoch(self)
        return {
            "epoch_id": snapshot.epoch_id,
            "model_version": snapshot.model_version,
            "harness_version": snapshot.harness_version,
            "record_ids": list(snapshot.record_ids),
            "record_cell_ids": list(snapshot.record_cell_ids),
            "closed": snapshot.closed,
        }

    @classmethod
    def from_dict(cls, data: object) -> EvaluationEpoch:
        """Restore an epoch from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "epoch_id",
                "model_version",
                "harness_version",
                "record_ids",
                "record_cell_ids",
                "closed",
            },
            "EvaluationEpoch",
        )
        raw_record_ids = values["record_ids"]
        if type(raw_record_ids) is not list:
            raise ValueError("EvaluationEpoch record_ids must be an array")
        record_ids = cast(list[object], raw_record_ids)
        raw_cell_ids = values["record_cell_ids"]
        if type(raw_cell_ids) is not list:
            raise ValueError("EvaluationEpoch record_cell_ids must be an array")
        cell_ids = cast(list[object], raw_cell_ids)
        return cls(
            epoch_id=cast(str, values["epoch_id"]),
            model_version=cast(str, values["model_version"]),
            harness_version=cast(str, values["harness_version"]),
            record_ids=tuple(cast(str, item) for item in record_ids),
            record_cell_ids=tuple(cast(str, item) for item in cell_ids),
            closed=_require_exact_bool(values["closed"], "closed"),
        )


@dataclass(frozen=True, slots=True)
class _EpochHistoryEntry:
    lineage: tuple[EvaluationEpoch, ...]
    parent_epoch_ids: tuple[str | None, ...]
    records: tuple[tuple[str, V5EvaluationRecord], ...]


class EvaluationEpochHistory:
    """Runtime-owned local authority for one complete evaluation lineage.

    The public object is only an opaque handle. Module-private process-local storage owns exact
    snapshots, parent links, and accepted records. Callers must persist this authority in a
    stronger authenticated store when process-local integrity is insufficient.
    """

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("use EvaluationEpochHistory.enroll()")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("EvaluationEpochHistory is read-only")

    @classmethod
    def enroll(cls, initial_epoch: EvaluationEpoch) -> EvaluationEpochHistory:
        if cls is not EvaluationEpochHistory:
            raise ValueError("enrollment requires the exact EvaluationEpochHistory type")
        _require_epoch_process()
        epoch = _snapshot_epoch(initial_epoch)
        if epoch.closed:
            raise ValueError("initial epoch must be open; use the controlled close transition")
        if epoch.record_ids:
            raise ValueError("initial epoch must be empty; append records through the history")
        with _EPOCH_HISTORY_LOCK:
            handle = object.__new__(EvaluationEpochHistory)
            _EPOCH_HISTORY_STATE[handle] = _validated_history_entry(
                _EpochHistoryEntry((epoch,), (None,), ())
            )
        return handle

    def snapshot(self) -> tuple[EvaluationEpoch, ...]:
        """Return a detached, non-authoritative view of the complete lineage."""

        _require_exact_history(self)
        with _EPOCH_HISTORY_LOCK:
            entry = _validated_history_entry(_EPOCH_HISTORY_STATE[self])
            return tuple(_snapshot_epoch(epoch) for epoch in entry.lineage)


_EPOCH_HISTORY_PROCESS_ID = os.getpid()
_EPOCH_HISTORY_LOCK = RLock()
_EPOCH_HISTORY_STATE: WeakKeyDictionary[EvaluationEpochHistory, _EpochHistoryEntry] = (
    WeakKeyDictionary()
)


def _reset_epoch_history_after_fork() -> None:
    global _EPOCH_HISTORY_PROCESS_ID, _EPOCH_HISTORY_LOCK, _EPOCH_HISTORY_STATE
    _EPOCH_HISTORY_PROCESS_ID = os.getpid()
    _EPOCH_HISTORY_LOCK = RLock()
    _EPOCH_HISTORY_STATE = WeakKeyDictionary()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_epoch_history_after_fork)


def evaluation_record_id(record: V5EvaluationRecord) -> str:
    """Return a content-bound identifier for one revalidated canonical V5 record."""

    snapshot = _snapshot_evaluation_record(record)
    return _evaluation_record_id_from_snapshot(snapshot)


def evaluation_record_cell_id(record: V5EvaluationRecord) -> str:
    """Return the canonical logical V5 cell independent of observations and scores."""

    snapshot = _snapshot_evaluation_record(record)
    payload = {
        "case_id": snapshot.case.case_id,
        "stripe_id": snapshot.robustness.stripe_id,
        "subtype": snapshot.robustness.subtype,
        "repeat_id": snapshot.system.repeat_id,
        "seed": snapshot.system.seed,
        "model_version": snapshot.system.model_version,
        "harness_version": snapshot.system.harness_version,
    }
    return _sha256_json(payload)


def validate_epoch_record(epoch: EvaluationEpoch, record: V5EvaluationRecord) -> None:
    """Require one exact V5 record to match its declared immutable epoch membership."""

    epoch_snapshot = _snapshot_epoch(epoch)
    record_snapshot = _snapshot_evaluation_record(record)
    if record_snapshot.system.model_version != epoch_snapshot.model_version:
        raise ValueError("record model_version does not match evaluation epoch")
    if record_snapshot.system.harness_version != epoch_snapshot.harness_version:
        raise ValueError("record harness_version does not match evaluation epoch")
    record_id = _evaluation_record_id_from_snapshot(record_snapshot)
    cell_id = evaluation_record_cell_id(record_snapshot)
    try:
        record_index = epoch_snapshot.record_ids.index(record_id)
    except ValueError:
        raise ValueError("record identity is not declared in evaluation epoch record_ids")
    if epoch_snapshot.record_cell_ids[record_index] != cell_id:
        raise ValueError("record logical evaluation cell does not match its epoch manifest")


def append_epoch_record(
    epoch: EvaluationEpoch | EvaluationEpochHistory,
    record: V5EvaluationRecord,
) -> EvaluationEpoch:
    """Append online evidence without changing the open epoch's system versions."""

    if type(epoch) is EvaluationEpochHistory:
        return _append_authoritative_epoch_record(epoch, record)
    epoch_snapshot = _snapshot_epoch(epoch)
    if epoch_snapshot.closed:
        raise ValueError("cannot append a record to a closed evaluation epoch")
    record_snapshot = _snapshot_evaluation_record(record)
    if record_snapshot.system.model_version != epoch_snapshot.model_version:
        raise ValueError("record model_version does not match evaluation epoch")
    if record_snapshot.system.harness_version != epoch_snapshot.harness_version:
        raise ValueError("record harness_version does not match evaluation epoch")
    record_id = _evaluation_record_id_from_snapshot(record_snapshot)
    cell_id = evaluation_record_cell_id(record_snapshot)
    if record_id in epoch_snapshot.record_ids:
        raise ValueError("cannot append a duplicate evaluation record")
    if cell_id in epoch_snapshot.record_cell_ids:
        raise ValueError("cannot append a duplicate logical evaluation cell")
    return EvaluationEpoch(
        epoch_id=epoch_snapshot.epoch_id,
        model_version=epoch_snapshot.model_version,
        harness_version=epoch_snapshot.harness_version,
        record_ids=epoch_snapshot.record_ids + (record_id,),
        record_cell_ids=epoch_snapshot.record_cell_ids + (cell_id,),
        closed=False,
    )


def _append_authoritative_epoch_record(
    history: EvaluationEpochHistory,
    record: V5EvaluationRecord,
) -> EvaluationEpoch:
    _require_exact_history(history)
    record_snapshot = _snapshot_evaluation_record(record)
    with _EPOCH_HISTORY_LOCK:
        entry = _validated_history_entry(_EPOCH_HISTORY_STATE[history])
        updated_tip = append_epoch_record(entry.lineage[-1], record_snapshot)
        record_id = _evaluation_record_id_from_snapshot(record_snapshot)
        updated = _EpochHistoryEntry(
            (*entry.lineage[:-1], updated_tip),
            entry.parent_epoch_ids,
            (*entry.records, (record_id, record_snapshot)),
        )
        _EPOCH_HISTORY_STATE[history] = _validated_history_entry(updated)
        return _snapshot_epoch(updated_tip)


def close_evaluation_epoch(history: EvaluationEpochHistory) -> EvaluationEpoch:
    """Close the authoritative tip exactly once without accepting caller-supplied lineage."""

    _require_exact_history(history)
    with _EPOCH_HISTORY_LOCK:
        entry = _validated_history_entry(_EPOCH_HISTORY_STATE[history])
        source = entry.lineage[-1]
        if source.closed:
            raise ValueError("source evaluation epoch is already closed")
        closed = EvaluationEpoch(
            source.epoch_id,
            source.model_version,
            source.harness_version,
            source.record_ids,
            source.record_cell_ids,
            True,
        )
        updated = _EpochHistoryEntry(
            (*entry.lineage[:-1], closed), entry.parent_epoch_ids, entry.records
        )
        _EPOCH_HISTORY_STATE[history] = _validated_history_entry(updated)
        return _snapshot_epoch(closed)


def begin_candidate_epoch(
    epoch_history: EvaluationEpochHistory,
    *,
    epoch_id: str,
    model_version: str,
    harness_version: str,
) -> EvaluationEpoch:
    """Begin a candidate epoch after a closed lineage with non-reused changed versions."""

    _require_exact_history(epoch_history)
    with _EPOCH_HISTORY_LOCK:
        entry = _validated_history_entry(_EPOCH_HISTORY_STATE[epoch_history])
        history = entry.lineage
        source = history[-1]
        if not source.closed:
            raise ValueError("source epoch must be closed before a candidate version is evaluated")
        candidate = EvaluationEpoch(
            epoch_id=epoch_id,
            model_version=model_version,
            harness_version=harness_version,
            record_ids=(),
            record_cell_ids=(),
            closed=False,
        )
        _validate_candidate_transition(history, source, candidate)
        updated = _EpochHistoryEntry(
            (*history, candidate),
            (*entry.parent_epoch_ids, source.epoch_id),
            entry.records,
        )
        _EPOCH_HISTORY_STATE[epoch_history] = _validated_history_entry(updated)
        return _snapshot_epoch(candidate)


@dataclass(frozen=True, slots=True)
class LessonCharacteristics:
    """Typed facts used by the routing decision table instead of lesson prose."""

    kind: LessonKind
    normative: bool = False
    ambiguous: bool = False
    difficult_to_reverse: bool = False

    def __post_init__(self) -> None:
        _validate_characteristics_fields(self)

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible characteristics record."""

        snapshot = _snapshot_characteristics(self)
        return {
            "kind": snapshot.kind.value,
            "normative": snapshot.normative,
            "ambiguous": snapshot.ambiguous,
            "difficult_to_reverse": snapshot.difficult_to_reverse,
        }

    @classmethod
    def from_dict(cls, data: object) -> LessonCharacteristics:
        """Restore characteristics from their exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {"kind", "normative", "ambiguous", "difficult_to_reverse"},
            "LessonCharacteristics",
        )
        return cls(
            kind=_parse_exact_enum(values["kind"], LessonKind, "kind"),
            normative=_require_exact_bool(values["normative"], "normative"),
            ambiguous=_require_exact_bool(values["ambiguous"], "ambiguous"),
            difficult_to_reverse=_require_exact_bool(
                values["difficult_to_reverse"],
                "difficult_to_reverse",
            ),
        )


def classify_learning_surface(characteristics: LessonCharacteristics) -> LearningSurface:
    """Select one destination from explicit typed characteristics."""

    snapshot = _snapshot_characteristics(characteristics)
    if snapshot.normative or snapshot.ambiguous or snapshot.difficult_to_reverse:
        return LearningSurface.HUMAN
    if snapshot.kind is LessonKind.ONE_OFF_OBSERVATION:
        return LearningSurface.TRACE_ONLY
    if snapshot.kind is LessonKind.EPISODE_FACT:
        return LearningSurface.MEMORY
    if snapshot.kind is LessonKind.STABLE_PROCEDURAL_CONVENTION:
        return LearningSurface.HARNESS
    if snapshot.kind is LessonKind.REUSABLE_DEPENDENCY:
        return LearningSurface.SKILL_GRAPH
    if snapshot.kind is LessonKind.PERSISTENT_INTRINSIC_BEHAVIOR:
        return LearningSurface.MODEL
    raise ValueError("kind has no learning-surface decision")


@dataclass(frozen=True, slots=True)
class LessonProposal:
    """An immutable, evidence-bound proposal with one primary destination."""

    lesson_id: str
    summary: str
    characteristics: LessonCharacteristics
    primary_destination: LearningSurface
    evidence_refs: tuple[EvidenceReference, ...]
    rationale: str
    reversible: bool
    review_status: LessonReviewStatus
    _routing_binding: tuple[LessonKind, bool, bool, bool, LearningSurface] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_nonblank_string(self.lesson_id, "lesson_id")
        _require_nonblank_string(self.summary, "summary")
        characteristics = _snapshot_characteristics(self.characteristics)
        if type(self.primary_destination) is not LearningSurface:
            raise ValueError("primary_destination must be exactly one LearningSurface")
        expected_destination = classify_learning_surface(characteristics)
        if self.primary_destination is not expected_destination:
            raise ValueError("primary_destination must match classified characteristics")
        references = _snapshot_evidence_refs(self.evidence_refs)
        if not any(reference.is_observable for reference in references):
            raise ValueError("LessonProposal requires observable evidence_refs")
        reference_ids = {reference.reference_id for reference in references}
        if len(reference_ids) != len(references):
            raise ValueError("evidence_refs must have unique reference_id values")
        _require_nonblank_string(self.rationale, "rationale")
        _require_exact_bool(self.reversible, "reversible")
        if type(self.review_status) is not LessonReviewStatus:
            raise ValueError("review_status must be an exact LessonReviewStatus")
        object.__setattr__(self, "characteristics", characteristics)
        object.__setattr__(self, "evidence_refs", references)
        object.__setattr__(
            self,
            "_routing_binding",
            _make_routing_binding(characteristics, expected_destination),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a revalidated JSON-compatible proposal snapshot."""

        snapshot = _snapshot_proposal(self)
        return {
            "lesson_id": snapshot.lesson_id,
            "summary": snapshot.summary,
            "characteristics": snapshot.characteristics.to_dict(),
            "primary_destination": snapshot.primary_destination.value,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
            "rationale": snapshot.rationale,
            "reversible": snapshot.reversible,
            "review_status": snapshot.review_status.value,
        }

    @classmethod
    def from_dict(cls, data: object) -> LessonProposal:
        """Restore a proposal from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "lesson_id",
                "summary",
                "characteristics",
                "primary_destination",
                "evidence_refs",
                "rationale",
                "reversible",
                "review_status",
            },
            "LessonProposal",
        )
        raw_references = values["evidence_refs"]
        if type(raw_references) is not list:
            raise ValueError("LessonProposal evidence_refs must be an array")
        reference_items = cast(list[object], raw_references)
        return cls(
            lesson_id=cast(str, values["lesson_id"]),
            summary=cast(str, values["summary"]),
            characteristics=LessonCharacteristics.from_dict(values["characteristics"]),
            primary_destination=_parse_exact_enum(
                values["primary_destination"],
                LearningSurface,
                "primary_destination",
            ),
            evidence_refs=tuple(EvidenceReference.from_dict(item) for item in reference_items),
            rationale=cast(str, values["rationale"]),
            reversible=_require_exact_bool(values["reversible"], "reversible"),
            review_status=_parse_exact_enum(
                values["review_status"],
                LessonReviewStatus,
                "review_status",
            ),
        )


def _validate_characteristics_fields(value: LessonCharacteristics) -> None:
    if type(value.kind) is not LessonKind:
        raise ValueError("kind must be an exact LessonKind")
    _require_exact_bool(value.normative, "normative")
    _require_exact_bool(value.ambiguous, "ambiguous")
    _require_exact_bool(value.difficult_to_reverse, "difficult_to_reverse")


def _validate_epoch_fields(value: EvaluationEpoch) -> None:
    _require_epoch_token(value.epoch_id, "epoch_id")
    _require_epoch_token(value.model_version, "model_version")
    _require_epoch_token(value.harness_version, "harness_version")
    if type(value.record_ids) is not tuple:
        raise ValueError("record_ids must be an exact tuple")
    record_ids = tuple(_require_sha256(record_id, "record_ids") for record_id in value.record_ids)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record_ids must not contain duplicate record IDs")
    if type(value.record_cell_ids) is not tuple:
        raise ValueError("record_cell_ids must be an exact tuple")
    cell_ids = tuple(
        _require_sha256(cell_id, "record_cell_ids") for cell_id in value.record_cell_ids
    )
    if len(cell_ids) != len(record_ids):
        raise ValueError("record_cell_ids must correspond exactly to record_ids")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("record_cell_ids must not contain duplicate logical evaluation cells")
    _require_exact_bool(value.closed, "closed")


def _epoch_binding(value: EvaluationEpoch) -> tuple[object, ...]:
    return (
        value.epoch_id,
        value.model_version,
        value.harness_version,
        value.record_ids,
        value.record_cell_ids,
        value.closed,
    )


def _validate_epoch_binding(value: EvaluationEpoch) -> None:
    binding = value._construction_binding
    if type(binding) is not tuple or len(binding) != 6 or binding != _epoch_binding(value):
        raise ValueError("EvaluationEpoch construction binding changed after construction")


def _snapshot_epoch(value: object) -> EvaluationEpoch:
    if type(value) is not EvaluationEpoch:
        raise ValueError("epoch must be an exact EvaluationEpoch")
    _validate_epoch_fields(value)
    _validate_epoch_binding(value)
    return EvaluationEpoch(
        epoch_id=value.epoch_id,
        model_version=value.model_version,
        harness_version=value.harness_version,
        record_ids=value.record_ids,
        record_cell_ids=value.record_cell_ids,
        closed=value.closed,
    )


def _require_exact_history(value: object) -> EvaluationEpochHistory:
    _require_epoch_process()
    if type(value) is not EvaluationEpochHistory:
        raise ValueError("epoch_history must be an exact EvaluationEpochHistory")
    if value not in _EPOCH_HISTORY_STATE:
        raise ValueError("EvaluationEpochHistory is not enrolled in this process")
    return value


def _require_epoch_process() -> None:
    if os.getpid() != _EPOCH_HISTORY_PROCESS_ID:
        raise RuntimeError("evaluation epoch authority cannot cross a process boundary")


def _validated_history_entry(value: object) -> _EpochHistoryEntry:
    if type(value) is not _EpochHistoryEntry:
        raise ValueError("evaluation epoch lineage store entry is invalid")
    if type(value.lineage) is not tuple or not value.lineage:
        raise ValueError("evaluation epoch lineage must be a nonempty exact tuple")
    lineage = tuple(_snapshot_epoch(epoch) for epoch in value.lineage)
    if type(value.parent_epoch_ids) is not tuple or len(value.parent_epoch_ids) != len(lineage):
        raise ValueError("evaluation epoch lineage parent continuity is invalid")
    expected_parents: tuple[str | None, ...] = (None,) + tuple(
        epoch.epoch_id for epoch in lineage[:-1]
    )
    if value.parent_epoch_ids != expected_parents:
        raise ValueError("evaluation epoch lineage parent/tip continuity is invalid")
    epoch_ids = tuple(epoch.epoch_id for epoch in lineage)
    if len(set(epoch_ids)) != len(epoch_ids):
        raise ValueError("evaluation epoch lineage contains reused epoch_id")
    if any(not epoch.closed for epoch in lineage[:-1]):
        raise ValueError("evaluation epoch lineage has a non-closed historical epoch")
    for index in range(1, len(lineage)):
        _validate_candidate_transition(lineage[:index], lineage[index - 1], lineage[index])
    if type(value.records) is not tuple:
        raise ValueError("evaluation epoch authoritative records must be an exact tuple")
    records: list[tuple[str, V5EvaluationRecord]] = []
    record_map: dict[str, V5EvaluationRecord] = {}
    for item in value.records:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("evaluation epoch authoritative record entry is invalid")
        record_id = _require_sha256(item[0], "authoritative record_id")
        record = _snapshot_evaluation_record(item[1])
        if _evaluation_record_id_from_snapshot(record) != record_id:
            raise ValueError("authoritative evaluation record binding is invalid")
        if record_id in record_map:
            raise ValueError("authoritative evaluation record is duplicated")
        record_map[record_id] = record
        records.append((record_id, record))
    declared_ids = {record_id for epoch in lineage for record_id in epoch.record_ids}
    if set(record_map) != declared_ids:
        raise ValueError("authoritative records do not match complete epoch lineage manifests")
    for epoch in lineage:
        for record_id in epoch.record_ids:
            validate_epoch_record(epoch, record_map[record_id])
    return _EpochHistoryEntry(lineage, expected_parents, tuple(records))


def _validate_candidate_transition(
    prior_lineage: tuple[EvaluationEpoch, ...],
    source: EvaluationEpoch,
    candidate: EvaluationEpoch,
) -> None:
    if not source.closed:
        raise ValueError("every source epoch in the lineage must be closed")
    if candidate.epoch_id in {epoch.epoch_id for epoch in prior_lineage}:
        raise ValueError("candidate epoch_id must be new within the authoritative epoch lineage")
    model_changed = candidate.model_version != source.model_version
    harness_changed = candidate.harness_version != source.harness_version
    if not model_changed and not harness_changed:
        raise ValueError("candidate transition must change model_version or harness_version")
    if model_changed and candidate.model_version in {
        epoch.model_version for epoch in prior_lineage
    }:
        raise ValueError("candidate model_version must not reuse a prior model_version")
    if harness_changed and candidate.harness_version in {
        epoch.harness_version for epoch in prior_lineage
    }:
        raise ValueError("candidate harness_version must not reuse a prior harness_version")


def _snapshot_evaluation_record(value: object) -> V5EvaluationRecord:
    if type(value) is not V5EvaluationRecord:
        raise ValueError("record must be an exact V5EvaluationRecord")
    try:
        return V5EvaluationRecord.from_dict(value.to_dict())
    except (AttributeError, TypeError) as exc:
        raise ValueError("record must remain a valid V5EvaluationRecord") from exc


def _evaluation_record_id_from_snapshot(record: V5EvaluationRecord) -> str:
    return _sha256_json(record.to_dict())


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_sha256(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be exactly sha256:<64 lowercase hex>")
    return value


def _require_epoch_token(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a nonblank string without surrounding whitespace")
    return value


def _snapshot_characteristics(value: object) -> LessonCharacteristics:
    if type(value) is not LessonCharacteristics:
        raise ValueError("characteristics must be an exact LessonCharacteristics")
    _validate_characteristics_fields(value)
    return LessonCharacteristics(
        kind=value.kind,
        normative=value.normative,
        ambiguous=value.ambiguous,
        difficult_to_reverse=value.difficult_to_reverse,
    )


def _snapshot_proposal(value: object) -> LessonProposal:
    if type(value) is not LessonProposal:
        raise ValueError("proposal must be an exact LessonProposal")
    if type(value.evidence_refs) is not tuple:
        raise ValueError("LessonProposal evidence_refs must remain an exact tuple")
    characteristics = _snapshot_characteristics(value.characteristics)
    _validate_routing_binding(value, characteristics)
    return LessonProposal(
        lesson_id=value.lesson_id,
        summary=value.summary,
        characteristics=characteristics,
        primary_destination=value.primary_destination,
        evidence_refs=value.evidence_refs,
        rationale=value.rationale,
        reversible=value.reversible,
        review_status=value.review_status,
    )


def _make_routing_binding(
    characteristics: LessonCharacteristics,
    destination: LearningSurface,
) -> tuple[LessonKind, bool, bool, bool, LearningSurface]:
    return (
        characteristics.kind,
        characteristics.normative,
        characteristics.ambiguous,
        characteristics.difficult_to_reverse,
        destination,
    )


def _validate_routing_binding(
    proposal: LessonProposal,
    characteristics: LessonCharacteristics,
) -> None:
    binding = proposal._routing_binding
    if type(binding) is not tuple or len(binding) != 5:
        raise ValueError("LessonProposal routing binding is invalid")
    if (
        type(binding[0]) is not LessonKind
        or type(binding[1]) is not bool
        or type(binding[2]) is not bool
        or type(binding[3]) is not bool
        or type(binding[4]) is not LearningSurface
    ):
        raise ValueError("LessonProposal routing binding is invalid")
    expected = _make_routing_binding(characteristics, proposal.primary_destination)
    if binding != expected:
        raise ValueError("LessonProposal routing binding changed after construction")


def _snapshot_evidence_refs(values: object) -> tuple[EvidenceReference, ...]:
    if type(values) not in {tuple, list}:
        raise ValueError("evidence_refs must be an exact tuple or list")
    items = cast(tuple[object, ...] | list[object], values)
    references: list[EvidenceReference] = []
    for reference in items:
        if type(reference) is not EvidenceReference:
            raise ValueError("evidence_refs must contain exact EvidenceReference values")
        checked_reference = cast(EvidenceReference, reference)
        _require_nonblank_string(checked_reference.reference_id, "evidence_refs reference_id")
        if type(checked_reference.source_kind) is not EvidenceSourceKind:
            raise ValueError("evidence_refs source_kind must be an exact EvidenceSourceKind")
        references.append(
            EvidenceReference(checked_reference.reference_id, checked_reference.source_kind)
        )
    return tuple(references)


def _require_nonblank_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank built-in string")
    return value


def _require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be an exact bool")
    return value


def _parse_exact_enum(
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a built-in string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} has unknown value {value!r}") from exc


def _require_exact_mapping(
    data: object,
    expected_fields: set[str],
    record_name: str,
) -> Mapping[str, object]:
    if type(data) is not dict:
        raise ValueError(f"{record_name} must be an exact object")
    if set(data) != expected_fields:
        raise ValueError(f"{record_name} requires exactly {sorted(expected_fields)!r}")
    return cast(Mapping[str, object], data)


__all__ = [
    "EvaluationEpoch",
    "EvaluationEpochHistory",
    "LearningSurface",
    "LessonCharacteristics",
    "LessonKind",
    "LessonProposal",
    "LessonReviewStatus",
    "append_epoch_record",
    "begin_candidate_epoch",
    "classify_learning_surface",
    "close_evaluation_epoch",
    "evaluation_record_cell_id",
    "evaluation_record_id",
    "validate_epoch_record",
]
