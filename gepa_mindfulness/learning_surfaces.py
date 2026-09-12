"""Typed destinations and immutable proposals for controlled learning."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, TypeVar, cast
from uuid import uuid4
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


class ValidationSplit(str, Enum):
    """A non-training evaluation split eligible for lifecycle validation."""

    HELD_OUT = "held_out"
    PROTECTED = "protected"


@dataclass(frozen=True, slots=True)
class EvaluationAuthority:
    """Canonical locator and identity for one durable evaluation authority catalog."""

    catalog_path: str
    catalog_id: str
    authority_domain: str

    def __post_init__(self) -> None:
        _require_epoch_token(self.catalog_path, "catalog_path")
        canonical = _canonical_catalog_path(self.catalog_path)
        if canonical != self.catalog_path:
            raise ValueError("catalog_path must be canonical")
        _require_epoch_token(self.catalog_id, "catalog_id")
        _require_epoch_token(self.authority_domain, "authority_domain")


@dataclass(frozen=True, slots=True)
class CandidateTargetClaim:
    """Evaluation-catalog claim for one candidate before its first record is appended."""

    catalog_id: str
    authority_domain: str
    lineage_id: str
    epoch_id: str
    epoch_revision: int
    candidate_id: str
    artifact_digest: str
    model_version: str
    harness_version: str
    claim_digest: str

    def __post_init__(self) -> None:
        for name in (
            "catalog_id",
            "authority_domain",
            "lineage_id",
            "epoch_id",
            "candidate_id",
            "model_version",
            "harness_version",
        ):
            _require_epoch_token(getattr(self, name), name)
        _require_nonnegative_integer(self.epoch_revision, "epoch_revision")
        _require_sha256(self.artifact_digest, "artifact_digest")
        _require_sha256(self.claim_digest, "claim_digest")
        if self.claim_digest != _sha256_json(_candidate_claim_payload(self, False)):
            raise ValueError("candidate target claim digest does not match its exact provenance")

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible claim."""

        return _candidate_claim_payload(self, True)

    @classmethod
    def from_dict(cls, data: object) -> CandidateTargetClaim:
        values = _require_exact_mapping(
            data,
            {
                "catalog_id",
                "authority_domain",
                "lineage_id",
                "epoch_id",
                "epoch_revision",
                "candidate_id",
                "artifact_digest",
                "model_version",
                "harness_version",
                "claim_digest",
            },
            "CandidateTargetClaim",
        )
        return cls(
            cast(str, values["catalog_id"]),
            cast(str, values["authority_domain"]),
            cast(str, values["lineage_id"]),
            cast(str, values["epoch_id"]),
            cast(int, values["epoch_revision"]),
            cast(str, values["candidate_id"]),
            cast(str, values["artifact_digest"]),
            cast(str, values["model_version"]),
            cast(str, values["harness_version"]),
            cast(str, values["claim_digest"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationTarget:
    """Exact candidate or artifact target evaluated by a validation receipt."""

    artifact_id: str
    skill_id: str
    version: str
    artifact_digest: str

    def __post_init__(self) -> None:
        _require_epoch_token(self.artifact_id, "artifact_id")
        _require_epoch_token(self.skill_id, "skill_id")
        _require_epoch_token(self.version, "version")
        _require_sha256(self.artifact_digest, "artifact_digest")


@dataclass(frozen=True, slots=True)
class ValidationReceipt:
    """A durable attestation over passing canonical V5 records for one evaluated target."""

    receipt_id: str
    catalog_id: str
    catalog_path: str
    authority_domain: str
    lineage_id: str
    epoch_id: str
    epoch_revision: int
    split: ValidationSplit
    record_ids: tuple[str, ...]
    target_artifact_id: str
    target_skill_id: str
    target_version: str
    target_digest: str
    _construction_binding: tuple[object, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "receipt_id",
            "catalog_id",
            "lineage_id",
            "epoch_id",
            "target_artifact_id",
            "target_skill_id",
            "target_version",
        ):
            _require_epoch_token(getattr(self, field_name), field_name)
        _require_epoch_token(self.catalog_path, "catalog_path")
        if _canonical_catalog_path(self.catalog_path) != self.catalog_path:
            raise ValueError("catalog_path must be canonical")
        _require_epoch_token(self.authority_domain, "authority_domain")
        _require_sha256(self.target_digest, "target_digest")
        _require_nonnegative_integer(self.epoch_revision, "epoch_revision")
        if type(self.split) is not ValidationSplit:
            raise ValueError("split must be an exact non-training ValidationSplit")
        if type(self.record_ids) is not tuple or not self.record_ids:
            raise ValueError("record_ids must be a nonempty exact tuple")
        record_ids = tuple(_require_sha256(item, "record_ids") for item in self.record_ids)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("record_ids must contain unique ordered identities")
        object.__setattr__(self, "record_ids", record_ids)
        object.__setattr__(self, "_construction_binding", _validation_receipt_binding(self))

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible receipt."""

        _validate_receipt_fields(self)
        return {
            "receipt_id": self.receipt_id,
            "catalog_id": self.catalog_id,
            "catalog_path": self.catalog_path,
            "authority_domain": self.authority_domain,
            "lineage_id": self.lineage_id,
            "epoch_id": self.epoch_id,
            "epoch_revision": self.epoch_revision,
            "split": self.split.value,
            "record_ids": list(self.record_ids),
            "target_artifact_id": self.target_artifact_id,
            "target_skill_id": self.target_skill_id,
            "target_version": self.target_version,
            "target_digest": self.target_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> ValidationReceipt:
        """Restore an untrusted receipt snapshot; the issuing store must revalidate it."""

        values = _require_exact_mapping(
            data,
            {
                "receipt_id",
                "catalog_id",
                "catalog_path",
                "authority_domain",
                "lineage_id",
                "epoch_id",
                "epoch_revision",
                "split",
                "record_ids",
                "target_artifact_id",
                "target_skill_id",
                "target_version",
                "target_digest",
            },
            "ValidationReceipt",
        )
        raw_ids = values["record_ids"]
        if type(raw_ids) is not list:
            raise ValueError("ValidationReceipt record_ids must be an array")
        return cls(
            cast(str, values["receipt_id"]),
            cast(str, values["catalog_id"]),
            cast(str, values["catalog_path"]),
            cast(str, values["authority_domain"]),
            cast(str, values["lineage_id"]),
            cast(str, values["epoch_id"]),
            cast(int, values["epoch_revision"]),
            _parse_exact_enum(values["split"], ValidationSplit, "split"),
            tuple(cast(str, item) for item in cast(list[object], raw_ids)),
            cast(str, values["target_artifact_id"]),
            cast(str, values["target_skill_id"]),
            cast(str, values["target_version"]),
            cast(str, values["target_digest"]),
        )


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


@dataclass(frozen=True, slots=True)
class _EpochHistoryHandleState:
    database_path: str
    authority_domain: str
    lineage_id: str
    revision: int


class EvaluationEpochStore:
    """Injected durable authority catalog for evaluation lineages.

    SQLite transactions serialize root creation and transitions across processes. The authority
    domain scopes epoch and changed-version nonreuse. Catalog authority assumes the runtime owner
    controls the canonical database path and protects it from replacement or direct tampering.
    """

    __slots__ = ("_authority_domain", "_binding", "_database_path")
    _authority_domain: str
    _binding: tuple[str, str]
    _database_path: str

    def __init__(self, database_path: str | os.PathLike[str], authority_domain: str) -> None:
        path = _canonical_catalog_path(database_path)
        domain = _require_epoch_token(authority_domain, "authority_domain")
        if not os.path.isdir(os.path.dirname(path)):
            raise ValueError("evaluation epoch store parent directory must already exist")
        object.__setattr__(self, "_database_path", path)
        object.__setattr__(self, "_authority_domain", domain)
        object.__setattr__(self, "_binding", (path, domain))
        with _open_epoch_database(path) as connection:
            _initialize_epoch_database(connection)

    def authority(self) -> EvaluationAuthority:
        """Return the exact durable catalog identity that callers may pin as trusted."""

        path, domain = _validated_epoch_store(self)
        with _open_epoch_database(path) as connection:
            return EvaluationAuthority(path, _catalog_id(connection), domain)

    def create_root(
        self,
        *,
        lineage_id: str,
        epoch_id: str,
        model_version: str,
        harness_version: str,
    ) -> EvaluationEpochHistory:
        """Exclusively create one empty open root and claim its versions in this domain."""

        path, domain = _validated_epoch_store(self)
        lineage = _require_epoch_token(lineage_id, "lineage_id")
        root = EvaluationEpoch(epoch_id, model_version, harness_version, (), (), False)
        entry = _validated_history_entry(_EpochHistoryEntry((root,), (None,), ()))
        with _open_epoch_database(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM epoch_lineages WHERE authority_domain = ? AND lineage_id = ?",
                (domain, lineage),
            ).fetchone():
                raise ValueError("evaluation lineage already exists in this authority domain")
            _claim_catalog_identity(connection, "epoch_claims", domain, root.epoch_id, lineage)
            _claim_catalog_identity(
                connection, "model_version_claims", domain, root.model_version, lineage
            )
            _claim_catalog_identity(
                connection, "harness_version_claims", domain, root.harness_version, lineage
            )
            connection.execute(
                "INSERT INTO epoch_lineages VALUES (?, ?, ?, ?, ?)",
                (domain, lineage, 0, root.epoch_id, _serialize_history_entry(entry)),
            )
            connection.commit()
        return _new_history_handle(path, domain, lineage, 0)

    def open(self, lineage_id: str) -> EvaluationEpochHistory:
        """Open and fully validate a persisted lineage at its current revision."""

        path, domain = _validated_epoch_store(self)
        lineage = _require_epoch_token(lineage_id, "lineage_id")
        with _open_epoch_database(path) as connection:
            row = connection.execute(
                "SELECT revision, tip_epoch_id, payload FROM epoch_lineages "
                "WHERE authority_domain = ? AND lineage_id = ?",
                (domain, lineage),
            ).fetchone()
            if row is None:
                raise KeyError(lineage)
            revision = _require_nonnegative_integer(row[0], "revision")
            entry = _deserialize_history_entry(row[2])
            _validate_catalog(connection, domain, lineage, row[1], entry)
        return _new_history_handle(path, domain, lineage, revision)

    def resolve_epoch(
        self,
        lineage_id: str,
        epoch_id: str,
    ) -> tuple[int, EvaluationEpoch, tuple[V5EvaluationRecord, ...]]:
        """Return one detached canonical epoch and its records from this authority."""

        path, domain = _validated_epoch_store(self)
        lineage = _require_epoch_token(lineage_id, "lineage_id")
        identifier = _require_epoch_token(epoch_id, "epoch_id")
        with _open_epoch_database(path) as connection:
            row = connection.execute(
                "SELECT revision, tip_epoch_id, payload FROM epoch_lineages "
                "WHERE authority_domain = ? AND lineage_id = ?",
                (domain, lineage),
            ).fetchone()
            if row is None:
                raise KeyError(lineage)
            revision = _require_nonnegative_integer(row[0], "revision")
            entry = _deserialize_history_entry(row[2])
            _validate_catalog(connection, domain, lineage, row[1], entry)
        matches = tuple(epoch for epoch in entry.lineage if epoch.epoch_id == identifier)
        if len(matches) != 1:
            raise KeyError(identifier)
        epoch = matches[0]
        records = dict(entry.records)
        resolved = tuple(_snapshot_evaluation_record(records[item]) for item in epoch.record_ids)
        return revision, _snapshot_epoch(epoch), resolved

    def claim_candidate_target(
        self,
        *,
        lineage_id: str,
        epoch_id: str,
        candidate_id: str,
        artifact_digest: str,
    ) -> CandidateTargetClaim:
        """Atomically claim one empty open epoch for one exact candidate target."""

        path, domain = _validated_epoch_store(self)
        lineage = _require_epoch_token(lineage_id, "lineage_id")
        epoch_identifier = _require_epoch_token(epoch_id, "epoch_id")
        candidate_identifier = _require_epoch_token(candidate_id, "candidate_id")
        digest = _require_sha256(artifact_digest, "artifact_digest")
        with _open_epoch_database(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT revision, tip_epoch_id, payload FROM epoch_lineages "
                    "WHERE authority_domain = ? AND lineage_id = ?",
                    (domain, lineage),
                ).fetchone()
                if row is None:
                    raise KeyError(lineage)
                revision = _require_nonnegative_integer(row[0], "revision")
                entry = _deserialize_history_entry(row[2])
                _validate_catalog(connection, domain, lineage, row[1], entry)
                existing_rows = connection.execute(
                    "SELECT payload FROM candidate_target_claims "
                    "WHERE authority_domain = ? AND "
                    "(candidate_id = ? OR epoch_id = ? OR artifact_digest = ?)",
                    (domain, candidate_identifier, epoch_identifier, digest),
                ).fetchall()
                if existing_rows:
                    claims = tuple(
                        CandidateTargetClaim.from_dict(json.loads(item[0]))
                        for item in existing_rows
                    )
                    if len(claims) == 1:
                        claim = claims[0]
                        expected = (
                            lineage,
                            epoch_identifier,
                            candidate_identifier,
                            digest,
                        )
                        actual = (
                            claim.lineage_id,
                            claim.epoch_id,
                            claim.candidate_id,
                            claim.artifact_digest,
                        )
                        if actual == expected:
                            connection.commit()
                            return claim
                    raise ValueError("candidate target claim alias or rebind is already registered")
                epoch = entry.lineage[-1]
                if (
                    epoch.epoch_id != epoch_identifier
                    or epoch.closed
                    or epoch.record_ids
                    or epoch.record_cell_ids
                ):
                    raise ValueError("candidate claim requires the exact empty open tip epoch")
                content = {
                    "catalog_id": _catalog_id(connection),
                    "authority_domain": domain,
                    "lineage_id": lineage,
                    "epoch_id": epoch.epoch_id,
                    "epoch_revision": revision,
                    "candidate_id": candidate_identifier,
                    "artifact_digest": digest,
                    "model_version": epoch.model_version,
                    "harness_version": epoch.harness_version,
                }
                claim = CandidateTargetClaim(
                    catalog_id=cast(str, content["catalog_id"]),
                    authority_domain=domain,
                    lineage_id=lineage,
                    epoch_id=epoch.epoch_id,
                    epoch_revision=revision,
                    candidate_id=candidate_identifier,
                    artifact_digest=digest,
                    model_version=epoch.model_version,
                    harness_version=epoch.harness_version,
                    claim_digest=_sha256_json(content),
                )
                connection.execute(
                    "INSERT INTO candidate_target_claims VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        domain,
                        candidate_identifier,
                        epoch_identifier,
                        digest,
                        lineage,
                        json.dumps(claim.to_dict(), separators=(",", ":"), sort_keys=True),
                    ),
                )
                connection.commit()
                return CandidateTargetClaim.from_dict(claim.to_dict())
            except Exception:
                connection.rollback()
                raise

    def resolve_candidate_target_claim(self, candidate_id: str) -> CandidateTargetClaim:
        """Reload an exact target claim and validate it against the authoritative lineage."""

        path, domain = _validated_epoch_store(self)
        identifier = _require_epoch_token(candidate_id, "candidate_id")
        with _open_epoch_database(path) as connection:
            row = connection.execute(
                "SELECT payload FROM candidate_target_claims "
                "WHERE authority_domain = ? AND candidate_id = ?",
                (domain, identifier),
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            claim = CandidateTargetClaim.from_dict(json.loads(row[0]))
            if claim.catalog_id != _catalog_id(connection) or claim.authority_domain != domain:
                raise ValueError("candidate target claim belongs to a different authority")
            lineage_row = connection.execute(
                "SELECT revision, tip_epoch_id, payload FROM epoch_lineages "
                "WHERE authority_domain = ? AND lineage_id = ?",
                (domain, claim.lineage_id),
            ).fetchone()
            if lineage_row is None:
                raise ValueError("candidate target claim lineage is absent")
            revision = _require_nonnegative_integer(lineage_row[0], "revision")
            entry = _deserialize_history_entry(lineage_row[2])
            _validate_catalog(connection, domain, claim.lineage_id, lineage_row[1], entry)
        matches = tuple(epoch for epoch in entry.lineage if epoch.epoch_id == claim.epoch_id)
        if len(matches) != 1 or (
            matches[0].model_version,
            matches[0].harness_version,
        ) != (claim.model_version, claim.harness_version):
            raise ValueError("candidate target claim epoch or versions are no longer canonical")
        if revision < claim.epoch_revision:
            raise ValueError("candidate target claim revision is invalid")
        return CandidateTargetClaim.from_dict(claim.to_dict())

    def issue_validation_receipt(
        self,
        history: EvaluationEpochHistory,
        *,
        epoch_id: str,
        target: ValidationTarget,
        split: ValidationSplit,
        records: tuple[V5EvaluationRecord, ...],
    ) -> ValidationReceipt:
        """Persist an attestation over passing records in one closed authoritative epoch."""

        path, domain = _validated_epoch_store(self)
        if type(split) is not ValidationSplit:
            raise ValueError("split must be an exact non-training ValidationSplit")
        checked_target = _snapshot_validation_target(target)
        identifier = _require_epoch_token(epoch_id, "epoch_id")
        if type(records) is not tuple or not records:
            raise ValueError("records must be a nonempty exact tuple")
        checked_history = _require_exact_history(history)
        state = _EPOCH_HISTORY_STATE[checked_history]
        if (state.database_path, state.authority_domain) != (path, domain):
            raise ValueError("history and validation store must share catalog and authority domain")
        with _EPOCH_HISTORY_LOCK:
            handle_state, connection, entry = _begin_history_read(checked_history)
            try:
                epochs = tuple(epoch for epoch in entry.lineage if epoch.epoch_id == identifier)
                if len(epochs) != 1:
                    raise ValueError("epoch_id is absent from the authoritative lineage")
                epoch = epochs[0]
                if not epoch.closed:
                    raise ValueError("validation receipt requires a closed evaluation epoch")
                if epoch.harness_version != checked_target.version:
                    raise ValueError("evaluation epoch harness must match the target version")
                record_ids: list[str] = []
                for record in records:
                    snapshot = _snapshot_evaluation_record(record)
                    validate_epoch_record(epoch, snapshot)
                    if snapshot.system.harness_version != checked_target.version:
                        raise ValueError("evaluation records must match the target version")
                    if snapshot.outcome.passed is not True:
                        raise ValueError("validation receipt requires every outcome to pass")
                    record_ids.append(_evaluation_record_id_from_snapshot(snapshot))
                if len(set(record_ids)) != len(record_ids):
                    raise ValueError("validation receipt records must be unique")
                receipt = ValidationReceipt(
                    str(uuid4()),
                    _catalog_id(connection),
                    path,
                    domain,
                    handle_state.lineage_id,
                    epoch.epoch_id,
                    handle_state.revision,
                    split,
                    tuple(record_ids),
                    checked_target.artifact_id,
                    checked_target.skill_id,
                    checked_target.version,
                    checked_target.artifact_digest,
                )
                payload = json.dumps(receipt.to_dict(), separators=(",", ":"), sort_keys=True)
                connection.execute(
                    "INSERT INTO validation_receipts VALUES (?, ?, ?, ?)",
                    (domain, receipt.receipt_id, handle_state.lineage_id, payload),
                )
                connection.commit()
                return ValidationReceipt.from_dict(receipt.to_dict())
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def validate_validation_receipt(self, receipt: ValidationReceipt) -> ValidationReceipt:
        """Resolve an exact receipt from this store's durable catalog."""

        path, domain = _validated_epoch_store(self)
        snapshot = _snapshot_validation_receipt(receipt)
        with _open_epoch_database(path) as connection:
            if snapshot.catalog_id != _catalog_id(connection):
                raise ValueError("validation receipt belongs to a different catalog")
            if snapshot.catalog_path != path:
                raise ValueError("validation receipt belongs to a different catalog path")
            if snapshot.authority_domain != domain:
                raise ValueError("validation receipt belongs to a different authority domain")
            row = connection.execute(
                "SELECT payload FROM validation_receipts "
                "WHERE authority_domain = ? AND receipt_id = ? AND lineage_id = ?",
                (domain, snapshot.receipt_id, snapshot.lineage_id),
            ).fetchone()
            if row is None:
                raise ValueError("validation receipt was not issued by this store")
            canonical = ValidationReceipt.from_dict(json.loads(row[0]))
            if canonical != snapshot:
                raise ValueError("validation receipt differs from its durable catalog record")
            lineage = connection.execute(
                "SELECT revision, payload FROM epoch_lineages "
                "WHERE authority_domain = ? AND lineage_id = ?",
                (domain, snapshot.lineage_id),
            ).fetchone()
            if lineage is None or snapshot.epoch_revision > lineage[0]:
                raise ValueError("validation receipt lineage or revision is invalid")
            entry = _deserialize_history_entry(lineage[1])
            epoch = next(
                (item for item in entry.lineage if item.epoch_id == snapshot.epoch_id),
                None,
            )
            if epoch is None or not epoch.closed:
                raise ValueError("validation receipt epoch is not closed and authoritative")
            if not set(snapshot.record_ids).issubset(epoch.record_ids):
                raise ValueError("validation receipt records are absent from its epoch")
            if epoch.harness_version != snapshot.target_version:
                raise ValueError("validation receipt target version differs from its epoch")
            records = dict(entry.records)
            for record_id in snapshot.record_ids:
                record = records.get(record_id)
                if record is None or record.system.harness_version != snapshot.target_version:
                    raise ValueError("validation receipt target differs from a canonical record")
                if record.outcome.passed is not True:
                    raise ValueError("validation receipt contains a non-passing canonical record")
        return ValidationReceipt.from_dict(snapshot.to_dict())


class EvaluationEpochHistory:
    """Opaque revision-bound handle to one store-owned complete evaluation lineage.

    Canonical epochs, parent links, accepted records, and revision authority live in the injected
    SQLite catalog. The process-local state contains only the store locator and expected revision.
    """

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("lineage handles are created by EvaluationEpochStore")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("EvaluationEpochHistory is read-only")

    def snapshot(self) -> tuple[EvaluationEpoch, ...]:
        """Return a detached, non-authoritative view of the complete lineage."""

        _require_exact_history(self)
        with _EPOCH_HISTORY_LOCK:
            _state, connection, entry = _begin_history_read(self)
            connection.close()
            return tuple(_snapshot_epoch(epoch) for epoch in entry.lineage)


_EPOCH_HISTORY_PROCESS_ID = os.getpid()
_EPOCH_HISTORY_LOCK = RLock()
_EPOCH_HISTORY_STATE: WeakKeyDictionary[EvaluationEpochHistory, _EpochHistoryHandleState] = (
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
    epoch: EvaluationEpochHistory,
    record: V5EvaluationRecord,
) -> EvaluationEpoch:
    """Append online evidence without changing the open epoch's system versions."""

    _require_exact_history(epoch)
    record_snapshot = _snapshot_evaluation_record(record)
    with _EPOCH_HISTORY_LOCK:
        state, connection, entry = _begin_history_transition(epoch)
        source = entry.lineage[-1]
        if source.closed:
            connection.close()
            raise ValueError("cannot append a record to a closed evaluation epoch")
        if record_snapshot.system.model_version != source.model_version:
            connection.close()
            raise ValueError("record model_version does not match evaluation epoch")
        if record_snapshot.system.harness_version != source.harness_version:
            connection.close()
            raise ValueError("record harness_version does not match evaluation epoch")
        record_id = _evaluation_record_id_from_snapshot(record_snapshot)
        cell_id = evaluation_record_cell_id(record_snapshot)
        if record_id in {item[0] for item in entry.records}:
            connection.close()
            raise ValueError("cannot append a duplicate evaluation record")
        if cell_id in source.record_cell_ids:
            connection.close()
            raise ValueError("cannot append a duplicate logical evaluation cell")
        updated_tip = EvaluationEpoch(
            source.epoch_id,
            source.model_version,
            source.harness_version,
            (*source.record_ids, record_id),
            (*source.record_cell_ids, cell_id),
            False,
        )
        updated = _EpochHistoryEntry(
            (*entry.lineage[:-1], updated_tip),
            entry.parent_epoch_ids,
            (*entry.records, (record_id, record_snapshot)),
        )
        _commit_history_transition(epoch, state, connection, updated)
        return _snapshot_epoch(updated_tip)


def close_evaluation_epoch(history: EvaluationEpochHistory) -> EvaluationEpoch:
    """Close the authoritative tip exactly once without accepting caller-supplied lineage."""

    _require_exact_history(history)
    with _EPOCH_HISTORY_LOCK:
        state, connection, entry = _begin_history_transition(history)
        source = entry.lineage[-1]
        if source.closed:
            connection.close()
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
        _commit_history_transition(history, state, connection, updated)
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
        state, connection, entry = _begin_history_transition(epoch_history)
        history = entry.lineage
        source = history[-1]
        if not source.closed:
            connection.close()
            raise ValueError("source epoch must be closed before a candidate version is evaluated")
        try:
            candidate = EvaluationEpoch(
                epoch_id=epoch_id,
                model_version=model_version,
                harness_version=harness_version,
                record_ids=(),
                record_cell_ids=(),
                closed=False,
            )
            _validate_candidate_transition(history, source, candidate)
            _claim_catalog_identity(
                connection,
                "epoch_claims",
                state.authority_domain,
                candidate.epoch_id,
                state.lineage_id,
            )
            if candidate.model_version != source.model_version:
                _claim_catalog_identity(
                    connection,
                    "model_version_claims",
                    state.authority_domain,
                    candidate.model_version,
                    state.lineage_id,
                )
            if candidate.harness_version != source.harness_version:
                _claim_catalog_identity(
                    connection,
                    "harness_version_claims",
                    state.authority_domain,
                    candidate.harness_version,
                    state.lineage_id,
                )
        except Exception:
            connection.rollback()
            connection.close()
            raise
        updated = _EpochHistoryEntry(
            (*history, candidate),
            (*entry.parent_epoch_ids, source.epoch_id),
            entry.records,
        )
        _commit_history_transition(epoch_history, state, connection, updated)
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


def _open_epoch_database(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _initialize_epoch_database(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS catalog_metadata (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            catalog_id TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS epoch_lineages (
            authority_domain TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            tip_epoch_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, lineage_id)
        );
        CREATE TABLE IF NOT EXISTS epoch_claims (
            authority_domain TEXT NOT NULL,
            identity TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            PRIMARY KEY (authority_domain, identity)
        );
        CREATE TABLE IF NOT EXISTS model_version_claims (
            authority_domain TEXT NOT NULL,
            identity TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            PRIMARY KEY (authority_domain, identity)
        );
        CREATE TABLE IF NOT EXISTS harness_version_claims (
            authority_domain TEXT NOT NULL,
            identity TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            PRIMARY KEY (authority_domain, identity)
        );
        CREATE TABLE IF NOT EXISTS candidate_target_claims (
            authority_domain TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            epoch_id TEXT NOT NULL,
            artifact_digest TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, candidate_id),
            UNIQUE (authority_domain, epoch_id),
            UNIQUE (authority_domain, artifact_digest)
        );
        CREATE TABLE IF NOT EXISTS validation_receipts (
            authority_domain TEXT NOT NULL,
            receipt_id TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (authority_domain, receipt_id)
        );
        """)
    connection.execute(
        "INSERT OR IGNORE INTO catalog_metadata VALUES (1, ?)",
        (str(uuid4()),),
    )
    connection.commit()


def _catalog_id(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT catalog_id FROM catalog_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise ValueError("evaluation catalog identity is missing")
    return _require_epoch_token(row[0], "catalog_id")


def _validate_receipt_fields(value: ValidationReceipt) -> None:
    if type(value) is not ValidationReceipt:
        raise ValueError("receipt must be an exact ValidationReceipt")
    if value._construction_binding != _validation_receipt_binding(value):
        raise ValueError("ValidationReceipt construction binding changed after construction")
    ValidationReceipt(
        value.receipt_id,
        value.catalog_id,
        value.catalog_path,
        value.authority_domain,
        value.lineage_id,
        value.epoch_id,
        value.epoch_revision,
        value.split,
        value.record_ids,
        value.target_artifact_id,
        value.target_skill_id,
        value.target_version,
        value.target_digest,
    )


def _validation_receipt_binding(value: ValidationReceipt) -> tuple[object, ...]:
    return (
        value.receipt_id,
        value.catalog_id,
        value.catalog_path,
        value.authority_domain,
        value.lineage_id,
        value.epoch_id,
        value.epoch_revision,
        value.split.value if type(value.split) is ValidationSplit else value.split,
        value.record_ids,
        value.target_artifact_id,
        value.target_skill_id,
        value.target_version,
        value.target_digest,
    )


def _snapshot_validation_receipt(value: object) -> ValidationReceipt:
    if type(value) is not ValidationReceipt:
        raise ValueError("receipt must be an exact ValidationReceipt")
    checked = cast(ValidationReceipt, value)
    _validate_receipt_fields(checked)
    return ValidationReceipt.from_dict(checked.to_dict())


def _snapshot_validation_target(value: object) -> ValidationTarget:
    if type(value) is not ValidationTarget:
        raise ValueError("target must be an exact ValidationTarget")
    checked = cast(ValidationTarget, value)
    return ValidationTarget(
        checked.artifact_id,
        checked.skill_id,
        checked.version,
        checked.artifact_digest,
    )


def _validated_epoch_store(store: object) -> tuple[str, str]:
    if type(store) is not EvaluationEpochStore:
        raise ValueError("store must be an exact EvaluationEpochStore")
    checked_store = cast(EvaluationEpochStore, store)
    if checked_store._binding != (
        checked_store._database_path,
        checked_store._authority_domain,
    ):
        raise ValueError("EvaluationEpochStore binding changed after construction")
    return checked_store._database_path, checked_store._authority_domain


def _canonical_catalog_path(value: str | os.PathLike[str]) -> str:
    if type(value) is not str and not isinstance(value, os.PathLike):
        raise ValueError("catalog path must be an exact string or path-like value")
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(value))))


def _new_history_handle(
    path: str,
    domain: str,
    lineage_id: str,
    revision: int,
) -> EvaluationEpochHistory:
    handle = object.__new__(EvaluationEpochHistory)
    _EPOCH_HISTORY_STATE[handle] = _EpochHistoryHandleState(path, domain, lineage_id, revision)
    return handle


def _claim_catalog_identity(
    connection: sqlite3.Connection,
    table: str,
    domain: str,
    identity: str,
    lineage_id: str,
) -> None:
    if table not in {"epoch_claims", "model_version_claims", "harness_version_claims"}:
        raise RuntimeError("invalid evaluation authority catalog table")
    try:
        connection.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?)",
            (domain, identity, lineage_id),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError("identity is already claimed in this authority domain") from exc


def _serialize_history_entry(entry: _EpochHistoryEntry) -> str:
    snapshot = _validated_history_entry(entry)
    return json.dumps(
        {
            "lineage": [epoch.to_dict() for epoch in snapshot.lineage],
            "parent_epoch_ids": list(snapshot.parent_epoch_ids),
            "records": [
                {"record_id": record_id, "record": record.to_dict()}
                for record_id, record in snapshot.records
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _deserialize_history_entry(payload: object) -> _EpochHistoryEntry:
    if type(payload) is not str:
        raise ValueError("evaluation authority catalog payload must be exact text")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("evaluation authority catalog payload is not valid JSON") from exc
    values = _require_exact_mapping(
        data, {"lineage", "parent_epoch_ids", "records"}, "epoch lineage payload"
    )
    if type(values["lineage"]) is not list or type(values["parent_epoch_ids"]) is not list:
        raise ValueError("evaluation authority catalog lineage arrays are invalid")
    if type(values["records"]) is not list:
        raise ValueError("evaluation authority catalog records must be an array")
    lineage = tuple(
        EvaluationEpoch.from_dict(item) for item in cast(list[object], values["lineage"])
    )
    parents = tuple(
        cast(str | None, item) for item in cast(list[object], values["parent_epoch_ids"])
    )
    records: list[tuple[str, V5EvaluationRecord]] = []
    for item in cast(list[object], values["records"]):
        record_values = _require_exact_mapping(
            item, {"record_id", "record"}, "authoritative evaluation record"
        )
        records.append(
            (
                cast(str, record_values["record_id"]),
                V5EvaluationRecord.from_dict(cast(Mapping[str, object], record_values["record"])),
            )
        )
    return _validated_history_entry(_EpochHistoryEntry(lineage, parents, tuple(records)))


def _validate_catalog(
    connection: sqlite3.Connection,
    domain: str,
    lineage_id: str,
    tip_epoch_id: object,
    entry: _EpochHistoryEntry,
) -> None:
    snapshot = _validated_history_entry(entry)
    if type(tip_epoch_id) is not str or tip_epoch_id != snapshot.lineage[-1].epoch_id:
        raise ValueError("evaluation authority catalog tip does not match lineage")
    expected = {
        "epoch_claims": {epoch.epoch_id for epoch in snapshot.lineage},
        "model_version_claims": {epoch.model_version for epoch in snapshot.lineage},
        "harness_version_claims": {epoch.harness_version for epoch in snapshot.lineage},
    }
    for table, identities in expected.items():
        rows = connection.execute(
            f"SELECT identity FROM {table} WHERE authority_domain = ? AND lineage_id = ?",
            (domain, lineage_id),
        ).fetchall()
        if {row[0] for row in rows} != identities:
            raise ValueError("evaluation authority catalog claims do not match lineage")


def _begin_history_read(
    history: EvaluationEpochHistory,
) -> tuple[_EpochHistoryHandleState, sqlite3.Connection, _EpochHistoryEntry]:
    state = _EPOCH_HISTORY_STATE[history]
    connection = _open_epoch_database(state.database_path)
    try:
        row = connection.execute(
            "SELECT revision, tip_epoch_id, payload FROM epoch_lineages "
            "WHERE authority_domain = ? AND lineage_id = ?",
            (state.authority_domain, state.lineage_id),
        ).fetchone()
        if row is None:
            raise ValueError("authoritative evaluation lineage no longer exists")
        revision = _require_nonnegative_integer(row[0], "revision")
        if revision != state.revision:
            raise RuntimeError("evaluation history handle has a stale revision")
        entry = _deserialize_history_entry(row[2])
        _validate_catalog(connection, state.authority_domain, state.lineage_id, row[1], entry)
        return state, connection, entry
    except Exception:
        connection.close()
        raise


def _begin_history_transition(
    history: EvaluationEpochHistory,
) -> tuple[_EpochHistoryHandleState, sqlite3.Connection, _EpochHistoryEntry]:
    state = _EPOCH_HISTORY_STATE[history]
    connection = _open_epoch_database(state.database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT revision, tip_epoch_id, payload FROM epoch_lineages "
            "WHERE authority_domain = ? AND lineage_id = ?",
            (state.authority_domain, state.lineage_id),
        ).fetchone()
        if row is None:
            raise ValueError("authoritative evaluation lineage no longer exists")
        revision = _require_nonnegative_integer(row[0], "revision")
        if revision != state.revision:
            raise RuntimeError("evaluation history handle has a stale revision")
        entry = _deserialize_history_entry(row[2])
        _validate_catalog(connection, state.authority_domain, state.lineage_id, row[1], entry)
        return state, connection, entry
    except Exception:
        connection.rollback()
        connection.close()
        raise


def _commit_history_transition(
    history: EvaluationEpochHistory,
    state: _EpochHistoryHandleState,
    connection: sqlite3.Connection,
    entry: _EpochHistoryEntry,
) -> None:
    try:
        snapshot = _validated_history_entry(entry)
        next_revision = state.revision + 1
        result = connection.execute(
            "UPDATE epoch_lineages SET revision = ?, tip_epoch_id = ?, payload = ? "
            "WHERE authority_domain = ? AND lineage_id = ? AND revision = ?",
            (
                next_revision,
                snapshot.lineage[-1].epoch_id,
                _serialize_history_entry(snapshot),
                state.authority_domain,
                state.lineage_id,
                state.revision,
            ),
        )
        if result.rowcount != 1:
            raise RuntimeError("evaluation history transition failed a stale revision check")
        connection.commit()
    except Exception:
        connection.rollback()
        connection.close()
        raise
    connection.close()
    _EPOCH_HISTORY_STATE[history] = _EpochHistoryHandleState(
        state.database_path,
        state.authority_domain,
        state.lineage_id,
        next_revision,
    )


def _require_exact_history(value: object) -> EvaluationEpochHistory:
    _require_epoch_process()
    if type(value) is not EvaluationEpochHistory:
        raise ValueError("mutation requires an authoritative EvaluationEpochHistory")
    if value not in _EPOCH_HISTORY_STATE:
        raise ValueError("mutation requires a store-derived authoritative EvaluationEpochHistory")
    return value


def _require_epoch_process() -> None:
    if os.getpid() != _EPOCH_HISTORY_PROCESS_ID:
        raise RuntimeError("evaluation epoch authority cannot cross a process boundary")


def _require_nonnegative_integer(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative built-in integer")
    return value


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
        record_type = cast(Any, V5EvaluationRecord)
        record_value = cast(Any, value)
        return cast(V5EvaluationRecord, record_type.from_dict(record_value.to_dict()))
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


def _candidate_claim_payload(
    value: CandidateTargetClaim,
    include_digest: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "catalog_id": value.catalog_id,
        "authority_domain": value.authority_domain,
        "lineage_id": value.lineage_id,
        "epoch_id": value.epoch_id,
        "epoch_revision": value.epoch_revision,
        "candidate_id": value.candidate_id,
        "artifact_digest": value.artifact_digest,
        "model_version": value.model_version,
        "harness_version": value.harness_version,
    }
    if include_digest:
        payload["claim_digest"] = value.claim_digest
    return payload


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
            cast(Any, EvidenceReference)(
                checked_reference.reference_id, checked_reference.source_kind
            )
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
    "EvaluationAuthority",
    "EvaluationEpoch",
    "EvaluationEpochHistory",
    "EvaluationEpochStore",
    "LearningSurface",
    "LessonCharacteristics",
    "LessonKind",
    "LessonProposal",
    "LessonReviewStatus",
    "ValidationReceipt",
    "ValidationSplit",
    "ValidationTarget",
    "append_epoch_record",
    "begin_candidate_epoch",
    "classify_learning_surface",
    "close_evaluation_epoch",
    "evaluation_record_cell_id",
    "evaluation_record_id",
    "validate_epoch_record",
]
