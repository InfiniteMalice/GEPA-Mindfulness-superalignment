"""Experimental host-only immutable raw archive; no accepted claims or optimizer authority."""

# Standard library
from __future__ import annotations

import os
import stat
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility
from mindful_trace_gepa.action_bound_events import ActionRecord

from .artifact_records import ArtifactRecord, ArtifactSnapshot
from .cva_records import CVARecord, _record, digest, evaluator, references, text
from .diagnostic_records import restore_refs
from .runtime_governance import (
    AuthorityGrantRegistry,
    AuthorizationDecision,
    RuntimeCapability,
    TrustedClock,
    consume_authorization,
)
from .state import ArtifactObservation, EvidenceState, parse_rfc3339_datetime

MAX_SOURCE_BYTES = 32 * 1024 * 1024


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("research archive requires enabled=True")


def _now(clock: TrustedClock) -> str:
    value = clock.now()
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("archive clock requires aware datetime")
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class ResearchSource(CVARecord):
    artifact_id: str
    version: str
    source_digest: str
    byte_count: int
    evidence_refs: tuple[EvidenceReference, ...]
    actor_id: str
    created_at: str

    schema_version = "research-raw-source-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        for name in ("artifact_id", "version", "actor_id"):
            text(getattr(self, name), name)
        digest(self.source_digest)
        if type(self.byte_count) is not int or not 1 <= self.byte_count <= MAX_SOURCE_BYTES:
            raise ValueError("raw source byte budget exceeded")
        references(self.evidence_refs)
        parse_rfc3339_datetime(self.created_at, "created_at")


def propose_research_source(
    artifact_id: str,
    version: str,
    payload: bytes,
    *,
    actor_id: str,
    evidence_refs: tuple[EvidenceReference, ...],
    clock: TrustedClock,
    enabled: bool = False,
) -> ResearchSource:
    _enabled(enabled)
    if type(payload) is not bytes or not 1 <= len(payload) <= MAX_SOURCE_BYTES:
        raise ValueError("source requires exact nonempty bytes within declared budget")
    return ResearchSource(
        artifact_id,
        version,
        sha256(payload).hexdigest(),
        len(payload),
        evidence_refs,
        actor_id,
        _now(clock),
    )


@dataclass(frozen=True)
class ResearchCatalog(CVARecord):
    store_id: str
    revision: int
    snapshot: ArtifactSnapshot

    schema_version = "research-raw-catalog-v1"
    restorers = {"snapshot": ArtifactSnapshot.from_dict}

    def __post_init__(self) -> None:
        text(self.store_id, "store_id")
        if type(self.snapshot) is not ArtifactSnapshot:
            raise ValueError("catalog requires existing ArtifactSnapshot")
        snapshot = ArtifactSnapshot.from_dict(self.snapshot.to_dict())
        if (
            type(self.revision) is not int
            or self.revision != len(snapshot.artifacts)
            or snapshot.entity_ids != (self.store_id,)
            or snapshot.sources
            or snapshot.interpretations
            or snapshot.state.claims
        ):
            raise ValueError("raw catalog cannot admit claims or discard version history")
        for artifact in snapshot.artifacts:
            if (
                artifact.entity_ids != (self.store_id,)
                or artifact.availability != "available"
                or artifact.source_training_eligibility is not TrainingEligibility.DEVELOPMENT
            ):
                raise ValueError("raw archive sources remain diagnostic DEVELOPMENT data")


@dataclass(frozen=True)
class SourceAdmissionPacket(CVARecord):
    archive_identity: str
    before_digest: str
    source: ResearchSource

    schema_version = "research-source-admission-packet-v1"
    restorers = {"source": ResearchSource.from_dict}

    def __post_init__(self) -> None:
        digest(self.archive_identity)
        digest(self.before_digest)
        _record(self.source, ResearchSource)


@dataclass(frozen=True)
class ArchiveHostContext:
    evaluator: TrustedEvaluatorContract
    authenticate: Callable[[SourceAdmissionPacket], bool] | None

    def __post_init__(self) -> None:
        evaluator(self.evaluator)
        if self.authenticate is not None and not callable(self.authenticate):
            raise ValueError("source authentication requires a host callable")


class ResearchArchive:
    """Single-writer host archive with pinned recovery, not an accepted evidence store.

    The host owns serialization of admissions and the current catalog pin across restarts.
    POSIX directory-relative exclusive I/O prevents payload-controlled paths/symlinks. Root
    provisioning is a separate host action; construction never creates files or credentials.
    """

    def __init__(
        self,
        root: Path | str,
        store_id: str,
        *,
        catalog: ResearchCatalog | None = None,
        trusted_catalog_digest: str | None = None,
        enabled: bool = False,
    ) -> None:
        _enabled(enabled)
        text(store_id, "store_id")
        if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            raise ValueError("research archive requires POSIX directory-relative no-follow I/O")
        self._root = Path(root).resolve(strict=True)
        info = self._root.stat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("archive root must be a host-provisioned directory")
        self._store_id = store_id
        self._root_identity = (info.st_dev, info.st_ino)
        self.identity_digest = content_digest(
            {
                "root": str(self._root),
                "device": info.st_dev,
                "inode": info.st_ino,
                "store_id": store_id,
            }
        )
        empty = ResearchCatalog(
            store_id, 0, ArtifactSnapshot((store_id,), (), (), (), EvidenceState(()))
        )
        if catalog is None:
            if any(self._root.iterdir()):
                raise PermissionError("existing archive requires its trusted current catalog pin")
            self._catalog = empty
        else:
            selected = _record(catalog, ResearchCatalog)
            if selected.store_id != store_id or trusted_catalog_digest != selected.digest:
                raise PermissionError("archive recovery requires the exact trusted catalog pin")
            self._catalog = selected
            if selected.revision:
                self._check_catalog_file(selected)

    @property
    def catalog(self) -> ResearchCatalog:
        return _record(self._catalog, ResearchCatalog)

    @contextmanager
    def _directory(self):
        fd = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != self._root_identity:
                raise PermissionError("physical archive root changed")
            yield fd
        finally:
            os.close(fd)

    def _read_file(self, name: str, limit: int = MAX_SOURCE_BYTES) -> bytes:
        try:
            with self._directory() as root_fd:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
                with os.fdopen(fd, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= limit:
                        raise ValueError("source is not a bounded regular file")
                    data = stream.read(limit + 1)
                    after = os.fstat(stream.fileno())
                    if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        raise ValueError("source changed during read")
        except OSError as error:
            raise ValueError("archive source is missing or unsafe") from error
        if not 1 <= len(data) <= limit:
            raise ValueError("source exceeds bounded read")
        return data

    def _write_file(self, name: str, payload: bytes) -> None:
        try:
            with self._directory() as root_fd:
                fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root_fd,
                )
                with os.fdopen(fd, "wb") as stream:
                    if stream.write(payload) != len(payload):
                        raise OSError("incomplete archive source write")
                    stream.flush()
                    os.fsync(stream.fileno())
        except FileExistsError:
            if self._read_file(name) != payload:
                raise ValueError("existing immutable source differs; never overwrite")
        if self._read_file(name) != payload:
            raise ValueError("source publication did not preserve exact bytes")

    def _catalog_name(self, catalog: ResearchCatalog) -> str:
        return f"catalog-{catalog.revision:04d}-{catalog.digest}.json"

    def _check_catalog_file(self, catalog: ResearchCatalog) -> None:
        expected = canonical_json(catalog.to_dict()).encode("utf-8")
        if self._read_file(self._catalog_name(catalog), 4 * 1024 * 1024) != expected:
            raise ValueError("persisted catalog differs from pinned metadata")

    def _proposed(self, before: ResearchCatalog, source: ResearchSource) -> ResearchCatalog:
        state = _record(before, ResearchCatalog)
        proposal = _record(source, ResearchSource)
        if state != self._catalog or state.store_id != self._store_id:
            raise PermissionError("stale or foreign current catalog")
        if any(
            (a.artifact_id, a.version) == (proposal.artifact_id, proposal.version)
            for a in state.snapshot.artifacts
        ):
            raise ValueError("duplicate immutable artifact version")
        observation = ArtifactObservation(
            "raw:" + proposal.digest,
            "research-source:" + content_digest(self._store_id) + ":" + proposal.source_digest,
            proposal.source_digest,
            proposal.created_at,
            proposal.evidence_refs,
        )
        artifact = ArtifactRecord(
            proposal.artifact_id,
            proposal.version,
            observation,
            (self._store_id,),
            "available",
            TrainingEligibility.DEVELOPMENT,
        )
        snapshot = ArtifactSnapshot(
            (self._store_id,), state.snapshot.artifacts + (artifact,), (), (), EvidenceState(())
        )
        return ResearchCatalog(self._store_id, state.revision + 1, snapshot)

    def append_scope(self, before: ResearchCatalog, source: ResearchSource) -> str:
        after = self._proposed(before, source)
        return "research-raw-append:" + content_digest(
            {
                "archive": self.identity_digest,
                "before": before.to_dict(),
                "after": after.to_dict(),
                "source": source.to_dict(),
            }
        )

    def append(
        self,
        before: ResearchCatalog,
        source: ResearchSource,
        payload: bytes,
        *,
        context: ArchiveHostContext,
        action: ActionRecord,
        authorization: AuthorizationDecision,
        grant_registry: AuthorityGrantRegistry,
        clock: TrustedClock,
        enabled: bool = False,
    ) -> ResearchCatalog:
        _enabled(enabled)
        proposal = _record(source, ResearchSource)
        if (
            type(payload) is not bytes
            or len(payload) != proposal.byte_count
            or sha256(payload).hexdigest() != proposal.source_digest
        ):
            raise ValueError("source bytes/digest differ from bound proposal")
        after = self._proposed(before, proposal)
        if type(context) is not ArchiveHostContext:
            raise PermissionError("exact trusted source-host context required")
        evaluator(context.evaluator)
        if context.evaluator.evaluator_id == proposal.actor_id:
            raise PermissionError("self-review cannot authenticate source custody")
        if parse_rfc3339_datetime(proposal.created_at, "source time") > parse_rfc3339_datetime(
            _now(clock), "host time"
        ):
            raise ValueError("source timestamp follows current trusted clock")
        packet = SourceAdmissionPacket(self.identity_digest, before.digest, proposal)
        detached = SourceAdmissionPacket.from_dict(packet.to_dict())
        expected = detached.to_dict()
        authenticated = False
        if context.authenticate is not None:
            try:
                authenticated = (
                    context.authenticate(detached) is True and detached.to_dict() == expected
                )
            except Exception:
                authenticated = False
        if not authenticated:
            raise PermissionError("source custody is not authenticated by the trusted host")
        scope = self.append_scope(before, proposal)
        if (
            type(action) is not ActionRecord
            or action.action_class != "research_source_append"
            or action.authorization_scope != scope
            or action.reversible is not True
            or type(authorization) is not AuthorizationDecision
            or authorization.capability is not RuntimeCapability.WRITE
        ):
            raise PermissionError("source append requires exact issued WRITE authority")
        consume_authorization(authorization, action, grant_registry=grant_registry, clock=clock)
        # Failed writes retain forensic orphans; advance catalog only after verification.
        self._write_file(proposal.source_digest + ".blob", payload)
        self._write_file(self._catalog_name(after), canonical_json(after.to_dict()).encode("utf-8"))
        self._check_catalog_file(after)
        self._catalog = after
        return self.catalog

    def read(
        self,
        catalog: ResearchCatalog,
        artifact_id: str,
        version: str,
        *,
        trusted_catalog_digest: str,
        enabled: bool = False,
    ) -> bytes:
        _enabled(enabled)
        selected = _record(catalog, ResearchCatalog)
        if selected.store_id != self._store_id or trusted_catalog_digest != selected.digest:
            raise PermissionError("raw source read requires the exact host catalog pin")
        if selected.revision:
            self._check_catalog_file(selected)
        artifact = next(
            (
                a
                for a in selected.snapshot.artifacts
                if (a.artifact_id, a.version) == (artifact_id, version)
            ),
            None,
        )
        if artifact is None:
            raise ValueError("source artifact/version is missing from catalog")
        expected = (
            "research-source:" + content_digest(self._store_id) + ":" + artifact.observation.digest
        )
        if artifact.observation.artifact_ref != expected:
            raise ValueError("source locator differs from this logical archive")
        payload = self._read_file(artifact.observation.digest + ".blob")
        if sha256(payload).hexdigest() != artifact.observation.digest:
            raise ValueError("immutable source digest mismatch")
        return payload
