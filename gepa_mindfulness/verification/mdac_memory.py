"""Opt-in, content-free MDAC memory catalog over a host's current artifact workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .artifact_evidence import ArtifactAccessRequest, ArtifactQuery, _dependency_items
from .artifact_records import (
    ADMISSIONS,
    ArtifactSnapshot,
    DerivedInterpretation,
    artifact_digest,
    payload_digest,
    unique,
)
from .artifact_topology import EvidenceTopology
from .debate_records import _digest, _record
from .diagnostic_records import DiagnosticRecord, _text, choice, records, restore_records, strings
from .evidence_workspace import (
    EvidenceWorkspace,
    _visible_workspace_items,
    retrieve_evidence_workspace,
)
from .state import parse_rfc3339_datetime


@dataclass(frozen=True, slots=True)
class MemoryEntry(DiagnosticRecord):
    """A private host pointer with explicit provenance, retention and policy versions."""

    schema_version = "mdac-memory-entry-v1"
    item_id: str
    item_digest: str
    state_kind: str
    subject: str
    principal_id: str
    scope_id: str
    session_id: str
    topic_id: str | None
    retention: str
    expires_at: str
    authorization_version: str
    obligation_version: str

    def __post_init__(self) -> None:
        for name in (
            "item_id",
            "principal_id",
            "scope_id",
            "session_id",
            "authorization_version",
            "obligation_version",
        ):
            _text(getattr(self, name), name)
        if self.topic_id is not None:
            _text(self.topic_id, "topic_id")
        _digest(self.item_digest)
        choice(self.state_kind, "state kind", ("prescribed", "observed", "inferred", "predicted"))
        choice(self.subject, "subject", ("state", "authorization", "obligation"))
        choice(self.retention, "retention", ("session", "cross_session"))
        parse_rfc3339_datetime(self.expires_at, "expires_at")


@dataclass(frozen=True, slots=True)
class MemoryCatalog(DiagnosticRecord):
    """Private, revision-bound metadata. No quotations, summaries or assessment payloads."""

    schema_version = "mdac-memory-catalog-v1"
    restorers = {"entries": lambda value: restore_records(value, MemoryEntry)}
    snapshot_digest: str
    entries: tuple[MemoryEntry, ...]

    def __post_init__(self) -> None:
        _digest(self.snapshot_digest)
        if type(self.entries) is not tuple or len(self.entries) > 256:
            raise ValueError("catalog requires at most 256 entries")
        object.__setattr__(self, "entries", records(self.entries, MemoryEntry))
        unique([entry.item_id for entry in self.entries], "entry ids")


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Host-declared active session, topic and unresolved-obligation ledger version."""

    session_id: str
    topic_id: str
    obligation_version: str

    def __post_init__(self) -> None:
        for name in ("session_id", "topic_id", "obligation_version"):
            _text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class MemoryFrame:
    """One atomically read, host-owned revision from a local or external store."""

    catalog: MemoryCatalog
    snapshot: ArtifactSnapshot
    workspace: EvidenceWorkspace
    topology: EvidenceTopology


def revise_memory(
    previous: MemoryCatalog | None,
    snapshot: ArtifactSnapshot,
    *,
    upsert: tuple[MemoryEntry, ...] = (),
    delete: tuple[str, ...] = (),
    as_of: str | None = None,
    retain: Callable[[MemoryEntry], bool] | None = None,
    enabled: bool = False,
) -> MemoryCatalog | None:
    """Refresh pointers and invalidate changed dependencies before returning a new catalog."""
    if enabled is not True:
        return None
    now = parse_rfc3339_datetime(as_of, "as_of")
    snapshot = _record(snapshot, ArtifactSnapshot)
    items = {item.item_id: item for item in snapshot.sources + snapshot.interpretations}
    entries = (
        {} if previous is None else {e.item_id: e for e in _record(previous, MemoryCatalog).entries}
    )
    updates = records(upsert, MemoryEntry)
    update_ids = [e.item_id for e in updates]
    unique(update_ids, "upsert ids")
    changes = set(strings(delete, "delete"))
    if changes.intersection(update_ids):
        raise ValueError("delete and upsert overlap")
    for key, entry in entries.items():
        if (
            key not in items
            or artifact_digest(items[key]) != entry.item_digest
            or now >= parse_rfc3339_datetime(entry.expires_at, "expires_at")
        ):
            changes.add(key)
    approved = {}
    for entry in updates:
        key = entry.item_id
        item = items.get(key)
        if item is None or artifact_digest(item) != entry.item_digest:
            raise ValueError("upsert requires an exact live item")
        if isinstance(item, DerivedInterpretation) and entry.state_kind in {
            "observed",
            "prescribed",
        }:
            raise ValueError("derived state must remain inferred or predicted")
        if (
            now < parse_rfc3339_datetime(entry.expires_at, "expires_at")
            and retain is not None
            and retain(entry) is True
        ):
            approved[key] = entry
        if entries.get(key) != approved.get(key):
            changes.add(key)
    # Every transformation depends on its inputs' bytes AND their retention/usage contracts.
    retained = {
        key: entry
        for key, entry in entries.items()
        if key in items and changes.isdisjoint(_dependency_items(key, items))
    }
    retained.update(approved)
    complete = tuple(
        retained[key]
        for key in sorted(retained)
        if all(dep in retained for dep in _dependency_items(key, items))
    )
    return MemoryCatalog(artifact_digest(snapshot), complete)


def retrieve_memory(
    load: Callable[[], MemoryFrame] | None,
    query: ArtifactQuery,
    context: MemoryContext,
    *,
    detail: str = "pointer",
    restore_handle: str | None = None,
    authorize: Callable[[ArtifactAccessRequest], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any] | None:
    """Project allowed workspace rows after a fresh host load, ACL check and retention check."""
    if enabled is not True or load is None:
        return None
    choice(detail, "detail", ("pointer", "full"))
    query = _record(query, ArtifactQuery)
    if type(context) is not MemoryContext:
        raise ValueError("expected MemoryContext")
    context.__post_init__()
    frame = load()
    if type(frame) is not MemoryFrame:
        raise ValueError("expected MemoryFrame")
    snapshot = _record(frame.snapshot, ArtifactSnapshot)
    stored = _record(frame.catalog, MemoryCatalog)
    if stored.snapshot_digest != artifact_digest(snapshot):
        raise ValueError("memory snapshot revision mismatch")
    # The host owns the frame. Recheck all persisted bindings on every read.
    catalog = revise_memory(stored, snapshot, as_of=query.assessed_at, enabled=True)
    assert catalog is not None
    active_workspace = _record(frame.workspace, EvidenceWorkspace)
    current = retrieve_evidence_workspace(
        active_workspace, snapshot, frame.topology, query, authorize=authorize, enabled=True
    )
    allowed = {
        entry.item_id: entry
        for entry in catalog.entries
        if entry.principal_id == query.principal_id
        and entry.scope_id == query.scope_id
        and entry.authorization_version == query.policy_version
        and entry.obligation_version == context.obligation_version
        and (entry.topic_id is None or entry.topic_id == context.topic_id)
        and (entry.retention == "cross_session" or entry.session_id == context.session_id)
    }
    items = {item.item_id: item for item in snapshot.sources + snapshot.interpretations}
    audit = current["retrieval"]
    visible = _visible_workspace_items(
        active_workspace,
        tuple(audit["selected_item_ids"]),
        {row["item_id"]: row for row in audit["item_rows"]},
    )
    permitted = {
        key: row
        for key, row in zip(visible, current["producer_view"]["items"], strict=True)
        if set(_dependency_items(key, items)) <= allowed.keys()
    }
    # Bind restore tokens to visible item revisions and the request. Tokens confer no access.
    binding = payload_digest(
        (
            query.request_id,
            query.principal_id,
            query.scope_id,
            context.session_id,
            context.topic_id,
            query.policy_version,
            context.obligation_version,
        )
    )
    handles = {
        row["handle"]: payload_digest(
            (binding, key, tuple(allowed[dep] for dep in _dependency_items(key, items)))
        )
        for key, row in permitted.items()
    }
    rows = []
    for key, row in permitted.items():
        if restore_handle is not None and restore_handle != handles[row["handle"]]:
            continue
        entry = allowed[key]
        if row["kind"] == "synthetic" and entry.state_kind not in {"inferred", "predicted"}:
            raise ValueError("derived state must remain inferred or predicted")
        public = {
            "handle": handles[row["handle"]],
            "source_handles": [handles[source] for source in row["source_handles"]],
            "state_kind": entry.state_kind,
            "subject": entry.subject,
            "channel": row["channel"],
            "training_eligibility": row["training_eligibility"],
        }
        if detail == "full":
            public.update(text=row["text"], observed_at=row["observed_at"])
        rows.append(public)
    admissions = [admission.value for admission in ADMISSIONS]
    return {
        "schema_version": "mdac-memory-view-v1",
        "confers_authority": False,
        "optimizer_eligible": False,
        "training_eligibility": max(
            (row["training_eligibility"] for row in rows),
            key=admissions.index,
            default="DEVELOPMENT",
        ),
        "items": rows,
    }
