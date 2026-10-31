"""Inspectable workspace indexes reuse artifact lineage and current host access checks."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Third-party
# Local
from ..core.evidence import EvidenceReference
from .artifact_evidence import (
    ArtifactAccessRequest,
    ArtifactQuery,
    retrieve_artifact_evidence,
)
from .artifact_records import (
    ADMISSIONS,
    ArtifactDiagnosticRecord,
    ArtifactSnapshot,
    artifact_digest,
    record_tuple,
    unique,
)
from .artifact_topology import EvidenceTopology, validate_topology
from .debate_records import _digest, _record
from .diagnostic_records import _text, public_refs, restore_records, restore_refs

MAX_OUTPUT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class WorkspaceNode(ArtifactDiagnosticRecord):
    """Host-only placement of an existing item; IDs are never filesystem operations."""

    node_id: str
    item_id: str
    parent_id: str | None = None
    schema_version = "workspace-node-v1"

    def __post_init__(self) -> None:
        """Validate private coordinates without treating them as executable commands."""
        for name in ("node_id", "item_id"):
            _text(getattr(self, name), name)
        if self.parent_id is not None:
            _text(self.parent_id, "parent_id")
        if self.parent_id == self.node_id:
            raise ValueError("workspace node cannot parent itself")


@dataclass(frozen=True)
class EvidenceWorkspace(ArtifactDiagnosticRecord):
    """Versioned non-authoritative index; the host retains the complete original snapshot."""

    snapshot_digest: str
    nodes: tuple[WorkspaceNode, ...]
    source_refs: tuple[EvidenceReference, ...]
    schema_version = "evidence-workspace-v1"
    restorers = {
        "nodes": lambda value: restore_records(value, WorkspaceNode),
        "source_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        """Bound serialization and validate unique, acyclic placements."""
        _digest(self.snapshot_digest)
        if type(self.nodes) is not tuple or not 1 <= len(self.nodes) <= 256:
            raise ValueError("workspace requires 1..256 nodes")
        object.__setattr__(self, "nodes", record_tuple(self.nodes, WorkspaceNode))
        refs = public_refs(self.source_refs, required=True)
        if len(refs) > 64:
            raise ValueError("workspace exceeds 64 source references")
        object.__setattr__(self, "source_refs", refs)
        unique([node.node_id for node in self.nodes], "workspace node IDs")
        unique([node.item_id for node in self.nodes], "workspace item IDs")
        serialized = json.dumps(
            [[node.to_dict() for node in self.nodes], [ref.to_dict() for ref in refs]],
            ensure_ascii=False,
        ).encode("utf-8")
        if len(serialized) > 262144:
            raise ValueError("workspace index capacity exceeded")
        nodes = {node.node_id: node for node in self.nodes}
        for node in self.nodes:
            seen = {node.node_id}
            parent_id = node.parent_id
            while parent_id is not None:
                if parent_id not in nodes or parent_id in seen:
                    raise ValueError("unknown or cyclic workspace parent")
                seen.add(parent_id)
                if len(seen) > 16:
                    raise ValueError("workspace exceeds 16-level depth")
                parent_id = nodes[parent_id].parent_id


def prepare_evidence_workspace(
    snapshot: ArtifactSnapshot,
    nodes: tuple[WorkspaceNode, ...],
    source_refs: tuple[EvidenceReference, ...],
    *,
    enabled: bool = False,
) -> EvidenceWorkspace:
    """Bind an index to existing lineage, retaining every raw source exactly once."""
    if enabled is not True:
        raise ValueError("evidence workspace requires enabled=True")
    snapshot = _record(snapshot, ArtifactSnapshot)
    workspace = EvidenceWorkspace(artifact_digest(snapshot), nodes, source_refs)
    indexed = {node.item_id for node in workspace.nodes}
    sources = {source.item_id for source in snapshot.sources}
    derived = {item.item_id: item for item in snapshot.interpretations}
    if not sources <= indexed or not indexed <= sources | set(derived):
        raise ValueError("workspace must retain all raw sources and only known items")
    placements = {node.node_id: node for node in workspace.nodes}
    for node in workspace.nodes:
        if node.parent_id is None:
            continue
        parent = placements[node.parent_id]
        if parent.item_id not in derived or node.item_id not in {
            item_id for item_id, _ in derived[parent.item_id].inputs
        }:
            raise ValueError("workspace parent must be a direct derived-input relationship")
    return workspace


def retrieve_evidence_workspace(
    workspace: EvidenceWorkspace,
    snapshot: ArtifactSnapshot,
    topology: EvidenceTopology,
    query: ArtifactQuery,
    *,
    authorize: Callable[[ArtifactAccessRequest], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Return a private audit and a catalog projected solely from current permitted retrieval.

    Only producer_view may be sent to an answering agent. The complete result retains private
    host identities and is diagnostic, never optimizer evidence or a factual certificate.
    """
    if enabled is not True:
        raise ValueError("workspace retrieval requires enabled=True")
    workspace = _record(workspace, EvidenceWorkspace)
    snapshot = _record(snapshot, ArtifactSnapshot)
    topology = _record(topology, EvidenceTopology)
    query = _record(query, ArtifactQuery)
    prepared = prepare_evidence_workspace(
        snapshot, workspace.nodes, workspace.source_refs, enabled=True
    )
    if workspace != prepared:
        raise ValueError("workspace snapshot binding mismatch")
    validate_topology(snapshot, topology)
    # Reserve repeated source handles and both private/public text before any host callback.
    raw_size = len(
        json.dumps(
            [value.to_dict() for value in (workspace, snapshot, topology, query)],
            ensure_ascii=False,
        ).encode("utf-8")
    )
    item_count = len(snapshot.sources) + len(snapshot.interpretations)
    if (item_count + 8) * raw_size + 262144 > MAX_OUTPUT_BYTES:
        raise ValueError("workspace retrieval output capacity exceeded")
    retrieval = retrieve_artifact_evidence(
        snapshot, topology, query, authorize=authorize, enabled=True
    )
    permitted = dict(zip(retrieval.selected_item_ids, retrieval.producer_view["items"]))
    rows = {row["item_id"]: row for row in retrieval.item_rows}
    placements = {node.node_id: node for node in workspace.nodes}
    active = {node.item_id for node in workspace.nodes}
    # Every referenced source must itself be recoverable in this exact permitted view.
    visible = [
        item_id
        for item_id in retrieval.selected_item_ids
        if item_id in active and set(rows[item_id]["ancestor_source_ids"]) <= set(permitted)
    ]
    handles = {item_id: f"item-{i}" for i, item_id in enumerate(visible)}
    node_by_item = {node.item_id: node for node in workspace.nodes}
    sources = {source.item_id for source in snapshot.sources}
    catalog = []
    for item_id in visible:
        node = node_by_item[item_id]
        parent_item = placements[node.parent_id].item_id if node.parent_id is not None else None
        original = permitted[item_id]
        catalog.append(
            dict(
                handle=handles[item_id],
                parent_handle=handles.get(parent_item),
                kind="source" if item_id in sources else "synthetic",
                training_eligibility=rows[item_id]["inherited_training_eligibility"],
                text=original["text"],
                location=original["location"],
                observed_at=original["observed_at"],
                channel=original["channel"],
                source_handles=[handles[source] for source in rows[item_id]["ancestor_source_ids"]],
            )
        )
    admissions = tuple(admission.value for admission in ADMISSIONS)
    host_admission = max(
        (artifact.source_training_eligibility.value for artifact in snapshot.artifacts),
        key=admissions.index,
        default="DEVELOPMENT",
    )
    view_admission = max(
        (row["training_eligibility"] for row in catalog),
        key=admissions.index,
        default="DEVELOPMENT",
    )
    return dict(
        schema_version="workspace-retrieval-v1",
        training_eligibility=host_admission,
        optimizer_eligible=False,
        confers_authority=False,
        workspace_digest=artifact_digest(workspace),
        retrieval=retrieval.to_dict(),
        producer_view=dict(
            training_eligibility=view_admission,
            public_query=query.public_query,
            items=catalog,
        ),
    )
