"""Explicit structural evidence routes: conjunctive prerequisites and alternative support."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Third-party
# Local
from .artifact_records import (
    ArtifactDiagnosticRecord,
    ArtifactSnapshot,
    artifact_digest,
    record_tuple,
    unique,
)
from .claim_graph import ClaimGraph
from .debate_records import _digest, _record
from .diagnostic_records import _text, restore_records, strings


@dataclass(frozen=True)
class SupportRoute(ArtifactDiagnosticRecord):
    """One ordered conjunction of premises; other routes are independent alternatives."""

    route_id: str
    conclusion_claim_id: str
    prerequisite_claim_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    schema_version = "support-route-v1"

    def __post_init__(self) -> None:
        _text(self.route_id, "route_id")
        _text(self.conclusion_claim_id, "conclusion_claim_id")
        for name in ("prerequisite_claim_ids", "item_ids"):
            object.__setattr__(self, name, strings(getattr(self, name), name, required=True))
        if not 1 <= len(self.prerequisite_claim_ids) <= 16 or len(self.item_ids) > 256:
            raise ValueError("route exceeds prerequisite/item bounds")
        if self.conclusion_claim_id in self.prerequisite_claim_ids:
            raise ValueError("conclusion cannot be its own prerequisite")


@dataclass(frozen=True)
class EvidenceTopology(ArtifactDiagnosticRecord):
    """A declared claim graph and bounded support alternatives bound to one snapshot."""

    snapshot_digest: str
    graph: ClaimGraph
    routes: tuple[SupportRoute, ...]
    schema_version = "evidence-topology-v1"
    restorers = {
        "graph": ClaimGraph.from_dict,
        "routes": lambda v: restore_records(v, SupportRoute),
    }

    def __post_init__(self) -> None:
        _digest(self.snapshot_digest)
        object.__setattr__(self, "graph", _record(self.graph, ClaimGraph))
        object.__setattr__(self, "routes", record_tuple(self.routes, SupportRoute))
        if len(self.graph.nodes) > 256 or len(self.graph.dependencies) > 512:
            raise ValueError("topology graph exceeds bounds")
        if len(self.routes) > 64:
            raise ValueError("topology exceeds 64 routes")
        unique([r.route_id for r in self.routes], "route IDs")


def _validate_route(
    snapshot: ArtifactSnapshot, topology: EvidenceTopology, route: SupportRoute
) -> None:
    claims = {n.claim.claim_id for n in topology.graph.nodes}
    selected = set(route.prerequisite_claim_ids) | {route.conclusion_claim_id}
    if not selected <= claims:
        raise ValueError("route references unknown graph claim")
    items = {s.item_id: s for s in snapshot.sources + snapshot.interpretations}
    if not set(route.item_ids) <= set(items):
        raise ValueError("route references unknown item")
    bound = {items[key].claim.claim_id for key in route.item_ids}
    if not bound <= set(route.prerequisite_claim_ids):
        raise ValueError("item must bind a route prerequisite")
    order = {key: i for i, key in enumerate(route.prerequisite_claim_ids)}
    order[route.conclusion_claim_id] = len(order)
    adjacency: dict[str, set[str]] = {key: set() for key in selected}
    for edge in topology.graph.dependencies:
        parent, child = edge.parent_claim_id, edge.child_claim_id
        if parent not in selected or edge.dependency_type == "contradicts":
            continue
        if edge.dependency_type == "requires" and child not in selected:
            raise ValueError("route omits a mandatory dependency")
        if child in selected:
            if order[child] >= order[parent]:
                raise ValueError("prerequisites must precede their dependent claim")
            adjacency[parent].add(child)
    reached: set[str] = set()
    pending = [route.conclusion_claim_id]
    while pending:
        key = pending.pop()
        if key not in reached:
            reached.add(key)
            pending.extend(adjacency[key])
    if reached != selected:
        raise ValueError("route prerequisites must connect to conclusion")
    for claim in selected:
        if not adjacency[claim] and claim not in bound:
            raise ValueError("every leaf prerequisite requires a bound item")


def validate_topology(snapshot: ArtifactSnapshot, topology: EvidenceTopology) -> None:
    """Reject mismatched claims, unbound items and undeclared or reordered prerequisites."""
    snapshot = _record(snapshot, ArtifactSnapshot)
    topology = _record(topology, EvidenceTopology)
    if topology.snapshot_digest != artifact_digest(snapshot):
        raise ValueError("topology snapshot digest mismatch")
    claims = {c.claim_id: c for c in snapshot.state.claims}
    if any(claims.get(n.claim.claim_id) != n.claim for n in topology.graph.nodes):
        raise ValueError("graph claim differs from snapshot")
    for route in topology.routes:
        _validate_route(snapshot, topology, route)


def assess_support_routes(
    snapshot: ArtifactSnapshot,
    topology: EvidenceTopology,
    *,
    available_item_ids: tuple[str, ...],
    enabled: bool = False,
) -> dict[str, Any]:
    """Report structural availability only; authorized evidence and truth are separate checks."""
    if enabled is not True:
        raise ValueError("route assessment requires enabled=True")
    snapshot = _record(snapshot, ArtifactSnapshot)
    topology = _record(topology, EvidenceTopology)
    validate_topology(snapshot, topology)
    available = set(strings(available_item_ids, "available_item_ids"))
    if not available <= {s.item_id for s in snapshot.sources + snapshot.interpretations}:
        raise ValueError("available item is not in snapshot")
    rows = []
    for route in topology.routes:
        missing = [key for key in route.item_ids if key not in available]
        rows.append(
            dict(
                route_id=route.route_id,
                conclusion_claim_id=route.conclusion_claim_id,
                status="blocked" if missing else "available",
                missing_item_ids=missing,
            )
        )
    return dict(
        structural_only=True,
        confers_authority=False,
        routes=rows,
        available_route_ids=[r["route_id"] for r in rows if r["status"] == "available"],
        available_conclusion_ids=sorted(
            {r["conclusion_claim_id"] for r in rows if r["status"] == "available"}
        ),
        contradictions=[
            e.to_dict() for e in topology.graph.dependencies if e.dependency_type == "contradicts"
        ],
    )
