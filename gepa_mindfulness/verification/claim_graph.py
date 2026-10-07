"""Public claim decomposition over canonical EvidenceClaim records; no truth authority."""

from __future__ import annotations

from dataclasses import dataclass

from mindful_trace_gepa.confidence import ConfidenceSource

from ..core.evidence import EvidenceReference
from .diagnostic_records import (
    DiagnosticRecord,
    _text,
    _unit,
    choice,
    public_refs,
    records,
    restore_records,
    restore_refs,
    strings,
)
from .state import EvidenceClaim


@dataclass(frozen=True, slots=True)
class ClaimNode(DiagnosticRecord):
    """Public source and prioritization metadata around an unchanged canonical claim."""

    claim: EvidenceClaim
    source_actor: str
    confidence: float | None
    confidence_source: str
    task_relevance: float
    decision_importance: float

    schema_version = "claim-node-v1"
    restorers = {"claim": EvidenceClaim.from_dict}

    def __post_init__(self) -> None:
        if type(self.claim) is not EvidenceClaim:
            raise ValueError("claim must be an exact EvidenceClaim")
        object.__setattr__(self, "claim", EvidenceClaim.from_dict(self.claim.to_dict()))
        public_refs(self.claim.evidence_refs)
        _text(self.source_actor, "source_actor")
        choice(
            self.confidence_source, "confidence_source", tuple(v.value for v in ConfidenceSource)
        )
        for name in ("confidence", "task_relevance", "decision_importance"):
            value = getattr(self, name)
            if value is not None or name != "confidence":
                _unit(value, name)


@dataclass(frozen=True, slots=True)
class ClaimDependency(DiagnosticRecord):
    """A conclusion-to-premise edge with publicly observable decomposition provenance."""

    parent_claim_id: str
    child_claim_id: str
    dependency_type: str
    evidence_refs: tuple[EvidenceReference, ...]

    schema_version = "claim-dependency-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        _text(self.parent_claim_id, "parent_claim_id")
        _text(self.child_claim_id, "child_claim_id")
        choice(self.dependency_type, "dependency_type", ("requires", "supports", "contradicts"))
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))


@dataclass(frozen=True, slots=True)
class ClaimDecomposition(DiagnosticRecord):
    """A declared decomposition; stability is an empirical proxy, never a theorem."""

    parent_claim_id: str
    child_claim_ids: tuple[str, ...]
    source_actor: str
    evidence_refs: tuple[EvidenceReference, ...]
    decomposition_instability: float | None

    schema_version = "claim-decomposition-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        _text(self.parent_claim_id, "parent_claim_id")
        _text(self.source_actor, "source_actor")
        object.__setattr__(
            self, "child_claim_ids", strings(self.child_claim_ids, "child_claim_ids", required=True)
        )
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))
        if self.decomposition_instability is not None:
            _unit(self.decomposition_instability, "decomposition_instability")


@dataclass(frozen=True, slots=True)
class ClaimGraph(DiagnosticRecord):
    """Closed acyclic public decomposition graph, bounded to 1024 claims."""

    nodes: tuple[ClaimNode, ...]
    dependencies: tuple[ClaimDependency, ...] = ()
    decompositions: tuple[ClaimDecomposition, ...] = ()

    schema_version = "claim-graph-v1"
    restorers = {
        "nodes": lambda v: restore_records(v, ClaimNode),
        "dependencies": lambda v: restore_records(v, ClaimDependency),
        "decompositions": lambda v: restore_records(v, ClaimDecomposition),
    }

    def __post_init__(self) -> None:
        for name, cls in (
            ("nodes", ClaimNode),
            ("dependencies", ClaimDependency),
            ("decompositions", ClaimDecomposition),
        ):
            object.__setattr__(self, name, records(getattr(self, name), cls))
        ids = {node.claim.claim_id for node in self.nodes}
        if not 1 <= len(self.nodes) <= 1024 or len(ids) != len(self.nodes):
            raise ValueError("graph requires 1..1024 unique claims")
        adjacency: dict[str, set[str]] = {key: set() for key in ids}
        for edge in self.dependencies:
            if edge.parent_claim_id not in ids or edge.child_claim_id not in ids:
                raise ValueError("dependency references unknown claim")
            if edge.child_claim_id in adjacency[edge.parent_claim_id]:
                raise ValueError("duplicate dependency")
            adjacency[edge.parent_claim_id].add(edge.child_claim_id)
        for decomposition in self.decompositions:
            if decomposition.parent_claim_id not in ids:
                raise ValueError("decomposition references unknown parent")
            if not set(decomposition.child_claim_ids).issubset(
                adjacency[decomposition.parent_claim_id]
            ):
                raise ValueError("decomposition children require matching dependency edges")
        # Kahn traversal avoids recursion failures on long adversarial graphs.
        incoming = {key: 0 for key in ids}
        for children in adjacency.values():
            for child in children:
                incoming[child] += 1
        ready = [key for key, count in incoming.items() if count == 0]
        visited = 0
        while ready:
            parent = ready.pop()
            visited += 1
            for child in adjacency[parent]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
        if visited != len(ids):
            raise ValueError("claim graph contains a cycle")

    @property
    def unresolved_claim_ids(self) -> tuple[str, ...]:
        """Retain claims without a current evidence-supported resolution."""
        return tuple(
            node.claim.claim_id
            for node in self.nodes
            if node.claim.status in {"unverified", "unavailable", "stale"}
        )
