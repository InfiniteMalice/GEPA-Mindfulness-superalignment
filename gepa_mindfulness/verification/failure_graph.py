"""Immutable, epistemically qualified failure graphs."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

from gepa_mindfulness.core.evidence import EvidenceReference

from .state import (
    _require_exact_mapping,
    _require_nonblank_string,
    _require_rfc3339,
    _snapshot_evidence_refs,
)

RootCauseStatus = Literal["supported", "hypothesized"]


class FailureRelation(str, Enum):
    """The epistemic strength and meaning of one directed failure relation."""

    CAUSAL = "causal"
    CONTRIBUTING = "contributing"
    PRECEDING = "preceding"
    CORRELATED = "correlated"
    HYPOTHESIZED = "hypothesized"


class FailureRole(str, Enum):
    """One verifier-backed semantic role in a failure localization."""

    FIRST_ANOMALY = "first_anomaly"
    ROOT_CAUSE = "root_cause"
    DECISIVE_FAILURE = "decisive_failure"
    SYMPTOM = "symptom"
    RECOVERABLE_UNTIL = "recoverable_until"


@dataclass(frozen=True, slots=True)
class FailureNode:
    """One observed failure-stage event with canonical observable evidence."""

    failure_id: str
    event_id: str
    summary: str
    observed_at: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        """Validate identity, observation time, and detached evidence references."""

        _require_nonblank_string(self.failure_id, "failure_id")
        _require_nonblank_string(self.event_id, "event_id")
        _require_nonblank_string(self.summary, "summary")
        _require_rfc3339(self.observed_at, "observed_at")
        references = _snapshot_ordered_evidence(self.evidence_refs)
        if not references or not any(reference.is_observable for reference in references):
            raise ValueError("FailureNode requires at least one observable evidence reference")
        object.__setattr__(self, "evidence_refs", references)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible failure-node snapshot."""

        snapshot = _snapshot_node(self)
        return {
            "failure_id": snapshot.failure_id,
            "event_id": snapshot.event_id,
            "summary": snapshot.summary,
            "observed_at": snapshot.observed_at,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> FailureNode:
        """Restore a failure node from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "failure_id",
                "event_id",
                "summary",
                "observed_at",
                "evidence_refs",
            },
            "FailureNode",
        )
        return cls(
            failure_id=cast(str, values["failure_id"]),
            event_id=cast(str, values["event_id"]),
            summary=cast(str, values["summary"]),
            observed_at=cast(str, values["observed_at"]),
            evidence_refs=_restore_evidence_refs(values["evidence_refs"]),
        )


@dataclass(frozen=True, slots=True)
class FailureEdge:
    """One explicit relation between two failure nodes."""

    source_id: str
    target_id: str
    relation: FailureRelation
    verifier_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require exact endpoints and relation-appropriate verifier provenance."""

        _require_nonblank_string(self.source_id, "source_id")
        _require_nonblank_string(self.target_id, "target_id")
        if self.source_id == self.target_id:
            raise ValueError("a failure edge cannot relate a node to itself")
        if type(self.relation) is not FailureRelation:
            raise ValueError("relation must be an exact FailureRelation")
        references = _snapshot_verifier_refs(self.verifier_refs)
        if self.relation is FailureRelation.CAUSAL and not references:
            raise ValueError("causal edges require nonempty verifier_refs")
        if self.relation is FailureRelation.HYPOTHESIZED and references:
            raise ValueError("hypothesized edges cannot carry verifier_refs")
        object.__setattr__(self, "verifier_refs", references)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible failure-edge snapshot."""

        snapshot = _snapshot_edge(self)
        return {
            "source_id": snapshot.source_id,
            "target_id": snapshot.target_id,
            "relation": snapshot.relation.value,
            "verifier_refs": list(snapshot.verifier_refs),
        }

    @classmethod
    def from_dict(cls, data: object) -> FailureEdge:
        """Restore a failure edge from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {"source_id", "target_id", "relation", "verifier_refs"},
            "FailureEdge",
        )
        raw_relation = values["relation"]
        if type(raw_relation) is not str:
            raise ValueError("FailureEdge relation must be a built-in string")
        try:
            relation = FailureRelation(raw_relation)
        except ValueError as exc:
            raise ValueError(f"unknown FailureEdge relation {raw_relation!r}") from exc
        return cls(
            source_id=cast(str, values["source_id"]),
            target_id=cast(str, values["target_id"]),
            relation=relation,
            verifier_refs=_restore_verifier_refs(values["verifier_refs"]),
        )


@dataclass(frozen=True, slots=True)
class FailureRoleEvidence:
    """Verifier evidence supporting one asserted localization role."""

    role: FailureRole
    failure_id: str
    verifier_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.role) is not FailureRole:
            raise ValueError("role must be an exact FailureRole")
        _require_nonblank_string(self.failure_id, "failure_id")
        references = _snapshot_verifier_refs(self.verifier_refs)
        if not references:
            raise ValueError("FailureRoleEvidence requires verifier_refs")
        object.__setattr__(self, "verifier_refs", references)

    def to_dict(self) -> dict[str, object]:
        snapshot = _snapshot_role_evidence(self)
        return {
            "role": snapshot.role.value,
            "failure_id": snapshot.failure_id,
            "verifier_refs": list(snapshot.verifier_refs),
        }

    @classmethod
    def from_dict(cls, data: object) -> FailureRoleEvidence:
        values = _require_exact_mapping(
            data, {"role", "failure_id", "verifier_refs"}, "FailureRoleEvidence"
        )
        raw_role = values["role"]
        if type(raw_role) is not str:
            raise ValueError("FailureRoleEvidence role must be a built-in string")
        try:
            role = FailureRole(raw_role)
        except ValueError as exc:
            raise ValueError(f"unknown FailureRoleEvidence role {raw_role!r}") from exc
        return cls(
            role,
            cast(str, values["failure_id"]),
            _restore_verifier_refs(values["verifier_refs"]),
        )


@dataclass(frozen=True, slots=True)
class FailureLocalization:
    """Distinct roles assigned to nodes in one validated failure graph."""

    first_anomaly: str
    root_cause: str | None
    decisive_failure: str | None
    symptoms: tuple[str, ...]
    recoverable_until: str | None
    role_evidence: tuple[FailureRoleEvidence, ...]

    def __post_init__(self) -> None:
        """Validate exact, deterministic localization references."""

        _require_nonblank_string(self.first_anomaly, "first_anomaly")
        _require_optional_nonblank_string(self.root_cause, "root_cause")
        _require_optional_nonblank_string(self.decisive_failure, "decisive_failure")
        symptoms = _snapshot_identifiers(self.symptoms, "symptoms")
        if len(set(symptoms)) != len(symptoms):
            raise ValueError("symptoms must be unique")
        object.__setattr__(self, "symptoms", symptoms)
        _require_optional_nonblank_string(self.recoverable_until, "recoverable_until")
        evidence = _snapshot_role_evidence_items(self.role_evidence)
        _validate_role_evidence(self, evidence)
        object.__setattr__(self, "role_evidence", evidence)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible localization snapshot."""

        snapshot = _snapshot_localization(self)
        return {
            "first_anomaly": snapshot.first_anomaly,
            "root_cause": snapshot.root_cause,
            "decisive_failure": snapshot.decisive_failure,
            "symptoms": list(snapshot.symptoms),
            "recoverable_until": snapshot.recoverable_until,
            "role_evidence": [item.to_dict() for item in snapshot.role_evidence],
        }

    @classmethod
    def from_dict(cls, data: object) -> FailureLocalization:
        """Restore failure localization from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "first_anomaly",
                "root_cause",
                "decisive_failure",
                "symptoms",
                "recoverable_until",
                "role_evidence",
            },
            "FailureLocalization",
        )
        return cls(
            first_anomaly=cast(str, values["first_anomaly"]),
            root_cause=cast(str | None, values["root_cause"]),
            decisive_failure=cast(str | None, values["decisive_failure"]),
            symptoms=_restore_identifiers(values["symptoms"], "symptoms"),
            recoverable_until=cast(str | None, values["recoverable_until"]),
            role_evidence=_restore_role_evidence(values["role_evidence"]),
        )


@dataclass(frozen=True, slots=True)
class FailureGraph:
    """A validated failure graph and its epistemically bounded localization."""

    nodes: tuple[FailureNode, ...]
    edges: tuple[FailureEdge, ...]
    localization: FailureLocalization

    def __post_init__(self) -> None:
        """Detach nested records and validate graph structure and localization support."""

        nodes = _snapshot_nodes(self.nodes)
        edges = _snapshot_edges(self.edges)
        localization = _snapshot_localization(self.localization)
        _validate_graph(nodes, edges, localization)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "localization", localization)

    @property
    def root_cause_status(self) -> RootCauseStatus | None:
        """Return whether the localized root has supported or hypothesized linkage."""

        _, edges, localization = _validated_graph_snapshot(self)
        return _derive_root_cause_status(edges, localization)

    def supported_topological_order(self) -> tuple[str, ...]:
        """Order all nodes using only verifier-supported causal edges."""

        nodes, edges, _ = _validated_graph_snapshot(self)
        return _causal_topological_order(nodes, edges)

    def topological_order(self) -> tuple[str, ...]:
        """Alias the graph's explicitly supported causal ordering."""

        return self.supported_topological_order()

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible graph with explicit root epistemic status."""

        nodes, edges, localization = _validated_graph_snapshot(self)
        return {
            "nodes": [node.to_dict() for node in nodes],
            "edges": [edge.to_dict() for edge in edges],
            "localization": localization.to_dict(),
            "root_cause_status": _derive_root_cause_status(edges, localization),
        }

    @classmethod
    def from_dict(cls, data: object) -> FailureGraph:
        """Restore an exact graph and reject a dishonest serialized root status."""

        values = _require_exact_mapping(
            data,
            {"nodes", "edges", "localization", "root_cause_status"},
            "FailureGraph",
        )
        graph = cls(
            nodes=_restore_nodes(values["nodes"]),
            edges=_restore_edges(values["edges"]),
            localization=FailureLocalization.from_dict(values["localization"]),
        )
        serialized_status = values["root_cause_status"]
        if serialized_status is not None and type(serialized_status) is not str:
            raise ValueError("FailureGraph root_cause_status must be a built-in string or null")
        if serialized_status != graph.root_cause_status:
            raise ValueError("FailureGraph root_cause_status does not match its explicit edges")
        return graph


def _snapshot_ordered_evidence(values: object) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError("evidence_refs must be an ordered array")
    return _snapshot_evidence_refs(values)


def _restore_evidence_refs(values: object) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError("FailureNode evidence_refs must be an array")
    return tuple(EvidenceReference.from_dict(value) for value in values)


def _snapshot_verifier_refs(values: object) -> tuple[str, ...]:
    references = _snapshot_identifiers(values, "verifier_refs")
    if len(set(references)) != len(references):
        raise ValueError("verifier_refs must be unique")
    return references


def _restore_verifier_refs(values: object) -> tuple[str, ...]:
    return _snapshot_identifiers(values, "verifier_refs")


def _snapshot_role_evidence(value: object) -> FailureRoleEvidence:
    if type(value) is not FailureRoleEvidence:
        raise ValueError("role_evidence must contain exact FailureRoleEvidence values")
    try:
        return FailureRoleEvidence(value.role, value.failure_id, value.verifier_refs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"role_evidence contains an invalid record: {exc}") from exc


def _snapshot_role_evidence_items(value: object) -> tuple[FailureRoleEvidence, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("role_evidence must be an ordered array")
    return tuple(_snapshot_role_evidence(item) for item in value)


def _restore_role_evidence(value: object) -> tuple[FailureRoleEvidence, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("FailureLocalization role_evidence must be an array")
    return tuple(FailureRoleEvidence.from_dict(item) for item in value)


def _validate_role_evidence(
    localization: FailureLocalization,
    evidence: tuple[FailureRoleEvidence, ...],
) -> None:
    expected = {(FailureRole.FIRST_ANOMALY, localization.first_anomaly)}
    for role, failure_id in (
        (FailureRole.ROOT_CAUSE, localization.root_cause),
        (FailureRole.DECISIVE_FAILURE, localization.decisive_failure),
        (FailureRole.RECOVERABLE_UNTIL, localization.recoverable_until),
    ):
        if failure_id is not None:
            expected.add((role, failure_id))
    expected.update((FailureRole.SYMPTOM, symptom) for symptom in localization.symptoms)
    actual = {(item.role, item.failure_id) for item in evidence}
    if len(actual) != len(evidence) or actual != expected:
        raise ValueError("localization role evidence must exactly bind every asserted role")


def _snapshot_identifiers(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError(f"{field_name} must be an ordered array")
    identifiers: list[str] = []
    for value in values:
        identifiers.append(_require_nonblank_string(value, field_name))
    return tuple(identifiers)


def _restore_identifiers(values: object, field_name: str) -> tuple[str, ...]:
    return _snapshot_identifiers(values, field_name)


def _require_optional_nonblank_string(value: object, field_name: str) -> None:
    if value is not None:
        _require_nonblank_string(value, field_name)


def _snapshot_node(node: object) -> FailureNode:
    if type(node) is not FailureNode:
        raise ValueError("nodes must contain exact FailureNode values")
    try:
        return FailureNode(
            failure_id=node.failure_id,
            event_id=node.event_id,
            summary=node.summary,
            observed_at=node.observed_at,
            evidence_refs=node.evidence_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"nodes contains an invalid FailureNode: {exc}") from exc


def _snapshot_edge(edge: object) -> FailureEdge:
    if type(edge) is not FailureEdge:
        raise ValueError("edges must contain exact FailureEdge values")
    try:
        return FailureEdge(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation,
            verifier_refs=edge.verifier_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"edges contains an invalid FailureEdge: {exc}") from exc


def _snapshot_localization(localization: object) -> FailureLocalization:
    if type(localization) is not FailureLocalization:
        raise ValueError("localization must be an exact FailureLocalization")
    try:
        return FailureLocalization(
            first_anomaly=localization.first_anomaly,
            root_cause=localization.root_cause,
            decisive_failure=localization.decisive_failure,
            symptoms=localization.symptoms,
            recoverable_until=localization.recoverable_until,
            role_evidence=localization.role_evidence,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"localization is invalid: {exc}") from exc


def _snapshot_nodes(values: object) -> tuple[FailureNode, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError("nodes must be an ordered array")
    return tuple(_snapshot_node(value) for value in values)


def _snapshot_edges(values: object) -> tuple[FailureEdge, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError("edges must be an ordered array")
    return tuple(_snapshot_edge(value) for value in values)


def _restore_nodes(values: object) -> tuple[FailureNode, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError("FailureGraph nodes must be an array")
    return tuple(FailureNode.from_dict(value) for value in values)


def _restore_edges(values: object) -> tuple[FailureEdge, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError("FailureGraph edges must be an array")
    return tuple(FailureEdge.from_dict(value) for value in values)


def _validated_graph_snapshot(
    graph: object,
) -> tuple[tuple[FailureNode, ...], tuple[FailureEdge, ...], FailureLocalization]:
    if type(graph) is not FailureGraph:
        raise ValueError("graph must be an exact FailureGraph")
    nodes = _snapshot_nodes(graph.nodes)
    edges = _snapshot_edges(graph.edges)
    localization = _snapshot_localization(graph.localization)
    _validate_graph(nodes, edges, localization)
    return nodes, edges, localization


def _validate_graph(
    nodes: tuple[FailureNode, ...],
    edges: tuple[FailureEdge, ...],
    localization: FailureLocalization,
) -> None:
    node_ids = {node.failure_id for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("FailureGraph failure IDs must be unique")
    event_ids = {node.event_id for node in nodes}
    if len(event_ids) != len(nodes):
        raise ValueError("FailureGraph event IDs must be unique")
    edge_keys = {(edge.source_id, edge.target_id, edge.relation) for edge in edges}
    if len(edge_keys) != len(edges):
        raise ValueError("FailureGraph edges must be unique")
    for edge in edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            raise ValueError("FailureGraph has a dangling edge endpoint")
    localization_refs = (
        localization.first_anomaly,
        localization.root_cause,
        localization.decisive_failure,
        *localization.symptoms,
        localization.recoverable_until,
    )
    if any(value is not None and value not in node_ids for value in localization_refs):
        raise ValueError("FailureGraph has a dangling localization reference")
    _causal_topological_order(nodes, edges)
    _derive_root_cause_status(edges, localization)
    _validate_localization_paths(edges, localization)


def _validate_localization_paths(
    edges: tuple[FailureEdge, ...],
    localization: FailureLocalization,
) -> None:
    root = localization.root_cause
    anomaly = localization.first_anomaly
    decisive = localization.decisive_failure
    if root is not None and not _has_semantic_path(root, anomaly, edges):
        raise ValueError("FailureGraph first_anomaly is not downstream of root_cause")
    if decisive is not None and anomaly != decisive:
        if not _has_semantic_path(anomaly, decisive, edges):
            raise ValueError("FailureGraph decisive_failure is not downstream of first_anomaly")
    if decisive is None and localization.symptoms:
        raise ValueError("FailureGraph symptoms require a decisive_failure")
    contradictory = {value for value in (root, anomaly, decisive) if value is not None}
    if any(symptom in contradictory for symptom in localization.symptoms):
        raise ValueError("FailureGraph cannot assign one node to contradictory roles")
    for symptom in localization.symptoms:
        if decisive is None or not _has_semantic_path(
            decisive, symptom, edges, include_contributing=True
        ):
            raise ValueError("FailureGraph symptom is not downstream of decisive_failure")
    recoverable = localization.recoverable_until
    if recoverable is not None and decisive is not None:
        if not _has_semantic_path(anomaly, recoverable, edges, allow_zero=True):
            raise ValueError("FailureGraph recoverable_until precedes first_anomaly")
        if not _has_semantic_path(recoverable, decisive, edges, allow_zero=True):
            raise ValueError("FailureGraph recoverable_until is outside the decisive path")


def _has_semantic_path(
    source_id: str,
    target_id: str,
    edges: tuple[FailureEdge, ...],
    *,
    allow_zero: bool = False,
    include_contributing: bool = False,
) -> bool:
    if source_id == target_id:
        return allow_zero
    allowed = {FailureRelation.CAUSAL, FailureRelation.HYPOTHESIZED}
    if include_contributing:
        allowed.add(FailureRelation.CONTRIBUTING)
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if edge.relation in allowed:
            adjacency.setdefault(edge.source_id, []).append(edge.target_id)
    pending = [source_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for target in adjacency.get(current, []):
            if target == target_id:
                return True
            pending.append(target)
    return False


def _causal_topological_order(
    nodes: tuple[FailureNode, ...],
    edges: tuple[FailureEdge, ...],
) -> tuple[str, ...]:
    adjacency: dict[str, set[str]] = {node.failure_id: set() for node in nodes}
    indegree = {node.failure_id: 0 for node in nodes}
    for edge in edges:
        if edge.relation is not FailureRelation.CAUSAL:
            continue
        adjacency[edge.source_id].add(edge.target_id)
        indegree[edge.target_id] += 1
    available = [node_id for node_id, count in indegree.items() if count == 0]
    heapq.heapify(available)
    ordered: list[str] = []
    while available:
        source_id = heapq.heappop(available)
        ordered.append(source_id)
        for target_id in sorted(adjacency[source_id]):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                heapq.heappush(available, target_id)
    if len(ordered) != len(nodes):
        raise ValueError("FailureGraph causal edges contain a cycle")
    return tuple(ordered)


def _derive_root_cause_status(
    edges: tuple[FailureEdge, ...],
    localization: FailureLocalization,
) -> RootCauseStatus | None:
    root_cause = localization.root_cause
    if root_cause is None:
        return None
    target = localization.decisive_failure or localization.first_anomaly
    if _has_relation_path(root_cause, target, edges, require_hypothesis=False):
        return "supported"
    if _has_relation_path(root_cause, target, edges, require_hypothesis=True):
        return "hypothesized"
    raise ValueError("FailureGraph has an unsupported root_cause label")


def _has_relation_path(
    source_id: str,
    target_id: str,
    edges: tuple[FailureEdge, ...],
    *,
    require_hypothesis: bool,
) -> bool:
    if source_id == target_id:
        return False
    allowed = {FailureRelation.CAUSAL}
    if require_hypothesis:
        allowed.add(FailureRelation.HYPOTHESIZED)
    adjacency: dict[str, list[tuple[str, bool]]] = {}
    for edge in edges:
        if edge.relation in allowed:
            adjacency.setdefault(edge.source_id, []).append(
                (edge.target_id, edge.relation is FailureRelation.HYPOTHESIZED)
            )
    pending = [(source_id, False)]
    visited: set[tuple[str, bool]] = set()
    while pending:
        node_id, used_hypothesis = pending.pop()
        state = (node_id, used_hypothesis)
        if state in visited:
            continue
        visited.add(state)
        if node_id == target_id and (used_hypothesis or not require_hypothesis):
            return True
        for next_id, is_hypothesis in adjacency.get(node_id, []):
            pending.append((next_id, used_hypothesis or is_hypothesis))
    return False


__all__ = [
    "FailureEdge",
    "FailureGraph",
    "FailureLocalization",
    "FailureNode",
    "FailureRelation",
    "FailureRole",
    "FailureRoleEvidence",
    "RootCauseStatus",
]
