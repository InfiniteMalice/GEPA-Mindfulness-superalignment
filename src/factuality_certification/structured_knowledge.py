"""Graph-aware structured knowledge for factuality certification."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

# Local
from .types import AtomicClaim, ClaimSupport, EvidenceItem

RELATION_TYPES = {
    "supports",
    "contradicts",
    "implies",
    "depends_on",
    "part_of",
    "causes",
    "requires",
    "equivalent_to",
    "temporal_predecessor",
    "temporal_successor",
    "parameter_of",
    "component_of",
    "derived_from",
    "correlated_with",
}


@dataclass(frozen=True)
class ClaimNode:
    """Claim graph node with direct support and contradiction scores."""

    claim_id: str
    text: str
    claim_type: str
    direct_support_score: float
    contradiction_score: float
    confidence: float
    requires_current_source: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("direct_support_score", "contradiction_score", "confidence"):
            object.__setattr__(self, field_name, _clamp(getattr(self, field_name)))


@dataclass(frozen=True)
class ClaimRelation:
    """Inspectable relation between two claims."""

    source_claim_id: str
    target_claim_id: str
    relation_type: str
    strength: float
    direction: str = "directed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", _clamp(self.strength))


@dataclass(frozen=True)
class InferencePath:
    """A path that can indirectly reconstruct a target claim."""

    path_id: str
    claim_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    inferred_claim_id: str
    path_confidence: float
    reconstructability_score: float
    explanation_brief: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_ids", tuple(self.claim_ids))
        object.__setattr__(self, "relation_ids", tuple(self.relation_ids))
        object.__setattr__(self, "path_confidence", _clamp(self.path_confidence))
        object.__setattr__(self, "reconstructability_score", _clamp(self.reconstructability_score))


@dataclass(frozen=True, init=False)
class StructuredKnowledgeGraph:
    """Correlated-knowledge graph nesting atomic claims."""

    _nodes: tuple[ClaimNode, ...]
    relations: tuple[ClaimRelation, ...]
    inference_paths: tuple[InferencePath, ...]
    metadata: dict[str, Any]

    def __init__(
        self,
        nodes: tuple[ClaimNode, ...] | Any,
        relations: tuple[ClaimRelation, ...],
        inference_paths: tuple[InferencePath, ...],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if callable(nodes):
            nodes = nodes()
        object.__setattr__(self, "_nodes", tuple(nodes))
        object.__setattr__(self, "relations", tuple(relations))
        object.__setattr__(self, "inference_paths", tuple(inference_paths))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def nodes(self) -> tuple[ClaimNode, ...]:
        return self._nodes

    def edges(self) -> tuple[ClaimRelation, ...]:
        return self.relations

    def paths_to(self, target_id: str, max_depth: int = 3) -> tuple[tuple[str, ...], ...]:
        paths = [
            path.claim_ids
            for path in self.inference_paths
            if path.inferred_claim_id == target_id and len(path.claim_ids) <= max_depth + 1
        ]
        return tuple(paths)


@dataclass(frozen=True)
class StructuredKnowledgeAssessment:
    """Assessment for indirect reconstructability of a target claim."""

    target_claim_id: str
    direct_support_score: float
    indirect_support_score: float
    reconstructability_score: float
    correlated_knowledge_risk: float
    inferable_despite_missing_direct_support: bool
    inference_paths: tuple[InferencePath, ...]
    recommended_action: str
    requires_review: bool

    def __post_init__(self) -> None:
        for field_name in (
            "direct_support_score",
            "indirect_support_score",
            "reconstructability_score",
            "correlated_knowledge_risk",
        ):
            object.__setattr__(self, field_name, _clamp(getattr(self, field_name)))


class StructuredKnowledgeGraphBuilder:
    """Deterministic behavioral graph approximation for atomic claims."""

    def build(
        self,
        *,
        claims: list[AtomicClaim],
        claim_support: list[ClaimSupport],
        evidence: list[EvidenceItem],
        context: str | None = None,
    ) -> StructuredKnowledgeGraph:
        support_by_claim = {support.claim_id: support for support in claim_support}
        nodes = tuple(
            self._node_for_claim(claim, support_by_claim.get(claim.id)) for claim in claims
        )
        relations = self._relations(claims, support_by_claim, evidence)
        return StructuredKnowledgeGraph(
            nodes=nodes,
            relations=relations,
            inference_paths=(),
            metadata={
                "builder": "structured_knowledge_behavioral",
                "context_present": bool(context),
                "evidence_count": len(evidence),
            },
        )

    def _node_for_claim(
        self,
        claim: AtomicClaim,
        support: ClaimSupport | None,
    ) -> ClaimNode:
        direct = support.support_score if support is not None else 0.0
        contradiction = support.contradiction_score if support is not None else 0.0
        confidence = max(direct, 1.0 - contradiction if contradiction else direct)
        return ClaimNode(
            claim_id=claim.id,
            text=claim.text,
            claim_type=claim.claim_type,
            direct_support_score=direct,
            contradiction_score=contradiction,
            confidence=confidence,
            requires_current_source=claim.requires_current_source,
            metadata={"answer_span": claim.answer_span},
        )

    def _relations(
        self,
        claims: list[AtomicClaim],
        support_by_claim: dict[str, ClaimSupport],
        evidence: list[EvidenceItem],
    ) -> tuple[ClaimRelation, ...]:
        relations: list[ClaimRelation] = []
        for left_index, left in enumerate(claims):
            for right in claims[left_index + 1 :]:
                relation = self._relation_from_pair(left, right, support_by_claim)
                if relation is not None:
                    relations.append(relation)
        evidence_claims: dict[str, list[str]] = {}
        for support in support_by_claim.values():
            for evidence_id in support.evidence_ids:
                evidence_claims.setdefault(evidence_id, []).append(support.claim_id)
        for claim_ids in evidence_claims.values():
            for source, target in zip(claim_ids, claim_ids[1:]):
                relations.append(ClaimRelation(source, target, "derived_from", 0.6))
        return tuple(relations)

    def _relation_from_pair(
        self,
        left: AtomicClaim,
        right: AtomicClaim,
        support_by_claim: dict[str, ClaimSupport],
    ) -> ClaimRelation | None:
        left_text = left.text.lower()
        right_text = right.text.lower()
        shared = set(left_text.split()) & set(right_text.split())
        strength = min(0.8, len(shared) / 5)
        if "implies" in left_text or "therefore" in right_text:
            return ClaimRelation(left.id, right.id, "implies", max(0.7, strength))
        if strength >= 0.3:
            return ClaimRelation(left.id, right.id, "correlated_with", strength)
        left_support = support_by_claim.get(left.id)
        right_support = support_by_claim.get(right.id)
        if (
            left_support
            and right_support
            and set(left_support.evidence_ids) & set(right_support.evidence_ids)
        ):
            return ClaimRelation(left.id, right.id, "derived_from", 0.6)
        return None


def score_inference_path(path: InferencePath) -> float:
    """Return the path reconstructability score."""

    return _clamp(path.reconstructability_score)


def estimate_reconstructability(
    *,
    graph: StructuredKnowledgeGraph,
    target_claim_id: str,
    max_depth: int = 3,
) -> StructuredKnowledgeAssessment:
    """Estimate whether a claim is reconstructable via correlated knowledge."""

    nodes = {node.claim_id: node for node in graph.nodes()}
    target = nodes[target_claim_id]
    paths = _paths_to_target(graph, target_claim_id, max_depth)
    scored_paths = tuple(
        _make_inference_path(graph, nodes, path, target_claim_id) for path in paths
    )
    combined = 1.0 - prod(1.0 - path.reconstructability_score for path in scored_paths)
    indirect = combined
    direct = target.direct_support_score
    risk = max(indirect, target.contradiction_score)
    inferable = direct < 0.35 and indirect >= 0.45
    requires_review = inferable or (target.contradiction_score >= 0.6 and indirect >= 0.45)
    action = "manual_review" if requires_review else "answer"
    return StructuredKnowledgeAssessment(
        target_claim_id=target_claim_id,
        direct_support_score=direct,
        indirect_support_score=indirect,
        reconstructability_score=combined,
        correlated_knowledge_risk=risk,
        inferable_despite_missing_direct_support=inferable,
        inference_paths=scored_paths,
        recommended_action=action,
        requires_review=requires_review,
    )


def _paths_to_target(
    graph: StructuredKnowledgeGraph,
    target_claim_id: str,
    max_depth: int,
) -> tuple[tuple[str, ...], ...]:
    inbound = [
        relation for relation in graph.relations if relation.target_claim_id == target_claim_id
    ]
    return tuple(
        (relation.source_claim_id, target_claim_id) for relation in inbound if max_depth >= 1
    )


def _make_inference_path(
    graph: StructuredKnowledgeGraph,
    nodes: dict[str, ClaimNode],
    claim_ids: tuple[str, ...],
    target_claim_id: str,
) -> InferencePath:
    relations = [
        relation
        for relation in graph.relations
        if relation.source_claim_id == claim_ids[0] and relation.target_claim_id == target_claim_id
    ]
    path_nodes = [nodes[claim_id] for claim_id in claim_ids if claim_id != target_claim_id]
    path_confidence = min(node.confidence for node in path_nodes)
    relation_strength = prod(relation.strength for relation in relations) if relations else 0.0
    reconstructability = path_confidence * relation_strength
    relation_ids = tuple(f"{rel.source_claim_id}->{rel.target_claim_id}" for rel in relations)
    return InferencePath(
        path_id="path-" + "-".join(claim_ids),
        claim_ids=claim_ids,
        relation_ids=relation_ids,
        inferred_claim_id=target_claim_id,
        path_confidence=path_confidence,
        reconstructability_score=reconstructability,
        explanation_brief="Correlated claims provide indirect support.",
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "ClaimNode",
    "ClaimRelation",
    "InferencePath",
    "RELATION_TYPES",
    "StructuredKnowledgeAssessment",
    "StructuredKnowledgeGraph",
    "StructuredKnowledgeGraphBuilder",
    "estimate_reconstructability",
    "score_inference_path",
]
