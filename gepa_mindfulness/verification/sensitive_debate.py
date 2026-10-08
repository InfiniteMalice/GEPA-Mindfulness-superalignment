"""Empirical public premise ablations inspired by Sensitive Debate, arXiv:2610.02557.

These proxies do not implement fractional block sensitivity or inherit its guarantees.
The host supplies public decompositions and perturbation outcomes; no model is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.evidence import EvidenceReference
from .claim_graph import ClaimGraph
from .diagnostic_records import (
    DiagnosticRecord,
    _text,
    _unit,
    public_refs,
    records,
    restore_refs,
    strings,
)


@dataclass(frozen=True, slots=True)
class PremiseAblation(DiagnosticRecord):
    """One host-observed perturbation of a premise set and its public decision."""

    premise_ids: tuple[str, ...]
    original_decision: str
    perturbed_decision: str
    evidence_refs: tuple[EvidenceReference, ...]

    schema_version = "premise-ablation-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "premise_ids", strings(self.premise_ids, "premise_ids", required=True)
        )
        _text(self.original_decision, "original_decision")
        _text(self.perturbed_decision, "perturbed_decision")
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))


def decision_sensitivity(
    graph: ClaimGraph, ablations: tuple[PremiseAblation, ...]
) -> dict[str, float | None]:
    """Fraction of observed ablations changing the decision; untested premises stay unknown.

    Multi-premise ablations attribute set-level sensitivity to each participant, without claiming
    an individual causal effect. The original records retain the complete perturbed set.
    """
    graph = ClaimGraph.from_dict(graph.to_dict())
    observations: dict[str, list[bool]] = {node.claim.claim_id: [] for node in graph.nodes}
    for ablation in records(ablations, PremiseAblation):
        for premise in ablation.premise_ids:
            if premise not in observations:
                raise ValueError("ablation references unknown premise")
            observations[premise].append(ablation.original_decision != ablation.perturbed_decision)
    return {
        key: sum(values) / len(values) if values else None for key, values in observations.items()
    }


def select_challenges(
    graph: ClaimGraph,
    conclusion_id: str,
    sensitivity: dict[str, float | None],
    *,
    max_checks: int = 4,
    max_depth: int = 3,
    minimum_sensitivity: float = 0.0,
    enabled: bool = False,
) -> tuple[str, ...]:
    """Traverse all descendants within depth; select only unresolved high-value claims."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return ()
    graph = ClaimGraph.from_dict(graph.to_dict())
    nodes = {node.claim.claim_id: node for node in graph.nodes}
    if conclusion_id not in nodes:
        raise ValueError("unknown conclusion")
    for budget in (max_checks, max_depth):
        if type(budget) is not int or not 0 <= budget <= 1024:
            raise ValueError("budgets must be integers in [0, 1024]")
    _unit(minimum_sensitivity, "minimum_sensitivity")
    for key, value in sensitivity.items():
        if key not in nodes:
            raise ValueError("sensitivity references unknown claim")
        if value is not None:
            _unit(value, "decision_sensitivity")
    children: dict[str, list[str]] = {key: [] for key in nodes}
    for edge in graph.dependencies:
        children[edge.parent_claim_id].append(edge.child_claim_id)
    pending = [(conclusion_id, 0)]
    visited = {conclusion_id}
    candidates = []
    while pending:
        parent, depth = pending.pop(0)
        if depth >= max_depth:
            continue
        for child in sorted(children[parent]):
            if child in visited:
                continue
            visited.add(child)
            pending.append((child, depth + 1))
            score = sensitivity.get(child)
            if child not in graph.unresolved_claim_ids or score is None:
                continue
            if score <= minimum_sensitivity:
                continue
            candidates.append((score * nodes[child].decision_importance, child))
    return tuple(key for _, key in sorted(candidates, key=lambda x: (-x[0], x[1]))[:max_checks])
