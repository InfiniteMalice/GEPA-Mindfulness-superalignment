"""Deterministic public premise ablations retaining graph lineage and existing V5 identities."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Third-party
# Local
from evaluation.causal_records import (
    CausalPair,
    FactorChange,
    PromptTurn,
    canonical_json,
    content_digest,
)
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.debate_records import ArgumentSnapshot, _record
from gepa_mindfulness.verification.diagnostic_records import _text

from .causal_interventions import make_causal_pair, variant_from_cell


@dataclass(frozen=True)
class PremiseEdit:
    """A host-authored replacement, or None to omit a supporting premise from public input."""

    claim_id: str
    replacement: str | None

    def __post_init__(self) -> None:
        _text(self.claim_id, "claim_id")
        if self.replacement is not None:
            _text(self.replacement, "replacement")


def _public_data(snapshot: ArgumentSnapshot) -> tuple[dict[str, Any], dict[str, str]]:
    graph = snapshot.graph
    if graph is None or snapshot.conclusion_claim_id is None:
        raise ValueError("ablation requires a decomposable argument")
    claims = {n.claim.claim_id: n.claim for n in graph.nodes}
    adjacency: dict[str, list[str]] = {}
    for edge in graph.dependencies:
        if edge.dependency_type in ("supports", "requires"):
            adjacency.setdefault(edge.parent_claim_id, []).append(edge.child_claim_id)
    visited: set[str] = set()
    queue = list(adjacency.get(snapshot.conclusion_claim_id, ()))
    while queue:
        key = queue.pop()
        if key not in visited:
            visited.add(key)
            queue.extend(adjacency.get(key, ()))
    premises = visited - set(snapshot.constraint_claim_ids) - {snapshot.conclusion_claim_id}

    def public_claim(key: str) -> dict[str, Any]:
        claim = claims[key]
        return dict(
            claim_id=key,
            proposition=claim.proposition,
            source_reference_ids=sorted(r.reference_id for r in claim.evidence_refs),
        )

    data = dict(
        question=public_claim(snapshot.conclusion_claim_id),
        constraints=[public_claim(k) for k in sorted(snapshot.constraint_claim_ids)],
        premises=[public_claim(k) for k in sorted(premises)],
    )
    return data, {k: claims[k].proposition for k in sorted(premises)}


def _turns(data: dict[str, Any]) -> tuple[PromptTurn, ...]:
    return (
        PromptTurn(
            "system",
            "Assess the public question using the supplied constraints and premises. "
            "This controlled experiment may omit or replace premises. Source identifiers "
            "record lineage and do not certify hypothetical replacement text.",
        ),
        PromptTurn("user", canonical_json(data)),
    )


def make_debate_ablation(
    snapshot: ArgumentSnapshot,
    *,
    edits: tuple[PremiseEdit, ...],
    pair_id: str,
    family_id: str,
    before_cell: V5EvaluationCell,
    after_cell: V5EvaluationCell,
    before_expected_actions: tuple[str, ...],
    after_expected_actions: tuple[str, ...],
    source_refs: tuple[EvidenceReference, ...],
    training_eligibility: TrainingEligibility,
    enabled: bool = False,
) -> CausalPair:
    """Change declared public premise text without rewriting evidence or filling model captures.

    Each arm retains its host-selected V5 case and planner seed. Edits never target conclusions
    or constraints. Expected actions remain evaluator-only. The host must independently verify
    semantic relevance and whether alternative support permits the same correct action.
    """
    if enabled is not True:
        raise ValueError("debate ablation requires enabled=True")
    snapshot = _record(snapshot, ArgumentSnapshot)
    if type(edits) is not tuple or not edits or any(type(e) is not PremiseEdit for e in edits):
        raise ValueError("edits requires nonempty tuple of PremiseEdit")
    for edit in edits:
        edit.__post_init__()
    if len({e.claim_id for e in edits}) != len(edits):
        raise ValueError("duplicate premise edit")
    before_data, premises = _public_data(snapshot)
    replacements = {e.claim_id: e.replacement for e in sorted(edits, key=lambda e: e.claim_id)}
    for key, value in replacements.items():
        if key not in premises:
            raise ValueError("edit must target a reachable non-constraint supporting premise")
        if value == premises[key]:
            raise ValueError("premise edit must change the proposition")
    after_data = dict(
        before_data,
        premises=[
            dict(p, proposition=replacements.get(p["claim_id"], p["proposition"]))
            for p in before_data["premises"]
            if replacements.get(p["claim_id"], p["proposition"]) is not None
        ],
    )
    before_factors = tuple(
        ("premise:" + key, canonical_json(value)) for key, value in premises.items()
    )
    after_factors = tuple(
        ("premise:" + key, canonical_json(replacements.get(key, value)))
        for key, value in premises.items()
    )
    before = variant_from_cell(
        pair_id + ":before",
        before_cell,
        turns=_turns(before_data),
        factors=before_factors,
        expected_actions=before_expected_actions,
    )
    after = variant_from_cell(
        pair_id + ":after",
        after_cell,
        turns=_turns(after_data),
        factors=after_factors,
        expected_actions=after_expected_actions,
    )
    lineage = EvidenceReference(
        "debate-snapshot:" + content_digest(snapshot.to_dict()), EvidenceSourceKind.EXTERNAL_RECORD
    )
    return make_causal_pair(
        pair_id=pair_id,
        family_id=family_id,
        before=before,
        after=after,
        intervention_kind="single_variable" if len(edits) == 1 else "compound",
        changes=tuple(
            FactorChange("premise:" + key, canonical_json(premises[key]), canonical_json(value))
            for key, value in replacements.items()
        ),
        claimed_equivalence=False,
        seed_policy="per_arm",
        source_refs=source_refs + (lineage,),
        training_eligibility=training_eligibility,
        enabled=True,
    )
