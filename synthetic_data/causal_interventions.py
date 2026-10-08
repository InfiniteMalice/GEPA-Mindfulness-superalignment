"""Controlled public scenarios around existing strict relation-flip fixtures."""

from __future__ import annotations

from evaluation.cases.registry import load_case_manifest
from evaluation.causal_records import (
    CausalPair,
    CausalVariant,
    FactorChange,
    PromptTurn,
    canonical_json,
)
from evaluation.v5_records import CaseIdentity, RobustnessIdentity, SystemIdentity
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility

from .relation_flips import Relation, RelationPair, render_probe


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("causal experiments require enabled=True")


def variant_from_cell(
    variant_id: str,
    cell: V5EvaluationCell,
    *,
    turns: tuple[PromptTurn, ...],
    factors: tuple[tuple[str, str], ...],
    expected_actions: tuple[str, ...],
) -> CausalVariant:
    """Retain exact planner identity; the host must justify the arm's semantic case label."""
    if type(cell) is not V5EvaluationCell:
        raise ValueError("cell must be V5EvaluationCell")
    cell.__post_init__()
    manifest = load_case_manifest()
    case = next(c for c in manifest.cases if c.id == cell.case_id)
    return CausalVariant(
        variant_id,
        CaseIdentity(case.id, cell.case_version, case.key, case.title),
        RobustnessIdentity(cell.stripe_id, cell.subtype),
        SystemIdentity(cell.repeat_id, cell.seed, cell.model_version, cell.harness_version),
        turns,
        factors,
        expected_actions,
    )


def make_causal_pair(
    *,
    pair_id: str,
    family_id: str,
    before: CausalVariant,
    after: CausalVariant,
    intervention_kind: str,
    changes: tuple[FactorChange, ...],
    claimed_equivalence: bool,
    seed_policy: str,
    source_refs: tuple[EvidenceReference, ...],
    training_eligibility: TrainingEligibility,
    enabled: bool = False,
) -> CausalPair:
    """Validate declared structural changes without certifying natural-language semantics."""
    _enabled(enabled)
    record = CausalPair(
        pair_id,
        family_id,
        before,
        after,
        intervention_kind,
        changes,
        claimed_equivalence,
        seed_policy,
        source_refs,
        training_eligibility,
    )
    return CausalPair.from_dict(record.to_dict())


def _relation_value(pair: RelationPair, arm: str) -> str:
    world = getattr(pair, arm)
    if pair.relation is Relation.AUTHORIZATION:
        value = next(p.allowed for p in world.permissions if p.action_id == pair.action_id)
    elif pair.relation is Relation.REVERSIBILITY:
        value = next(a.reversible for a in world.actions if a.action_id == pair.action_id)
    else:
        value = next(f.value for f in world.facts if f.fact_id == pair.relation.value)
    return canonical_json(value)


def adapt_relation_pair(
    pair: RelationPair,
    *,
    before_cell: V5EvaluationCell,
    after_cell: V5EvaluationCell,
    style: str = "plain",
    seed_policy: str = "shared",
    enabled: bool = False,
) -> CausalPair:
    """Preserve the fixture oracle, prompts and admission without weakening its evaluator."""
    _enabled(enabled)
    if type(pair) is not RelationPair:
        raise ValueError("pair must be RelationPair")
    pair = RelationPair.from_dict(pair.to_dict())
    variants = []
    values = [_relation_value(pair, arm) for arm in ("before", "after")]
    for arm, cell, value in zip(("before", "after"), (before_cell, after_cell), values):
        variants.append(
            variant_from_cell(
                f"{pair.pair_id}:{arm}",
                cell,
                turns=(PromptTurn("user", render_probe(pair, arm, style=style, enabled=True)),),
                factors=((pair.relation.value, value),),
                expected_actions=(pair.expected(arm),),
            )
        )
    return make_causal_pair(
        pair_id=pair.pair_id,
        family_id=f"relation:{pair.relation.value}",
        before=variants[0],
        after=variants[1],
        intervention_kind="single_variable",
        changes=(FactorChange(pair.relation.value, values[0], values[1]),),
        claimed_equivalence=not pair.relation.decisive,
        seed_policy=seed_policy,
        source_refs=(
            EvidenceReference(f"relation:{pair.digest}", EvidenceSourceKind.EXTERNAL_RECORD),
        ),
        training_eligibility=pair.before.training_eligibility,
        enabled=True,
    )


def render_causal_variant(
    pair: CausalPair,
    variant_id: str,
    *,
    enabled: bool = False,
) -> tuple[dict[str, str], ...]:
    """Return only actor-visible role/content turns; no oracle or adjudication metadata."""
    _enabled(enabled)
    if type(pair) is not CausalPair:
        raise ValueError("pair must be CausalPair")
    pair = CausalPair.from_dict(pair.to_dict())
    for variant in (pair.before, pair.after):
        if variant.variant_id == variant_id:
            return tuple({"role": turn.role, "content": turn.content} for turn in variant.turns)
    raise ValueError("unknown variant ID")
