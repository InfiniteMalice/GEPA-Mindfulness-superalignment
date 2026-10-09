"""Role permutations with explicit rights, duties and authority controls."""

# Standard library
from __future__ import annotations

from typing import Any

# Third-party
# Local
from evaluation.causal_records import CausalPair, FactorChange, PromptTurn, canonical_json
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.verification.debate_records import _record
from semantic_intent_robustness.perspective_protocol import (
    PerspectiveSource,
    RoleAssignment,
    _items,
)
from synthetic_data.causal_interventions import _enabled, make_causal_pair, variant_from_cell


def make_role_reversal_pair(
    source: PerspectiveSource,
    *,
    after_roles: tuple[RoleAssignment, ...],
    actor_mapping: tuple[tuple[str, str], ...],
    identity_only: bool,
    pair_id: str,
    family_id: str,
    before_cell: V5EvaluationCell,
    after_cell: V5EvaluationCell,
    before_expected_actions: tuple[str, ...],
    after_expected_actions: tuple[str, ...],
    enabled: bool = False,
) -> CausalPair:
    """Render declared role assignments without rewriting or certifying natural language."""
    _enabled(enabled)
    source = _record(source, PerspectiveSource)
    after_roles = _items(after_roles, RoleAssignment)
    actors = {r.actor_id for r in source.roles}
    if type(identity_only) is not bool:
        raise ValueError("identity_only requires a boolean")
    if type(actor_mapping) is not tuple or any(
        type(row) is not tuple or len(row) != 2 or any(type(x) is not str for x in row)
        for row in actor_mapping
    ):
        raise ValueError("actor mapping requires exact pairs")
    mapping = dict(actor_mapping)
    if (
        len(mapping) != len(actor_mapping)
        or set(mapping) != actors
        or set(mapping.values()) != actors
        or all(k == v for k, v in mapping.items())
        or len(after_roles) != len(actors)
        or {r.actor_id for r in after_roles} != actors
    ):
        raise ValueError("requires a nonidentity bijection over the exact actor roster")
    before_by = {r.actor_id: r for r in source.roles}
    after_by = {r.actor_id: r for r in after_roles}
    before_fields: dict[str, Any] = {"actor_mapping": {a: a for a in sorted(actors)}}
    after_fields: dict[str, Any] = {"actor_mapping": dict(sorted(mapping.items()))}
    for actor in sorted(actors):
        for field in ("role_id", "rights", "duties", "authority"):
            key = f"role:{actor}:{field}"
            before_fields[key] = getattr(before_by[actor], field)
            after_fields[key] = getattr(after_by[mapping[actor]], field)
            if identity_only and before_fields[key] != after_fields[key]:
                raise ValueError("identity-only permutation changed role properties")
    # The mapping gives the actual actor for each stable role slot. All public changes
    # are represented in the same factor table used to construct FactorChange records.
    shared = PromptTurn(
        "user",
        canonical_json(
            dict(
                source_text=source.source_text,
                facts=[dict(claim_id=c.claim_id, proposition=c.proposition) for c in source.facts],
                constraints=[
                    dict(claim_id=c.claim_id, proposition=c.proposition) for c in source.constraints
                ],
            )
        ),
    )
    variants = tuple(
        variant_from_cell(
            f"{pair_id}:{arm}",
            cell,
            turns=(shared, PromptTurn("user", canonical_json(values))),
            factors=tuple((k, canonical_json(v)) for k, v in sorted(values.items())),
            expected_actions=expected,
        )
        for arm, cell, values, expected in (
            ("before", before_cell, before_fields, before_expected_actions),
            ("after", after_cell, after_fields, after_expected_actions),
        )
    )
    before, after = variants
    changes = tuple(
        FactorChange(k, v, dict(after.factors)[k])
        for k, v in before.factors
        if v != dict(after.factors)[k]
    )
    if before.turns == after.turns:
        raise ValueError("role reversal cannot be a rendered no-op")
    return make_causal_pair(
        pair_id=pair_id,
        family_id=family_id,
        before=before,
        after=after,
        intervention_kind="single_variable" if len(changes) == 1 else "compound",
        changes=changes,
        claimed_equivalence=identity_only,
        seed_policy="shared" if before.system.seed == after.system.seed else "per_arm",
        source_refs=source.source_refs,
        training_eligibility=source.source_training_eligibility,
        enabled=True,
    )
