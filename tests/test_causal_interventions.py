"""Controlled adapters preserve legacy prompts, source restrictions and exact changes."""

from dataclasses import replace

import pytest

from evaluation.causal_records import FactorChange, PromptTurn
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility
from synthetic_data.causal_interventions import (
    adapt_relation_pair,
    make_causal_pair,
    render_causal_variant,
    variant_from_cell,
)
from synthetic_data.relation_flips import Relation, make_relation_pair, render_probe


def cells():
    return plan_v5_cells(
        case_ids=(1, 14),
        stripe_ids=("NONE",),
        repeats=1,
        model_version="fixture-model",
        harness_version="fixture-host",
    )


@pytest.mark.parametrize("relation", list(Relation))
@pytest.mark.parametrize("style", ["plain", "reordered", "urgent"])
@pytest.mark.parametrize("reverse", [False, True])
def test_legacy_adapter_preserves_rendered_prompts_and_oracle(relation, style, reverse):
    legacy = make_relation_pair(relation, reverse=reverse, enabled=True)
    before_cell, after_cell = cells()
    p = adapt_relation_pair(
        legacy,
        before_cell=before_cell,
        after_cell=after_cell,
        style=style,
        seed_policy="per_arm",
        enabled=True,
    )
    for arm in ("before", "after"):
        variant = getattr(p, arm)
        assert render_causal_variant(p, variant.variant_id, enabled=True) == (
            {"role": "user", "content": render_probe(legacy, arm, style=style, enabled=True)},
        )
        assert variant.expected_actions == (legacy.expected(arm),)
    assert p.before.case.case_id == 1
    assert p.after.case.case_id == 14
    assert p.after.system.seed == after_cell.seed
    assert p.changes[0].factor == relation.value
    assert legacy.digest in p.source_refs[0].reference_id
    assert p.claimed_equivalence is (not relation.decisive)


@pytest.mark.parametrize(
    "factor,before,after,equivalent",
    [
        ("wording", '"original"', '"paraphrase"', True),
        ("evidence_order", '["a","b"]', '["b","a"]', True),
        ("authorization", "false", "true", False),
        ("authorization", "true", "false", False),
        ("freshness", "true", "false", False),
        ("trust", "false", "true", False),
        ("clarification", "null", '"supplied"', False),
        ("reward_pressure", "false", "true", True),
        ("evaluator_pressure", "false", "true", True),
        ("perspective", '"owner"', '"operator"', True),
    ],
)
def test_deterministic_multi_turn_transformations(factor, before, after, equivalent):
    a, b = cells()
    first = variant_from_cell(
        "before",
        a,
        turns=(PromptTurn("user", "Review the public record."),),
        factors=((factor, before),),
        expected_actions=("answer",),
    )
    second = variant_from_cell(
        "after",
        b,
        turns=(
            PromptTurn("user", "Review the public record."),
            PromptTurn("assistant", "Please clarify the relevant fact."),
            PromptTurn("user", f"Recorded value: {after}"),
        ),
        factors=((factor, after),),
        expected_actions=("resume",),
    )
    kwargs = dict(
        pair_id="sequence",
        family_id="family",
        before=first,
        after=second,
        intervention_kind="single_variable",
        changes=(FactorChange(factor, before, after),),
        claimed_equivalence=equivalent,
        seed_policy="per_arm",
        source_refs=(EvidenceReference("fixture", EvidenceSourceKind.EXTERNAL_RECORD),),
        training_eligibility=TrainingEligibility.DEVELOPMENT,
    )
    p = make_causal_pair(**kwargs, enabled=True)
    assert p.digest == make_causal_pair(**kwargs, enabled=True).digest
    assert p.after.turns[-1].content == f"Recorded value: {after}"
    assert len(p.after.turns) == 3
    with pytest.raises(ValueError):
        make_causal_pair(**kwargs)


def test_renderer_excludes_evaluator_metadata():
    cell = cells()[0]
    p = adapt_relation_pair(
        make_relation_pair(Relation.CONSENT, enabled=True),
        before_cell=cell,
        after_cell=cell,
        enabled=True,
    )
    p = replace(p, before=replace(p.before, expected_actions=("SECRET_ORACLE",)))
    turns = render_causal_variant(p, p.before.variant_id, enabled=True)
    assert all(set(t) == {"role", "content"} for t in turns)
    assert "SECRET_ORACLE" not in str(turns)
    turns[0]["content"] = "mutated export"
    assert p.before.turns[0].content != "mutated export"


def test_default_disabled():
    legacy = make_relation_pair(Relation.CONSENT, enabled=True)
    cell = cells()[0]
    with pytest.raises(ValueError):
        adapt_relation_pair(legacy, before_cell=cell, after_cell=cell)
    p = adapt_relation_pair(legacy, before_cell=cell, after_cell=cell, enabled=True)
    with pytest.raises(ValueError):
        render_causal_variant(p, p.before.variant_id)
    with pytest.raises(ValueError):
        render_causal_variant(p, "not-an-arm", enabled=True)


def test_compound_remains_labeled_compound():
    from test_causal_records import pair

    p = pair()
    compound = replace(
        p,
        before=replace(p.before, factors=p.before.factors + (("fresh", "false"),)),
        after=replace(p.after, factors=p.after.factors + (("fresh", "true"),)),
        changes=p.changes + (FactorChange("fresh", "false", "true"),),
        intervention_kind="compound",
    )
    assert compound.to_dict()["intervention_kind"] == "compound"
    assert len(compound.changes) == 2


def test_hidden_eval_admission_survives_adapter():
    legacy = make_relation_pair(
        Relation.CONSENT,
        eligibility=TrainingEligibility.HIDDEN_EVAL,
        enabled=True,
    )
    cell = cells()[0]
    p = adapt_relation_pair(legacy, before_cell=cell, after_cell=cell, enabled=True)
    assert p.training_eligibility is TrainingEligibility.HIDDEN_EVAL
