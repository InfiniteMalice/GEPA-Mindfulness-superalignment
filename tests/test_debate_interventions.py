"""Controlled public premise edits reuse existing paired causal evaluation."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest

# Local
from test_causal_diagnostics import captures, judgment, run
from test_debate_records import snapshot

from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.training.eligibility import TrainingEligibility
from synthetic_data.debate_interventions import PremiseEdit, make_debate_ablation


def ablation(edits=(PremiseEdit("p", None),), **overrides):
    """Build independently classified fixture arms without generating model responses."""
    cells = plan_v5_cells(
        case_ids=(1, 14),
        stripe_ids=("NONE",),
        repeats=1,
        model_version="fixture",
        harness_version="v1",
    )
    s = snapshot()
    options = dict(
        edits=edits,
        pair_id="ablation",
        family_id="public-record",
        before_cell=cells[0],
        after_cell=cells[1],
        before_expected_actions=("hidden-oracle-before",),
        after_expected_actions=("hidden-oracle-after",),
        source_refs=s.evidence_refs,
        training_eligibility=TrainingEligibility.REGRESSION,
        enabled=True,
    )
    options.update(overrides)
    return make_debate_ablation(s, **options)


def test_single_removal_is_deterministic_and_preserves_other_public_data():
    p = ablation()
    assert p == ablation()
    assert p.intervention_kind == "single_variable"
    assert p.changes[0].factor == "premise:p"
    assert p.changes[0].after == "null"
    before, after = (json.loads(v.turns[1].content) for v in (p.before, p.after))
    assert before["question"] == after["question"]
    assert [x for x in before["premises"] if x["claim_id"] != "p"] == after["premises"]
    assert "hidden-oracle" not in str(p.after.turns)
    assert "confidence" not in str(p.after.turns)
    assert p.training_eligibility is TrainingEligibility.REGRESSION
    assert p.before.case.case_id != p.after.case.case_id
    assert p.seed_policy == "per_arm"


def test_compound_reversal_order_is_canonical():
    edits = (PremiseEdit("q", None), PremiseEdit("p", "The record is invalid"))
    p = ablation(edits)
    assert p == ablation(tuple(reversed(edits)))
    assert p.intervention_kind == "compound"
    assert len(p.changes) == 2
    assert snapshot().graph.nodes[1].claim.proposition == "The record is valid"


@pytest.mark.parametrize(
    "edits",
    [
        (),
        (PremiseEdit("unknown", None),),
        (PremiseEdit("c", None),),
        (PremiseEdit("p", None),) * 2,
        (PremiseEdit("p", "The record is valid"),),
    ],
)
def test_invalid_targets_and_noops_rejected(edits):
    with pytest.raises(ValueError):
        ablation(edits)


def test_disabled_train_blank_and_constraint_edits_rejected():
    with pytest.raises(ValueError):
        ablation(enabled=False)
    with pytest.raises(ValueError):
        ablation(training_eligibility=TrainingEligibility.TRAIN)
    with pytest.raises(ValueError):
        PremiseEdit("p", " ")
    s = replace(snapshot(), constraint_claim_ids=("p",))
    cell = plan_v5_cells(
        case_ids=(1,),
        stripe_ids=("NONE",),
        repeats=1,
        model_version="fixture",
        harness_version="v1",
    )[0]
    with pytest.raises(ValueError):
        make_debate_ablation(
            s,
            edits=(PremiseEdit("p", None),),
            pair_id="p",
            family_id="f",
            before_cell=cell,
            after_cell=cell,
            before_expected_actions=("a",),
            after_expected_actions=("a",),
            source_refs=s.evidence_refs,
            training_eligibility=TrainingEligibility.DEVELOPMENT,
            enabled=True,
        )


def test_alternative_support_survives_ablation():
    p = ablation()
    cs = captures(p)
    j = judgment(p, (p,), cs, required_update=False, update_satisfied=None, action_changed=False)
    report = run((p,), cs, (j,))
    assert report["pairs"][0]["classification"] != "sensitivity_failure"
    assert report["mechanism_recovery_established"] is False


def test_decisive_premise_justifies_update():
    p = ablation((PremiseEdit("p", "The record is invalid"),))
    cs = captures(p)
    j = judgment(p, (p,), cs)
    report = run((p,), cs, (j,))
    assert report["pairs"][0]["classification"] == "correct_sensitivity"
    assert report["pairs"][0]["original_case"] == p.before.case.case_id
    assert report["pairs"][0]["destination_case"] == p.after.case.case_id
