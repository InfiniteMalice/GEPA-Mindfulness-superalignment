"""Synthetic monitoring results cannot masquerade as model outcome improvements."""

from dataclasses import replace

import pytest

from evaluation.metacognitive import CONDITIONS, evaluate_metacognitive
from evaluation.metacognitive_fixtures import CHALLENGES, make_challenge
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.verification.metacognitive_peo import monitor_recorded_episode


def cells():
    return plan_v5_cells(
        case_ids=[1],
        stripe_ids=["NONE"],
        repeats=1,
        model_version="synthetic-model-v1",
        harness_version="synthetic-host-v1",
    )


def test_disabled_does_not_inspect_inputs():
    assert evaluate_metacognitive(object()) is None
    with pytest.raises(ValueError):
        evaluate_metacognitive(object(), enabled=1)


@pytest.mark.parametrize("name", CHALLENGES)
def test_each_challenge_has_independent_monitoring_and_strategy_expectations(name):
    fixture = make_challenge(name, cells()[0])
    before = fixture.episode.inputs.sources.to_dict()
    result = evaluate_metacognitive((fixture,), enabled=True)
    rows = result.to_dict()["rows"]
    assert [r["condition"] for r in rows] == list(CONDITIONS)
    assert rows[0]["assessment"] is None and rows[0]["proposal"] is None
    assert rows[1]["assessment"] is not None and rows[1]["proposal"] is None
    assert rows[2]["monitoring_match"] is True
    assert rows[2]["strategy_match"] is True
    assert rows[3]["execution_status"] == "SIMULATION_ONLY"
    assert all(r["final_behavior_status"] == "NOT_RUN" for r in rows)
    assert fixture.episode.inputs.sources.to_dict() == before


def test_healthy_negative_control_does_not_propose_intervention():
    fixture = make_challenge("healthy", cells()[0])
    row = evaluate_metacognitive((fixture,), enabled=True).to_dict()["rows"][2]
    assert row["assessment"]["findings"] == []
    assert row["proposal"]["kind"] == "continue"
    assert row["strategy_match"] is True


def test_all_eight_ablations_run_matched_inputs_and_disclose_unknown_metrics():
    result = evaluate_metacognitive(
        (make_challenge("shared_unsupported", cells()[0]),),
        enabled=True,
    ).to_dict()
    assert len(result["ablations"]) == 8
    assert result["ablations"]["shared_dependency"][0]["without"]["shared_findings"] == 0
    assert result["ablations"]["corrupted_input"][0]["rejected"] is True
    assert result["metrics"]["verified_task_success"]["value"] is None
    assert result["metrics"]["verified_task_success"]["reason"]
    assert result["metrics"]["tokens"]["value"] is None
    assert "alignment_score" not in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("acceptable_proposals", ("contine",)),
        ("expected_findings", ("shared_dependncy",)),
    ],
)
def test_unknown_scoring_vocabulary_is_rejected(field, value):
    fixture = make_challenge("healthy", cells()[0])
    with pytest.raises(ValueError):
        replace(fixture, **{field: value})


def test_baseline_rows_do_not_export_monitor_calibration_or_coverage():
    fixture = make_challenge("excessive_confidence", cells()[0])
    rows = evaluate_metacognitive((fixture,), enabled=True).to_dict()["rows"]
    assert rows[0]["measured_calibration"] is None
    assert rows[0]["unassessed_domains"] is None
    assert rows[1]["measured_calibration"]["brier_score"] == pytest.approx(0.9025)


def test_labels_are_evaluation_only_and_cannot_change_selection():
    fixture = make_challenge("shared_unsupported", cells()[0])
    changed = replace(fixture, expected_findings=(), acceptable_proposals=("continue",))
    a = evaluate_metacognitive((fixture,), enabled=True).to_dict()["rows"][2]
    b = evaluate_metacognitive((changed,), enabled=True).to_dict()["rows"][2]
    assert a["assessment"] == b["assessment"] and a["proposal"] == b["proposal"]
    assert a["monitoring_match"] is True and b["monitoring_match"] is False


def test_case_stripe_repeat_coverage_is_canonical_and_deterministic():
    plan = plan_v5_cells(
        repeats=2,
        model_version="synthetic-model-v1",
        harness_version="synthetic-host-v1",
    )
    assert len(plan) == 17 * 11 * 2
    for cell in plan:
        a = make_challenge("shared_unsupported", cell)
        b = make_challenge("shared_unsupported", cell)
        assert a.episode == b.episode
        assert a.episode.inputs.sources.cell == cell
        assert a.episode.inputs.sources.context.repeat_id == cell.repeat_id
    a = monitor_recorded_episode(
        make_challenge("shared_unsupported", plan[0]).episode, enabled=True
    )
    b = monitor_recorded_episode(
        make_challenge("shared_unsupported", plan[-1]).episode, enabled=True
    )
    assert [(f.kind, f.next_step) for f in a.assessment.findings] == [
        (f.kind, f.next_step) for f in b.assessment.findings
    ]


def test_duplicate_units_and_mixed_budgets_are_rejected():
    fixture = make_challenge("healthy", cells()[0])
    with pytest.raises(ValueError):
        evaluate_metacognitive((fixture, fixture), enabled=True)
    with pytest.raises(ValueError):
        evaluate_metacognitive((fixture,), enabled=True, remaining_steps=True)


def test_held_out_labels_are_retained_but_never_used_to_tune_policy():
    fixture = make_challenge("healthy", cells()[0], split="HELD_OUT")
    wire = evaluate_metacognitive((fixture,), enabled=True).to_dict()
    assert wire["rows"][2]["split"] == "HELD_OUT"
    assert wire["training_eligibility"] == "REGRESSION"
    assert wire["metrics"]["generalization"]["value"] is None


def test_every_challenge_rebinds_complex_contracts_to_later_repeats():
    plan = plan_v5_cells(
        case_ids=[17],
        stripe_ids=["NONE"],
        repeats=2,
        model_version="synthetic-model-v1",
        harness_version="synthetic-host-v1",
    )
    fixtures = tuple(make_challenge(name, plan[-1]) for name in CHALLENGES)
    result = evaluate_metacognitive(fixtures, enabled=True).to_dict()
    assert all(r["monitoring_match"] for r in result["rows"] if r["condition"] == "C")
    assert len(result["ablations"]) == 8
    assert all(len(v) == 10 for v in result["ablations"].values())


def test_continuity_and_independent_label_ablations_have_real_effects():
    fixtures = tuple(
        make_challenge(name, cells()[0])
        for name in (
            "motivated_forgetting",
            "excessive_confidence",
        )
    )
    result = evaluate_metacognitive(fixtures, enabled=True).to_dict()
    continuity = result["ablations"]["evidence_continuity"][0]
    assert continuity["with"]["continuity_findings"] == 1
    assert continuity["without"]["continuity_findings"] == 0
    labels = result["ablations"]["trusted_label_admission"][1]
    assert labels["enabled"]["calibration"]["brier_score"] == pytest.approx(0.9025)
    assert labels["disabled"]["calibration"] is None


def test_exhausted_budget_does_not_fake_strategy_benefit():
    fixture = make_challenge("shared_unsupported", cells()[0])
    wire = evaluate_metacognitive((fixture,), enabled=True, remaining_steps=0).to_dict()
    row = wire["rows"][2]
    assert row["proposal"]["kind"] == "continue" and row["strategy_match"] is None
    assert wire["metrics"]["strategy_selection_quality"]["value"] is None
