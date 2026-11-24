"""Optional CPU controls with real policy-gradient updates on a synthetic action task."""

from dataclasses import replace

import pytest

pytest.importorskip("torch")

from evaluation.mdac_comparison_fixture import fixture_inputs
from evaluation.mdac_comparisons import TrainingRequest, compare
from evaluation.mdac_report_adapters import comparison_runs
from gepa_mindfulness.examples.mdac_rl_comparison import make_study


def test_six_arm_cpu_model_comparison_resets_and_matches_budgets():
    inputs = make_study()
    table = fixture_inputs()
    assert inputs["manifest"].digest != table["manifest"].digest
    assert inputs["study"].evaluator_version != table["study"].evaluator_version
    assert (
        inputs["lessons"][0].criterion.record.evaluator_version == inputs["study"].evaluator_version
    )
    inputs["study"] = replace(inputs["study"], seeds=(7,))
    report = compare(**inputs, enabled=True)
    assert len(report["cells"]) == 12
    starts = {c["checkpoint_start"] for c in report["cells"]}
    assert starts == {inputs["study"].checkpoint}
    for cell in report["cells"]:
        assert cell["checkpoint_end"] not in starts
        assert cell["actual"]["training_tokens"] == 4 * cell["actual"]["samples"] == 48
        assert cell["actual"]["compute_units"] == 12
        assert cell["actual"]["evaluation_tokens"] == 5
        assert cell["metrics"]["diagnostic_precision"]["denominator"] == 0
        assert cell["metrics"]["unknown_rate"]["value"] is None
        assert cell["metrics"]["mechanism_profile"]["M01"]["predicted"] is None
    assert all(
        rate["value"] is None
        for arm in report["diagnostic_stability"].values()
        for rate in arm.values()
    )
    rows = comparison_runs(report, inputs["manifest"])
    assert {r["kind"] for r in rows} == {"illustrative"}
    assert {r["rl_method"] for r in rows} >= {"ppo", "grpo"}
    assert compare(**inputs, enabled=True)["result_digest"] == report["result_digest"]


def test_trainer_accepts_only_registered_training_prompts_and_initial_state():
    inputs = make_study()
    study = inputs["study"]
    request = TrainingRequest(
        "ppo",
        7,
        study.checkpoint,
        study.conditions,
        study.governance_version,
        (inputs["cases"][0].prompt,),
        1,
        4,
        study.model_config_digest,
        "{}",
    )
    with pytest.raises(ValueError, match="training prompt"):
        inputs["train"](request)
    with pytest.raises(ValueError, match="checkpoint"):
        inputs["train"](replace(request, checkpoint="other"))
