"""Controls for matched curricula and evaluator-only known-fault labels."""

from dataclasses import replace

import pytest

from evaluation.mdac_comparison_fixture import fixture_inputs
from evaluation.mdac_comparisons import ARMS, compare, summarize


def test_six_matched_arms_and_heldout_labels_stay_with_grader():
    fixture = fixture_inputs()
    report = compare(**fixture, enabled=True)
    assert set(cell["arm"] for cell in report["cells"]) == set(ARMS)
    assert all(cell["actual"]["samples"] == 12 for cell in report["cells"])
    assert all(cell["actual"]["compute_units"] == 12 for cell in report["cells"])
    assert len({cell["checkpoint_start"] for cell in report["cells"]}) == 1
    assert len({cell["condition_digest"] for cell in report["cells"]}) == 2
    assert report["evidence_kind"] == "cpu_fixture"
    assert report["optimizer_input"] is False
    assert report == compare(**fixture_inputs(), enabled=True)
    for cell in report["cells"]:
        assert cell["metrics"]["diagnostic_precision"]["denominator"] >= 0
        assert "ood_alignment" in cell["metrics"]
        assert cell["metrics"]["false_deception_attribution"]["denominator"] > 0


def test_disabled_duplicate_and_mismatched_compute_are_rejected():
    fixture = fixture_inputs()
    with pytest.raises(ValueError, match="enabled"):
        compare(**fixture)
    duplicate = fixture["cases"] * 2
    with pytest.raises(ValueError, match="unique"):
        compare(**(fixture | {"cases": duplicate}), enabled=True)
    original = fixture["train"]

    def overbudget(request):
        result = original(request)
        return replace(result, compute_units=result.compute_units + 1)

    with pytest.raises(ValueError, match="budget"):
        compare(**(fixture | {"train": overbudget}), enabled=True)


def test_case_labels_are_unavailable_to_train_or_inference():
    fixture = fixture_inputs()
    prior = fixture["train"]

    def observe(request):
        assert all(type(prompt) is str for prompt in request.prompts)
        assert not any(hasattr(request, name) for name in ("labels", "mechanism", "cases"))
        return prior(request)

    compare(**(fixture | {"train": observe}), enabled=True)


def test_diagnostic_false_positives_and_unknowns_have_real_denominators():
    from evaluation.mdac_comparisons import Judgment, Verdict

    b = Verdict(0, 1)
    a = Verdict(1, 0, ("M15",), 0.9)
    rows = [
        Judgment("case-1", "final_held_out", ("M01",), b, a),
        Judgment("case-2", "ood", (), b, b),
    ]
    metrics = summarize(rows)
    assert metrics["false_deception_attribution"]["value"] == 0.5
    assert metrics["unknown_rate"]["value"] == 0.5
    assert metrics["multilabel_exact"]["value"] == 0.5
    assert metrics["capability_regression"]["numerator"] == 1
    assert metrics["repair_latency"]["value"] is None


def test_frozen_partitions_and_actual_intervention_receipts():
    from evaluation.causal_records import content_digest
    from evaluation.mdac_comparisons import intervention_effect

    fixture = fixture_inputs()
    lesson = fixture["lessons"][0]
    with pytest.raises(ValueError, match="partition"):
        compare(
            **(fixture | {"lessons": (replace(lesson, family=fixture["cases"][0].family),)}),
            enabled=True,
        )
    report = dict(
        schema_version="mdac-interventions-v1",
        status="observed",
        training_eligibility="DEVELOPMENT",
        confers_authority=False,
        effects={"repair": {"drs": 0.5, "matched_trials": 2}},
        effective_repairs=["repair"],
        budget_exhausted=False,
    )
    report["result_digest"] = content_digest(report)
    assert intervention_effect(report, "repair") == (report["result_digest"], 0.5)
    with pytest.raises(ValueError, match="intervention"):
        intervention_effect(report | {"training_eligibility": "HIDDEN_EVAL"}, "repair")
    with pytest.raises(ValueError, match="intervention"):
        intervention_effect(report | {"status": "inconclusive"}, "repair")


def test_runtime_adapters_preserve_seed_checkpoint_and_method():
    from evaluation.mdac_comparisons import TrainingRequest, rl_training_config

    r = TrainingRequest(
        "ppo", 42, "/models/local", "fixed", "policy", ("test",) * 4, 128, 128, "a" * 64
    )
    for method in ("ppo", "grpo"):
        cfg = rl_training_config(
            replace(r, method=method),
            dataset="catalog.jsonl",
            output_dir="runs/arm",
            batch_size=1,
            max_new_tokens=8,
        )
        assert cfg.algorithm.name == method
        assert cfg.seed == 42
        assert cfg.policy.model_name == "/models/local"
    with pytest.raises(ValueError, match="DAPO"):
        rl_training_config(
            replace(r, method="dapo"),
            dataset="catalog.jsonl",
            output_dir="runs/arm",
            batch_size=1,
            max_new_tokens=8,
        )


def test_unbound_causal_weights_cannot_change_the_schedule():
    fixture = fixture_inputs()
    wrong = replace(fixture["lessons"][0], drs=0.25)
    with pytest.raises(ValueError, match="retained intervention"):
        compare(**(fixture | {"lessons": (wrong, *fixture["lessons"][1:])}), enabled=True)


def test_curriculum_effect_is_paired_on_seed_method_and_frozen_cases():
    report = compare(**fixture_inputs(), enabled=True)
    comparisons = report["paired_curriculum_effects"]
    assert len(comparisons) == 10
    assert all(x["reference_arm"] == "A" for x in comparisons)
    assert all(x["alignment_delta"]["denominator"] == 5 for x in comparisons)


@pytest.mark.parametrize("field", ["training_tokens", "model_config_digest"])
def test_common_token_budget_and_model_signature_are_enforced(field):
    fixture = fixture_inputs()
    original = fixture["train"]

    def train(request):
        result = original(request)
        assert getattr(request, field) == getattr(fixture["study"], field)
        return replace(result, **{field: 0 if field == "training_tokens" else "other"})

    with pytest.raises(ValueError, match="budget"):
        compare(**(fixture | {"train": train}), enabled=True)


def test_evaluation_generation_respects_the_frozen_limit():
    from evaluation.mdac_comparisons import Completion

    fixture = fixture_inputs()
    with pytest.raises(ValueError, match="evaluation token"):
        compare(
            **(fixture | {"baseline": lambda prompt, seed, limit: Completion("answer", limit + 1)}),
            enabled=True,
        )
