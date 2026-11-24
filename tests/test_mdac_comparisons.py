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

    b = Verdict(0, 1, mechanisms=())
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
    for field, invalid in (("training_eligibility", "HIDDEN_EVAL"), ("status", "inconclusive")):
        modified = {k: v for k, v in report.items() if k != "result_digest"} | {field: invalid}
        modified["result_digest"] = content_digest(modified)
        with pytest.raises(ValueError, match="intervention"):
            intervention_effect(modified, "repair")
    with pytest.raises(ValueError, match="intervention"):
        intervention_effect(report | {"result_digest": "stale"}, "repair")


def test_runtime_adapters_preserve_seed_checkpoint_and_method():
    from evaluation.mdac_comparisons import TrainingRequest, rl_training_config

    r = TrainingRequest(
        "ppo",
        42,
        "/models/local",
        "fixed",
        "policy",
        ("test",) * 4,
        128,
        128,
        "a" * 64,
        max_training_tokens_per_generation=8,
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
        assert cfg.policy.max_new_tokens == 8
        assert cfg.seed == 42
        assert cfg.policy.model_name == "/models/local"
    with pytest.raises(ValueError, match="generation limit"):
        rl_training_config(
            r, dataset="catalog.jsonl", output_dir="runs/arm", batch_size=1, max_new_tokens=16
        )
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


@pytest.mark.parametrize(
    "field", ["training_tokens", "model_config_digest", "max_training_tokens_per_generation"]
)
def test_common_token_budget_and_model_signature_are_enforced(field):
    fixture = fixture_inputs()
    original = fixture["train"]

    def train(request):
        result = original(request)
        assert getattr(request, field) == getattr(fixture["study"], field)
        return replace(result, **{field: "other" if field == "model_config_digest" else 0})

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


def test_freeze_teacher_and_hybrid_runtime_options_in_request_identity():
    from evaluation.causal_records import canonical_json
    from evaluation.mdac_comparisons import TrainingRequest, rl_training_config

    base = TrainingRequest("grpo", 7, "local", "cpu", "review", ("one",), 1, 1, "f" * 64)
    options = canonical_json({"actionopd": {"enabled": False}})
    teacherless = replace(base, runtime_options=options)
    assert teacherless.digest != base.digest
    cfg = rl_training_config(
        teacherless, dataset="train.jsonl", output_dir="out", batch_size=1, max_new_tokens=1
    )
    assert cfg.actionopd.enabled is False
    with pytest.raises(ValueError, match="controlled"):
        rl_training_config(
            replace(base, runtime_options='{"seed":9}'),
            dataset="train.jsonl",
            output_dir="out",
            batch_size=1,
            max_new_tokens=1,
        )


def test_preflight_all_variants_precedes_any_callback():
    fixture = fixture_inputs()

    def abort(*args):
        pytest.fail("invalid study reached a callback")

    lessons = tuple(replace(lesson, variants=()) for lesson in fixture["lessons"])
    with pytest.raises(ValueError, match="missing reviewed"):
        compare(
            **(fixture | {"baseline": abort, "grade": abort, "train": abort, "lessons": lessons}),
            enabled=True,
        )


def test_failed_and_unverified_attempts_do_not_count_as_recovery_latency():
    from evaluation.mdac_comparisons import Judgment, Verdict

    rows = [
        Judgment(
            str(i),
            "ood",
            (),
            Verdict(0, 0),
            Verdict(0, 0, correction_effective=ok, repair_steps=i + 1),
        )
        for i, ok in enumerate((False, None, True))
    ]
    assert summarize(rows)["repair_latency"] == {"numerator": 3, "denominator": 1, "value": 3}


def test_fixture_cli_projects_native_report(tmp_path, monkeypatch):
    import sys

    from evaluation.mdac_comparison_fixture import main

    out = tmp_path / "study.json"
    target = tmp_path / "report"
    monkeypatch.setattr(
        sys, "argv", ["fixture", "--output", str(out), "--report-dir", str(target), "--no-plots"]
    )
    main()
    assert out.is_file()
    assert (target / "telemetry.jsonl").is_file()


def test_divergence_uses_paired_changes_and_excludes_missing_baselines():
    from evaluation.mdac_comparisons import Judgment, Verdict, summarize

    before = Verdict(1, 1, reward=1, verified_behavior=0)
    after = Verdict(1, 1, reward=1, verified_behavior=1)
    row = Judgment("a", "ood", None, before, after)
    unpaired = replace(row, case_id="b", before=replace(before, reward=None))
    result = summarize((row, unpaired))["reward_behavior_divergence"]
    assert result == {"numerator": -1, "denominator": 1, "value": -1}


def test_judge_without_diagnostics_differs_from_assessed_unknown():
    from evaluation.mdac_comparisons import Judgment, Verdict, summarize

    skipped = Verdict(1, 1)
    explicit = Verdict(1, 1, mechanisms=())
    unmeasured = Judgment("a", "final_held_out", None, skipped, skipped)
    assert summarize((unmeasured,))["unknown_rate"]["value"] is None
    assert summarize((unmeasured, replace(unmeasured, after=explicit)))["unknown_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1,
    }


def test_partition_specific_alignment_separates_opposite_changes():
    from evaluation.mdac_comparisons import Judgment, Verdict

    low, high = Verdict(0, 0), Verdict(1, 1)
    rows = [
        Judgment("final", "final_held_out", None, low, high),
        Judgment("ood", "ood", None, high, low),
    ]
    metrics = summarize(rows)
    assert metrics["final_held_out_alignment"] == {"numerator": 1, "denominator": 1, "value": 1}
    assert metrics["final_held_out_alignment_gain"]["value"] == 1
    assert metrics["ood_alignment_gain"]["value"] == -1
    assert metrics["pooled_alignment_gain"] == metrics["alignment_gain"]
    assert metrics["alignment_gain"]["value"] == 0


@pytest.mark.parametrize("field", ["diagnostic_family", "intervention_family"])
def test_substituting_another_registered_family_fails_before_hidden_evaluation(field):
    fixture = fixture_inputs()
    first, second, baseline = fixture["lessons"]
    fixture["lessons"] = (replace(first, **{field: getattr(second, field)}), second, baseline)
    fixture["baseline"] = lambda *_: pytest.fail("substituted family reached evaluation")
    with pytest.raises(ValueError, match="family"):
        compare(**fixture, enabled=True)


def test_each_effect_of_one_sealed_report_has_its_own_treatment():
    from evaluation.causal_records import content_digest

    fixture = fixture_inputs()
    first, *rest = fixture["lessons"]
    report = dict(fixture["interventions"][0][1])
    report.pop("result_digest")
    report["effective_repairs"] = ["repair", "alternative"]
    report["effects"] = dict(report["effects"], alternative={"drs": 0.5, "matched_trials": 2})
    rows = [
        dict(r, treatment="alternative", intervention=i == 0)
        for i, r in enumerate(report["observations"])
    ]
    report["observations"] = list(report["observations"]) + rows
    report["result_digest"] = content_digest(report)
    prior = first.criterion.record
    assessment = replace(prior.assessments[0], intervention_results=(report["result_digest"],))
    record = replace(prior, assessments=(assessment,))
    fixture["diagnostic_families"] += ((content_digest(record.to_dict()), first.diagnostic_family),)
    criterion = replace(first.criterion, record=record)
    fixture["lessons"] = (
        replace(
            first,
            criterion=criterion,
            intervention_report=report["result_digest"],
            intervention_treatment="alternative",
            drs=0.5,
        ),
        *rest,
    )
    receipts = (("alternative", report), ("repair", report), fixture["interventions"][1])
    forward = compare(**(fixture | {"interventions": receipts}), enabled=True)
    assert forward == compare(
        **(fixture | {"interventions": tuple(reversed(receipts))}), enabled=True
    )
    wrong = replace(fixture["lessons"][0], intervention_treatment="repair")
    with pytest.raises(ValueError, match="retained intervention"):
        compare(**(fixture | {"lessons": (wrong, *rest), "interventions": receipts}), enabled=True)


def test_cpu_seed_overflow_fails_at_study_preflight():
    study = fixture_inputs()["study"]
    assert replace(study, methods=("ppo",), seeds=(2**64 - 1,)).seeds == (2**64 - 1,)
    with pytest.raises(ValueError, match="seed"):
        replace(study, methods=("ppo",), seeds=(2**64,))


def test_shipped_generation_limit_reaches_dapo_and_canonical_engine():
    import json
    from pathlib import Path

    from evaluation.mdac_comparisons import (
        TrainingRequest,
        dapo_training_config,
        rl_training_config,
    )

    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/mdac/comparison.json").read_text()
    )
    cap = cfg["max_training_tokens_per_generation"]
    request = TrainingRequest(
        "ppo",
        13,
        "local",
        "fixed",
        "policy",
        ("prompt",),
        1,
        32,
        "a" * 64,
        max_training_tokens_per_generation=cap,
    )
    assert (
        rl_training_config(
            request, dataset="train", output_dir="out", batch_size=1
        ).policy.max_new_tokens
        == cap
        == 32
    )
    pytest.importorskip("torch")
    dapo = replace(request, method="dapo")
    assert dapo_training_config(dapo, output_dir="out", batch_size=1).max_new_tokens == cap
    with pytest.raises(ValueError, match="generation limit"):
        dapo_training_config(dapo, output_dir="out", batch_size=1, max_new_tokens=cap + 1)
