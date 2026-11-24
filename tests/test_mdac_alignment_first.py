"""An opt-in staged learner must retain source eligibility and independent admission."""

import pytest
from mdac_alignment_fixture import lifecycle_fixture

from evaluation.mdac_alignment_first import ORDERS, run_lifecycle


@pytest.mark.parametrize("order", ORDERS)
def test_cpu_cycle_runs_trainer_evaluator_allocator_and_governor(tmp_path, order):
    inputs = lifecycle_fixture(tmp_path / order)
    result = run_lifecycle(order=order, **inputs, enabled=True)
    assert len(result["rounds"]) == 3
    assert sum(r["samples"] for r in result["rounds"]) == 12
    case = result["case_interventions"][0]
    source = result["allocation"]["inputs"]["criteria"][0]
    assert source["record"]["assessments"][0]["state"] == "supported"
    assert case["result_digest"] in source["record"]["assessments"][0]["intervention_results"]
    assert (
        source["record"]["task_id"]
        == case["roster"][0]["identity"]["task_id"]
        == "lossy_compression"
    )
    assert source["current"]["normalized"] == 0 and case["effects"]["repair"]["drs"] == 1
    assert result["intervention"]["status"] == "observed"
    assert result["intervention"]["effects"]["repair"]["drs"] == 1
    assert result["intervention"]["effects"]["sham"]["drs"] == 0
    assert result["verification"] and all(row["trial_refs"] for row in result["verification"])
    assert set(result["final_profiles"]) == {"final_held_out", "ood"}
    assert result["training_eligibility"] == "HIDDEN_EVAL"
    assert result["admission"]["decision"] == "DEFER"
    assert result["admission"]["execute_candidate"] is False
    assert result["rounds"][-1]["retention_samples"] >= 1
    assert result["admission"]["evidence_digest"]
    assert result["admission_profile"]["checkpoint"] == result["rounds"][-1]["checkpoint"]
    assert all(r["validation"]["rows"] for r in result["rounds"])
    print(order, result["rounds"][-1]["checkpoint"], result["admission"]["decision"])


def test_ineligible_inputs_block_before_training(tmp_path):
    from dataclasses import replace

    inputs = lifecycle_fixture(tmp_path)
    first = inputs["catalog"][0]
    invalid = replace(
        first, request=replace(first.request, metadata={"training_eligibility": "HIDDEN_EVAL"})
    )
    inputs["catalog"] = (invalid, *inputs["catalog"][1:])
    with pytest.raises(ValueError, match="eligib"):
        run_lifecycle(order="alignment_first", **inputs, enabled=True)


def test_disabled_and_mismatched_admission_binding(tmp_path):
    from dataclasses import replace

    inputs = lifecycle_fixture(tmp_path)
    with pytest.raises(ValueError, match="enabled"):
        run_lifecycle(order="mixed", **inputs)
    original = inputs["prepare_admission"]

    def unrelated(*args):
        context = original(*args)
        return replace(
            context, proposal=replace(context.proposal, artifact_digest="sha256:" + "f" * 64)
        )

    with pytest.raises(ValueError, match="checkpoint"):
        run_lifecycle(order="mixed", **(inputs | {"prepare_admission": unrelated}), enabled=True)


def test_local_checkpoint_grpo_cycle(tmp_path):
    import os
    from pathlib import Path

    local = os.getenv("MDAC_LOCAL_TEST_MODEL")
    if not local:
        pytest.skip("set MDAC_LOCAL_TEST_MODEL to a local Transformers causal checkpoint")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mdac_alignment_fixture import use_local_grpo

    inputs = lifecycle_fixture(tmp_path)
    trainer = use_local_grpo(inputs, Path(local), tmp_path / "trained")
    case_probe = next(p for p in inputs["probes"] if p.family == "lossy_compression")
    assert not inputs["grade"](case_probe, "disclose")
    report = run_lifecycle(order="alignment_first", **inputs, enabled=True)
    assert all(
        not observation["control"]
        for observation in report["case_interventions"][0]["observations"]
    )
    assert report["phase_evidence"]["repair_continuation"] == "cpu_fixture"
    assert report["case_interventions"][0]["status"] == "inconclusive"
    assert (
        report["allocation"]["inputs"]["criteria"][0]["record"]["assessments"][0]["state"]
        == "inconclusive"
    )
    assert len(trainer.checkpoints) > 1
    assert sum(r["tokens"] for r in report["rounds"]) == 48
    assert report["rounds"][-1]["checkpoint"] != trainer.start
    assert report["admission"]["decision"] == "DEFER"
    print(
        "tiny GRPO",
        trainer.start,
        report["rounds"][-1]["checkpoint"],
        report["admission"]["decision"],
    )


def test_failed_repair_cannot_enter_third_training_round(tmp_path):
    from dataclasses import replace

    fixture = lifecycle_fixture(tmp_path)
    original = fixture["verify_repair"]
    calls = []

    def verify(*args):
        calls.append(args[0].family)
        return replace(original(*args), reason="retention failed", training_example=None)

    with pytest.raises(ValueError, match="continuation-verified"):
        run_lifecycle(
            order="alignment_first", **(fixture | {"verify_repair": verify}), enabled=True
        )
    assert calls


def test_order_controls_have_equal_exposures_and_distinct_initial_trajectories(tmp_path):
    from collections import Counter

    reports = [
        run_lifecycle(order=o, **lifecycle_fixture(tmp_path / o), enabled=True) for o in ORDERS
    ]
    initial = [r["rounds"][0]["training_families"] for r in reports]
    assert len({tuple(v) for v in initial}) == 3
    totals = [
        Counter(f for row in r["rounds"][:2] for f in row["training_families"]) for r in reports
    ]
    assert totals[0] == totals[1] == totals[2]


@pytest.mark.parametrize("tamper", ["checkpoint", "empty", "family", "ref"])
def test_repair_receipts_bind_current_family_checkpoint_and_trial_roster(tmp_path, tamper):
    from dataclasses import replace

    fixture = lifecycle_fixture(tmp_path)
    host = fixture["verify_repair"]

    def verify(example, checkpoint):
        result = host(example, "stale" if tamper == "checkpoint" else checkpoint)
        if tamper == "empty":
            return replace(result, trials=())
        if tamper == "family":
            return host(replace(example, family="other"), checkpoint)
        if tamper == "ref":
            return replace(result, trials=result.trials[1:])
        return result

    with pytest.raises(ValueError, match="repair"):
        run_lifecycle(order="mixed", **(fixture | {"verify_repair": verify}), enabled=True)


def test_reused_final_prompt_fails_before_training(tmp_path):
    from dataclasses import replace

    f = lifecycle_fixture(tmp_path)
    dev = next(p.prompt for p in f["probes"] if p.partition == "diagnostic_development")
    f["probes"] = tuple(
        replace(p, prompt=dev) if p.partition == "final_held_out" else p for p in f["probes"]
    )
    with pytest.raises(ValueError, match="prompt"):
        run_lifecycle(order="mixed", **f, enabled=True)


def test_calibration_alone_cannot_support_a_development_cause(tmp_path):
    from dataclasses import replace

    f = lifecycle_fixture(tmp_path)
    f["case_tests"] = ()
    original = f["diagnose"]

    def diagnose(profile, predict, reports):
        criteria, family, report = original(profile, predict, reports)
        c = criteria[0]
        a = replace(c.record.assessments[0], state="supported")
        c = replace(c, record=replace(c.record, assessments=(a,)))
        return (c,), family, report

    with pytest.raises(ValueError, match="case-specific"):
        run_lifecycle(order="mixed", **(f | {"diagnose": diagnose}), enabled=True)


def test_zero_effect_sham_cannot_be_a_calibrated_repair(tmp_path):
    from evaluation.causal_records import content_digest

    f = lifecycle_fixture(tmp_path)
    original = f["diagnose"]

    def altered(profile, predict, reports):
        cs, family, report = original(profile, predict, reports)
        report["effective_repairs"] = ["sham"]
        report["result_digest"] = content_digest(
            {k: v for k, v in report.items() if k != "result_digest"}
        )
        return cs, family, report

    with pytest.raises(ValueError, match="paired repair"):
        run_lifecycle(order="mixed", **(f | {"diagnose": altered}), enabled=True)


def test_replaced_model_weights_fail_before_an_update(tmp_path):
    import os
    from pathlib import Path

    local = os.getenv("MDAC_LOCAL_TEST_MODEL")
    if not local:
        pytest.skip("set MDAC_LOCAL_TEST_MODEL to run the local weight integrity control")
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    from evaluation.mdac_comparisons import TrainingRequest
    from gepa_mindfulness.examples.mdac_small_model import LocalOneTokenGRPO

    trainer = LocalOneTokenGRPO(Path(local), tmp_path / "trained", {"safe": "disclose"})
    wrong = AutoModelForCausalLM.from_pretrained(local, local_files_only=True)
    with torch.no_grad():
        next(wrong.parameters()).add_(1)
    path = tmp_path / "replaced"
    wrong.save_pretrained(path)
    trainer.checkpoints[trainer.start] = path
    request = TrainingRequest(
        "grpo", 7, trainer.start, "fixture", "policy", ("safe",), 1, 4, trainer.config_digest
    )
    with pytest.raises(ValueError, match="parent checkpoint"):
        trainer.train(request, ("disclose",))


def test_case_replay_requires_the_declared_factor_evaluator_and_observed_input():
    from dataclasses import replace

    from evaluation.causal_records import content_digest
    from evaluation.mdac_alignment_first import CaseReplay, _case_support
    from evaluation.mdac_interventions import run_interventions
    from gepa_mindfulness.training.eligibility import TrainingEligibility
    from synthetic_data.mdac_interventions import (
        EVALUATOR,
        FixtureOracle,
        fixture_actor,
        known_fault_block,
    )

    family = "missing_context"
    block = known_fault_block(
        family, task_id=family, checkpoint="checkpoint", eligibility=TrainingEligibility.DEVELOPMENT
    )
    oracle = FixtureOracle()
    report = run_interventions(
        (block,),
        execute=fixture_actor,
        adjudicate=oracle.adjudicate,
        authenticate=oracle.authenticate,
        max_calls=4,
        alternatives=("salience",),
        enabled=True,
    )
    repair = next(r for r in report["observations"] if r["treatment"] == "repair")
    turns = repair["source_record"]["before"]["turns"]
    row = dict(
        correct=repair["control"],
        response=repair["captures"][0]["actions"][-1],
        prompt_digest=content_digest("\n".join(t["content"] for t in turns)),
    )
    spec = CaseReplay(
        family, "M01", "repair", "entries", tuple(vars(EVALUATOR).values()), lambda *_: report
    )
    assert _case_support(spec, report, row, "checkpoint") == report["result_digest"]
    for wrong in (
        replace(spec, factor="fee"),
        replace(spec, evaluator=("other", "1", "rubric")),
        replace(spec, family="other"),
    ):
        with pytest.raises(ValueError, match="case-specific"):
            _case_support(wrong, report, row, "checkpoint")
    with pytest.raises(ValueError, match="case-specific"):
        _case_support(spec, report, row | {"response": "misbound"}, "checkpoint")
