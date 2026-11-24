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
    report = run_lifecycle(order="alignment_first", **inputs, enabled=True)
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
