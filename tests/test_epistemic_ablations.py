"""Reproducible ablations retain family splits and separate measured dimensions."""

import pytest


def test_ablation_matrix_is_complete_and_fixture_run_is_labeled() -> None:
    from evaluation.epistemic_ablations import ablation_matrix, fixture_smoke

    matrix = ablation_matrix()
    assert tuple(matrix) == tuple("ABCDEFGHIJK")
    assert matrix["A"] == ()
    assert "sift" in matrix["K"] and "sift" not in matrix["J"]
    smoke = fixture_smoke()
    assert smoke["measurement_kind"] == "DETERMINISTIC_CONTRACT_FIXTURE"
    assert smoke["selected_check_ids"] == ("discriminate",)
    assert smoke["model_benchmark_run"] is False


def test_family_split_has_no_variant_leakage_and_summary_preserves_costs() -> None:
    from evaluation.epistemic_ablations import family_split, summarize_trials

    assert family_split("same-family", seed=42) == family_split("same-family", seed=42)
    rows = [
        {
            "run_id": "r1",
            "ablation": "A",
            "family_id": "f1",
            "split": "DEVELOPMENT",
            "metrics": {"correctness": 0.5, "rationale_migration": 0.2},
            "cost": {"tool_calls": 2, "latency_seconds": 3.0, "verification_cost": 1.0},
        },
        {
            "run_id": "r2",
            "ablation": "A",
            "family_id": "f2",
            "split": "DEVELOPMENT",
            "metrics": {"correctness": 1.0, "rationale_migration": 0.0},
            "cost": {"tool_calls": 1, "latency_seconds": 1.0, "verification_cost": 0.5},
        },
    ]
    summary = summarize_trials(rows)
    assert summary["A"]["metrics"]["correctness"]["mean"] == 0.75
    assert summary["A"]["cost_totals"]["tool_calls"] == 3
    assert summary["A"]["metrics"]["rationale_migration"]["mean"] == 0.1
    with pytest.raises(ValueError, match="split"):
        summarize_trials(rows + [rows[0] | {"run_id": "r3", "split": "HIDDEN_EVAL"}])
    with pytest.raises(ValueError):
        summarize_trials(rows + [rows[0]])
