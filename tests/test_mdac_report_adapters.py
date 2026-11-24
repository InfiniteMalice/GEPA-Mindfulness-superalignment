import json
from pathlib import Path

import pytest

from evaluation import mdac_report_adapters as adapters
from evaluation.causal_records import content_digest
from evaluation.mdac_reporting import SplitManifest, export_report


def manifest():
    return SplitManifest(
        "d",
        "e",
        tuple(
            (name, part)
            for name, part in [
                ("train", "training"),
                ("seen", "final_held_out"),
                ("novel", "ood"),
                ("dev", "diagnostic_development"),
                ("cv", "curriculum_validation"),
                ("admit", "governance_admission"),
            ]
        ),
    )


def sealed(version, m, **data):
    inputs = dict(
        study=dict(
            checkpoint="initial",
            seeds=[7],
            methods=["fixture"],
            dataset_version="d",
            evaluator_version="e",
            model_config_digest="a" * 64,
            evidence_kind="cpu_fixture",
        ),
        split_digest=m.digest,
        order="alignment_first",
    )
    report = dict(schema_version=version, inputs=inputs, evidence_kind="cpu_fixture", **data)
    return dict(report, result_digest=content_digest(report))


def verdict(score):
    return dict(
        alignment=score,
        capability=score,
        mechanisms=["M01"] if not score else [],
        diagnostic_confidence=0.75,
        reward=score,
        verified_behavior=score,
    )


def test_comparison_export_counts_training_and_shared_baselines_once(tmp_path):
    m = manifest()
    cells = []
    for arm in ("A", "B"):
        cells.append(
            dict(
                arm=arm,
                method="fixture",
                seed=7,
                checkpoint_start="initial",
                checkpoint_end=arm,
                selected_families=["train"],
                schedule_channels=["retention_replay"],
                actual=dict(
                    training_tokens=8,
                    compute_units=1,
                    samples=1,
                    evaluation_tokens=3,
                    baseline_evaluation_tokens=2,
                ),
                judgments=[
                    dict(
                        case_id=family,
                        family=family,
                        partition=part,
                        truth=["secret"],
                        before=verdict(0),
                        after=verdict(v),
                        before_tokens=1,
                        after_tokens=n,
                    )
                    for family, part, v, n in [
                        ("seen", "final_held_out", 1, 1),
                        ("novel", "ood", 0, 2),
                    ]
                ],
            )
        )
    report = sealed("mdac-comparison-v1", m, cells=cells)
    runs = adapters.comparison_runs(report, m)
    paths = export_report(m, runs, tmp_path / "compare", enabled=True, plots=False)
    assert "secret" not in Path(paths["telemetry"]).read_text()
    summary = json.loads(Path(paths["summary"]).read_text())["cohorts"]["illustrative"]
    assert summary["observed_costs"]["training_tokens"] == 16
    assert summary["observed_costs"]["evaluation_tokens"] == 8
    assert summary["observed_costs"]["compute_units"] == 2
    points = [p for p in summary["metrics"]["alignment"] if p["round"] == 1]
    assert {(p["partition"], p["value"]) for p in points} == {("final_held_out", 1), ("ood", 0)}
    with pytest.raises(ValueError, match="digest"):
        adapters.comparison_runs(dict(report, cells=[]), m)


def test_lifecycle_export_heldout_ood_and_admission(tmp_path):
    m = manifest()

    def profile(checkpoint, part, family, correct):
        return dict(
            checkpoint=checkpoint,
            partition=part,
            split_digest=m.digest,
            evaluator="e",
            rows=[
                dict(
                    family=family,
                    dimension="alignment",
                    correct=correct,
                    tokens=1,
                    response="private completion",
                )
            ],
        )

    admission = dict(
        report_digest="b" * 64,
        candidate_id="candidate",
        decision="DEFER",
        reasons=["missing independent evidence"],
        rollback_target="initial",
    )
    rounds = [
        dict(
            stage="correction",
            source="initial",
            checkpoint="c",
            validation=profile("c", "curriculum_validation", "cv", True),
            training_families=["train"],
            samples=1,
            tokens=4,
            compute_units=2,
            retention_samples=1,
        )
    ]
    report = sealed(
        "mdac-alignment-lifecycle-v1",
        m,
        initial=profile("initial", "curriculum_validation", "cv", False),
        rounds=rounds,
        allocation=dict(channel_counts={"retention_replay": 1}),
        diagnostic_profile=profile("initial", "diagnostic_development", "dev", False),
        admission_profile=profile("c", "governance_admission", "admit", True),
        admission=admission,
        final_profiles={
            p: profile("c", p, f, ok)
            for p, f, ok in [("final_held_out", "seen", True), ("ood", "novel", False)]
        },
    )
    rows = adapters.lifecycle_runs(report, m)
    paths = export_report(m, rows, tmp_path / "lifecycle", enabled=True, plots=False)
    parsed = json.loads(Path(paths["summary"]).read_text())["cohorts"]["illustrative"]
    assert parsed["observed_costs"]["training_tokens"] == 4
    assert parsed["observed_costs"]["evaluation_tokens"] == 6
    assert {r["partition"] for r in rows} >= {"final_held_out", "ood", "governance_admission"}
    (decision,) = [d for r in rows for d in r["decisions"]]
    assert decision["outcome"] == "DEFER" and decision["reason"] == "missing independent evidence"
    assert "private completion" not in Path(paths["telemetry"]).read_text()


def test_dense_figure_uses_readable_run_labels_and_preserves_all_points():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from evaluation.mdac_report_plots import _heatmap

    rows = [
        dict(
            metric="alignment",
            group="overall",
            arm="C",
            rl_method="grpo",
            round=3,
            seed=i,
            partition="final_held_out",
            run_id="a" * 64 + str(i),
            value=0.5,
            n=5,
            missing=0,
            interval=None,
            x=None,
            y=None,
        )
        for i in range(15)
    ]
    ax = Figure().subplots()
    _heatmap(ax, rows, "cross_rl")
    assert ax.images[0].get_array().count() == len(rows)
    for label in ax.get_yticklabels():
        assert "final_held_out" in label.get_text() and "a" * 64 not in label.get_text()


def test_private_case_labels_only_contribute_aggregate_diagnostic_calibration(tmp_path):
    m = manifest()
    seen = dict(
        case_id="seen",
        family="seen",
        partition="final_held_out",
        truth=["M01"],
        before=verdict(0),
        after=verdict(1),
        before_tokens=1,
        after_tokens=1,
    )
    novel = dict(seen, case_id="novel", family="novel", partition="ood", truth=None)
    cell = dict(
        arm="A",
        method="fixture",
        seed=7,
        checkpoint_start="initial",
        checkpoint_end="fit",
        selected_families=["train"],
        schedule_channels=["review"],
        actual=dict(
            training_tokens=3,
            samples=1,
            compute_units=1,
            evaluation_tokens=2,
            baseline_evaluation_tokens=2,
        ),
        judgments=[seen, novel],
    )
    rows = adapters.comparison_runs(sealed("mdac-comparison-v1", m, cells=[cell]), m)
    cases = [r for r in rows if r["round"] == 1 and r["partition"] == "final_held_out"][0]
    (cal,) = [p for p in cases["estimates"] if p["metric"] == "calibration"]
    (risk,) = [p for p in cases["estimates"] if p["metric"] == "selective_risk"]
    assert (cal["x"], cal["value"], cal["n"]) == (0.75, 0, 1)
    assert (risk["x"], risk["value"], risk["n"]) == (1, 1, 1)
    assert all(
        not any(p["metric"] == "calibration" for p in r["estimates"])
        for r in rows
        if r["partition"] == "ood"
    )


def test_plotting_digest_bound_conditions_without_clipping(tmp_path):
    pytest.importorskip("matplotlib")
    import warnings

    from test_mdac_reporting import estimate, event, manifest_data

    from evaluation.mdac_report_plots import render_plots
    from evaluation.mdac_reporting import _summary

    m = SplitManifest.from_dict(manifest_data())
    r = event(m, run="b" * 64)
    r["estimates"] = [estimate("mechanism_rate", "M02", x="evidence:sha256:" + "a" * 64)]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Tight layout not applied.*")
        assert render_plots(_summary(m, [r])["cohorts"], tmp_path)


def test_comparison_telemetry_cannot_be_admitted_by_the_repository_training_reader():
    from gepa_mindfulness.training.eligibility import require_training_eligible

    m = manifest()
    judgments = [
        dict(
            case_id=f,
            family=f,
            partition=p,
            before=verdict(1),
            after=verdict(1),
            truth=[],
            before_tokens=1,
            after_tokens=1,
        )
        for f, p in [("seen", "final_held_out"), ("novel", "ood")]
    ]
    cell = dict(
        arm="A",
        method="fixture",
        seed=7,
        checkpoint_start="initial",
        checkpoint_end="fit",
        selected_families=["train"],
        schedule_channels=["review"],
        actual=dict(
            training_tokens=1,
            samples=1,
            compute_units=1,
            evaluation_tokens=2,
            baseline_evaluation_tokens=2,
        ),
        judgments=judgments,
    )
    for row in adapters.comparison_runs(sealed("mdac-comparison-v1", m, cells=[cell]), m):
        with pytest.raises(ValueError, match="HIDDEN_EVAL"):
            require_training_eligible(row)


@pytest.mark.parametrize("first_known", [True, False])
def test_baseline_deduplication_binds_the_private_calibration_reference(first_known):
    m = manifest()
    after_cells = []
    for i, arm in enumerate(("A", "B")):
        known = first_known if i == 0 else not first_known
        judgments = [
            dict(
                case_id=f,
                family=f,
                partition=p,
                before=verdict(1),
                after=verdict(1),
                truth=[] if known else None,
                before_tokens=1,
                after_tokens=1,
            )
            for f, p in [("seen", "final_held_out"), ("novel", "ood")]
        ]
        after_cells.append(
            dict(
                arm=arm,
                method="fixture",
                seed=7,
                checkpoint_start="initial",
                checkpoint_end=arm,
                selected_families=["train"],
                schedule_channels=["review"],
                actual=dict(
                    training_tokens=1,
                    samples=1,
                    compute_units=1,
                    evaluation_tokens=2,
                    baseline_evaluation_tokens=2,
                ),
                judgments=judgments,
            )
        )
    report = sealed("mdac-comparison-v1", m, cells=after_cells)
    with pytest.raises(ValueError, match="shared baseline"):
        adapters.comparison_runs(report, m)
