"""Split isolation and reproducible reporting through the public entry points."""

import json
from pathlib import Path

import pytest

from evaluation import mdac_reporting as reporting


def manifest_data():
    return dict(
        schema_version="mdac-split-v1",
        dataset_version="dataset-1",
        evaluator_version="judge-2",
        assignments=[
            dict(family_id="family-" + str(i), partition=p)
            for i, p in enumerate(reporting.PARTITIONS)
        ],
    )


def test_split_digest_is_order_insensitive_and_versions_are_bound():
    data = manifest_data()
    manifest = reporting.SplitManifest.from_dict(data)
    manifest.require(
        "family-0", "training", dataset_version="dataset-1", evaluator_version="judge-2"
    )
    reordered = dict(data, assignments=data["assignments"][::-1])
    assert manifest.digest == reporting.SplitManifest.from_dict(reordered).digest
    assert reporting.SplitManifest.from_dict(manifest.to_dict()) == manifest
    for family, split, dataset, judge in [
        ("family-0", "ood", "dataset-1", "judge-2"),
        ("unknown", "training", "dataset-1", "judge-2"),
        ("family-0", "training", "dataset-1", "wrong"),
        ("family-0", "training", "old", "judge-2"),
    ]:
        with pytest.raises(ValueError):
            manifest.require(family, split, dataset_version=dataset, evaluator_version=judge)
    with pytest.raises(ValueError, match="family"):
        reporting.SplitManifest.from_dict(
            dict(
                data,
                assignments=data["assignments"] + [dict(family_id="family-0", partition="ood")],
            )
        )
    with pytest.raises(ValueError, match="fields"):
        reporting.SplitManifest.from_dict(dict(data, gold_labels=["private"]))


def test_training_projection_excludes_heldout_labels_and_evaluator_fields():
    manifest = reporting.SplitManifest.from_dict(manifest_data())
    record = dict(
        task_id="a",
        family_id="family-0",
        public_prompt="Which fact is supported?",
        expected_action="secret",
        dataset_version="dataset-1",
        evaluator_version="judge-2",
    )
    assert manifest.training_input(record) == dict(
        task_id="a", family_id="family-0", public_prompt=record["public_prompt"]
    )
    with pytest.raises(ValueError):
        manifest.training_input(dict(record, family_id="family-5"))


def estimate(metric, group="all", value=0.5, x=None, y=None):
    return dict(
        metric=metric,
        group=group,
        value=value,
        n=4,
        missing=1,
        interval=[0.2, 0.8],
        interval_method="host paired bootstrap",
        x=x,
        y=y,
    )


def event(manifest, kind="measured", run="run1"):
    return dict(
        schema_version="mdac-telemetry-v1",
        run_id=run,
        round=1,
        arm="C",
        rl_method="GRPO",
        checkpoint="checkpoint-a",
        model="model-a",
        config_hash="a" * 64,
        dataset_version=manifest.dataset_version,
        evaluator_version=manifest.evaluator_version,
        split_digest=manifest.digest,
        seed=17,
        partition="curriculum_validation",
        family_ids=["family-3"],
        kind=kind,
        cost=dict(
            training_tokens=80, evaluation_tokens=20, training_seconds=7, evaluation_seconds=3
        ),
        estimates=[
            estimate("mechanism_rate", "M01"),
            estimate("diagnostic_confidence", "M01"),
            estimate("allocation", "memory", value=0.5),
            estimate("alignment"),
            estimate("capability"),
            estimate("intervention_recovery", "compression"),
            estimate("regression", "ethical", x="pass", y="fail"),
            estimate("retention", "ethical", x="old", y="retained"),
            estimate("calibration", "binned", x=0.7),
            estimate("selective_risk", "thresholded", x=0.6),
            estimate("reward_divergence"),
            estimate("generalization_gap"),
            estimate("temporal_drift", "layer1"),
        ],
        decisions=[
            dict(
                outcome="REJECT",
                reason="retention regressed",
                candidate="checkpoint-a",
                evidence_refs=["receipt-a"],
            )
        ],
    )


def test_export_preserves_run_metadata_missingness_cost_and_source_classes(tmp_path):
    manifest = reporting.SplitManifest.from_dict(manifest_data())
    measured = event(manifest)
    fixture = event(manifest, "illustrative", "fixture")
    planned = dict(event(manifest, "unexecuted", "future"), cost=None, estimates=[], decisions=[])
    paths = reporting.export_report(
        manifest, [measured, fixture, planned], tmp_path / "report", enabled=True, plots=False
    )
    rows = [json.loads(line) for line in Path(paths["telemetry"]).read_text().splitlines()]
    assert rows == [measured, fixture, planned]
    summary = json.loads(Path(paths["summary"]).read_text())
    assert set(summary["cohorts"]) == {"measured", "illustrative", "unexecuted"}
    cohort = summary["cohorts"]["measured"]
    assert cohort["run_count"] == 1 and cohort["observed_costs"]["evaluation_tokens"] == 20
    assert (
        cohort["runs"][0]["seed"] == 17 and cohort["runs"][0]["decisions"] == measured["decisions"]
    )
    assert cohort["metrics"]["mechanism_rate"][0]["interval"] == [0.2, 0.8]
    assert cohort["metrics"]["selective_risk"][0]["n"] == 4
    assert summary["cohorts"]["unexecuted"]["observed_costs"]["training_tokens"] is None
    assert summary["split_digest"] == manifest.digest


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "unexecuted"},
        {"seed": True},
        {"family_ids": ["family-1"]},
        {"split_digest": "b" * 64},
        {"estimates": [estimate("alignment", value=float("nan"))]},
        {"expected_output": "private"},
        {"dataset_version": "dataset-2"},
    ],
)
def test_bad_run_writes_no_partial_artifact(tmp_path, change):
    manifest = reporting.SplitManifest.from_dict(manifest_data())
    destination = tmp_path / "report"
    with pytest.raises(ValueError):
        reporting.export_report(
            manifest,
            [event(manifest), dict(event(manifest, run="bad"), **change)],
            destination,
            enabled=True,
        )
    assert not destination.exists()


def test_plot_catalog_skips_absent_instrumentation_and_labels_data_classes(tmp_path):
    pytest.importorskip("matplotlib")
    manifest = reporting.SplitManifest.from_dict(manifest_data())
    row = event(manifest, "illustrative")
    row["estimates"] = [e for e in row["estimates"] if e["metric"] != "temporal_drift"]
    result = reporting.export_report(manifest, [row], tmp_path / "report", enabled=True)
    plotdir = Path(result["summary"]).parent / "plots" / "illustrative"
    assert {p.stem for p in plotdir.glob("*.png")} == set(reporting.PLOTS) - {"mechanistic"}
    assert all(p.stat().st_size > 1000 for p in plotdir.glob("*.png"))
    assert not (plotdir.parent / "measured").exists()
    assert not (plotdir / "mechanistic.png").exists()


def test_default_disabled_and_existing_exports_are_preserved(tmp_path):
    manifest = reporting.SplitManifest.from_dict(manifest_data())
    with pytest.raises(ValueError, match="enabled"):
        reporting.export_report(manifest, [], tmp_path / "none")
    dest = tmp_path / "report"
    reporting.export_report(manifest, [event(manifest)], dest, enabled=True, plots=False)
    first = (dest / "telemetry.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        reporting.export_report(manifest, [], dest, enabled=True)
    assert (dest / "telemetry.jsonl").read_bytes() == first


def test_duplicate_run_and_invalid_interval_are_rejected(tmp_path):
    manifest = reporting.SplitManifest.from_dict(manifest_data())
    row = event(manifest)
    with pytest.raises(ValueError, match="duplicate"):
        reporting.export_report(manifest, [row, row], tmp_path / "same", enabled=True)
    row["estimates"][0]["interval"] = [0.8, 0.2]
    with pytest.raises(ValueError, match="interval"):
        reporting.export_report(manifest, [row], tmp_path / "inverted", enabled=True)


def test_split_builder_assigns_whole_families_reproducibly():
    families = ["root-" + str(i) for i in range(14)]
    counts = {p: 2 for p in reporting.PARTITIONS}
    build = reporting.SplitManifest.allocate
    manifest = build(families, counts, seed=51, dataset_version="d", evaluator_version="e")
    assert manifest == build(
        families[::-1], counts, seed=51, dataset_version="d", evaluator_version="e"
    )
    assert manifest != build(families, counts, seed=52, dataset_version="d", evaluator_version="e")
    assert all(sum(p == q for _, q in manifest.assignments) == 2 for p in counts)
    with pytest.raises(ValueError, match="counts"):
        build(families[:-1], counts, seed=51, dataset_version="d", evaluator_version="e")


def test_training_projection_rejects_old_dataset():
    m = reporting.SplitManifest.from_dict(manifest_data())
    with pytest.raises(ValueError, match="version"):
        m.training_input(
            dict(
                task_id="a",
                family_id="family-0",
                public_prompt="hello",
                dataset_version="outdated",
                evaluator_version="judge-2",
            )
        )


def test_adapter_preserves_metric_direction_and_sample_count():
    from gepa_mindfulness.verification.mdac_metrics import measure

    metric = measure(
        "M02", [dict(id="one", relevant=True, retrieved=False)], enabled=True, independent=True
    )["recall"]
    point = reporting.metric_estimate("retention", metric, group="annotated evidence")
    assert point["value"] == 0 and point["n"] == 1
    assert point["interval"] == list(metric.interval)


def test_intervention_adapter_checks_receipt_and_preserves_paired_uncertainty():
    from evaluation.causal_records import content_digest

    report = dict(
        schema_version="mdac-interventions-v1",
        planned_blocks=3,
        interval_method="paired",
        effects={
            "repair": dict(
                matched_trials=2,
                control_passes=0,
                intervention_passes=1,
                drs=0.5,
                interval=[-0.2, 0.9],
            )
        },
    )
    receipt = dict(report, result_digest=content_digest(report))
    (point,) = reporting.intervention_estimates(receipt)
    assert (point["value"], point["n"], point["missing"], point["interval"]) == (
        0.5,
        2,
        1,
        [-0.2, 0.9],
    )
    with pytest.raises(ValueError, match="digest"):
        reporting.intervention_estimates(dict(receipt, planned_blocks=2))


def test_packaged_example_and_optional_local_model(tmp_path):
    import os
    from importlib.resources import files

    data = json.loads(
        files("gepa_mindfulness").joinpath("data/mdac_reporting_example.json").read_text()
    )
    manifest = reporting.SplitManifest.from_dict(data["manifest"])
    report = reporting.export_report(
        manifest, data["runs"], tmp_path / "example", enabled=True, plots=False
    )
    assert (
        json.loads(Path(report["summary"]).read_text())["cohorts"]["illustrative"]["run_count"] == 1
    )
    checkpoint = os.environ.get("MDAC_LOCAL_TEST_MODEL")
    if not checkpoint:
        pytest.skip("set MDAC_LOCAL_TEST_MODEL to export a measured local-model point")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).eval()
    with torch.inference_mode():
        ids = tokenizer("An example", return_tensors="pt")
        logits = model(**ids).logits[0, -1].float()
        probability, token = logits.softmax(-1).max(dim=-1)
        confidence = probability.item()
        target = tokenizer.encode(" sentence", add_special_tokens=False)[0]
        correct = float(token.item() == target)
    row = event(manifest)
    row.update(
        model=model.config.model_type,
        checkpoint=checkpoint,
        cost=None,
        decisions=[],
        estimates=[
            dict(
                estimate("calibration", "next-token", correct, x=confidence),
                interval=None,
                interval_method="single observation",
                n=1,
                missing=0,
            )
        ],
    )
    path = reporting.export_report(manifest, [row], tmp_path / "model", enabled=True, plots=False)
    assert json.loads(Path(path["telemetry"]).read_text())["estimates"][0]["x"] == confidence
