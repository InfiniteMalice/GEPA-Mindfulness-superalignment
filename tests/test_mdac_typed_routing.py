"""Procedural fast routing, independent calibration and exact host authority controls."""

from dataclasses import replace

import pytest

from evaluation.improvement_records import record_digest
from gepa_mindfulness.factuality_observability import typed_routing as tr
from gepa_mindfulness.factuality_observability.decision_workflow import run_decision_workflow
from gepa_mindfulness.factuality_observability.epistemic_routing import prepare_routing
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from tests.test_decision_workflow import setup_workflow
from tests.test_epistemic_routing import ENABLED, request


def label(case, source, choice, confidence, gold=Action.ACCEPT):
    return tr.CalibrationLabel(
        case,
        "model-v1",
        "clarification",
        choice,
        confidence,
        gold,
        source,
        "eval-v1",
        "obs:" + case,
    )


def calibration():
    return tr.calibrate_threshold(
        (
            label("a", "independent_label", Action.ACCEPT, 0.96),
            label("b", "verified_execution", Action.ACCEPT, 0.90),
            label("c", "verified_execution", Action.CLARIFY, 0.75),
        ),
        target_accuracy=0.95,
        minimum_examples=2,
    )


def inputs(features, kind="enumerable", choice=Action.ACCEPT, confidence=0.95, high=False):
    options = (Action.ACCEPT, Action.CLARIFY) if kind == "enumerable" else ()
    task = tr.RoutingTask("new", features.input_digest, "clarification", kind, options, high)
    if kind != "enumerable":
        return tr.TypedRoutingInput(task)
    pred = tr.FastPrediction(record_digest(task), "model-v1", choice, confidence)
    return tr.TypedRoutingInput(task, pred, calibration())


def test_threshold_maximizes_empirical_coverage_at_target():
    cal = calibration()
    assert cal.threshold == 0.90
    assert cal.accepted == 2 and cal.correct == 2
    assert len(cal.sample_ids) == 3
    with pytest.raises(ValueError, match="teacher|independent"):
        calibration_input = label("a", "teacher_consensus", Action.ACCEPT, 1.0)
        tr.calibrate_threshold((calibration_input,), minimum_examples=1)


@pytest.mark.parametrize(
    ("kind", "choice", "conf", "high", "route"),
    [
        ("enumerable", Action.ACCEPT, 0.95, False, Action.ACCEPT),
        ("enumerable", Action.CLARIFY, 0.95, False, Action.CLARIFY),
        ("enumerable", Action.ACCEPT, 0.75, False, Action.ESCALATE),
        ("enumerable", None, None, False, Action.ESCALATE),
        ("enumerable", Action.ACCEPT, 1.0, True, Action.ROUTE_EXTERNAL),
        ("deterministic", None, None, False, Action.ACCEPT),
        ("generative", None, None, False, Action.DECOMPOSE_AND_VERIFY),
        ("unknown", None, None, False, Action.ESCALATE),
    ],
)
def test_typed_routing(kind, choice, conf, high, route):
    features = prepare_routing(request(), ENABLED)
    value = inputs(features, kind, choice, conf, high)
    report = tr.evaluate_typed_task(features, value)
    assert report["routing"]["action"] == route.value
    assert not report["confers_authority"]


def test_required_verification_precedes_confident_enumerable_readout():
    req = request()
    features = prepare_routing(
        replace(req, context=replace(req.context, verification_required=True)), ENABLED
    )
    result = tr.evaluate_typed_task(features, inputs(features))
    assert result["routing"]["action"] == Action.ROUTE_EXTERNAL.value


def test_stale_case_or_checkpoint_and_leakage_rejected():
    features = prepare_routing(request(), ENABLED)
    value = inputs(features)
    for bad in [
        replace(value, task=replace(value.task, state_digest="f" * 64)),
        replace(value, prediction=replace(value.prediction, model_version="other")),
    ]:
        with pytest.raises(ValueError):
            tr.evaluate_typed_task(features, bad)
    task = replace(value.task, case_id="a")
    leaked = replace(
        value, task=task, prediction=replace(value.prediction, task_digest=record_digest(task))
    )
    with pytest.raises(ValueError, match="calibration"):
        tr.evaluate_typed_task(features, leaked)


def test_fast_route_cannot_grant_or_redirect_runtime_authority():
    req, args, calls, grants = setup_workflow()
    value = inputs(prepare_routing(req, ENABLED))
    args.update(backend=None, confidence_threshold=None, typed_routing=value)
    original = args["authorize"]

    def misbound(packet):
        action, decision, registry = original(packet)
        return replace(action, authorization_scope="wrong"), decision, registry

    with pytest.raises(ValueError, match="binding"):
        run_decision_workflow(req, **dict(args, authorize=misbound))
    assert not calls
    result = run_decision_workflow(req, **args)
    assert calls == [Action.ACCEPT] and len(grants) == 2
    assert result.to_dict()["source"]["typed_routing"] == value.to_dict()


def test_absent_or_insufficient_qualified_calibration_defers():
    insufficient = tr.calibrate_threshold(
        (label("bad", "verified_execution", Action.CLARIFY, 1.0),), minimum_examples=1
    )
    assert insufficient.threshold is None
    features = prepare_routing(request(), ENABLED)
    value = inputs(features)
    for cal in (None, insufficient):
        report = tr.evaluate_typed_task(features, replace(value, calibration=cal))
        assert report["routing"]["action"] == Action.ESCALATE.value


def test_heldout_reliability_calibrates_against_actual_labels():
    from evaluation.system_one_benchmark import benchmark_typed_calibration

    cal = calibration()
    rows = (
        label("x", "independent_label", Action.CLARIFY, 1.0),
        label("y", "verified_execution", Action.ACCEPT, 0.9),
        label("z", "verified_execution", None, None),
    )
    assert benchmark_typed_calibration(cal, rows) is None
    report = benchmark_typed_calibration(cal, rows, enabled=True)
    assert report["eligible"] == 3 and report["accepted"] == 2 and report["unknown"] == 1
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["selective_accuracy"] == 0.5
    assert report["brier_score"] == pytest.approx(0.505)
    with pytest.raises(ValueError, match="held-out"):
        benchmark_typed_calibration(cal, (cal.labels[0],), enabled=True)


def test_tied_scores_and_no_qualified_cutoff():
    rows = (
        label("a", "independent_label", Action.ACCEPT, 1.0),
        label("b", "verified_execution", Action.CLARIFY, 1.0),
    )
    cal = tr.calibrate_threshold(rows, target_accuracy=1.0, minimum_examples=1)
    assert cal.operating_point() == (None, 0, 0)


def test_unavailable_explicit_unknown_never_invokes_fast_model():
    from gepa_mindfulness.factuality_observability.decision_workflow import DecisionWorkflowConfig

    req, args, calls, _ = setup_workflow()
    value = inputs(prepare_routing(req, ENABLED), kind="unknown")
    args.update(backend=None, confidence_threshold=None, typed_routing=value)
    assert run_decision_workflow(req, **dict(args, config=DecisionWorkflowConfig())) is None
    assert not calls
    result = run_decision_workflow(req, **args)
    assert calls == [Action.ESCALATE]
    assert result.to_dict()["decisions"][0]["raw_choice"] is None


@pytest.mark.parametrize("target", sorted(tr.TARGETS))
def test_each_procedural_target_including_authorization_is_advisory(target):
    features = prepare_routing(request(), ENABLED)
    value = inputs(features, kind="deterministic")
    report = tr.evaluate_typed_task(
        features, replace(value, task=replace(value.task, target=target))
    )
    assert report["reason"] == "deterministic_guard"
    assert report["confers_authority"] is False


def test_optional_model_calibration_capture():
    import json
    import os
    from pathlib import Path

    from evaluation.system_one_benchmark import benchmark_typed_calibration

    path = os.environ.get("MDAC_TYPED_MODEL_LABELS")
    if path is None:
        pytest.skip("set MDAC_TYPED_MODEL_LABELS to independent calibration and held-out labels")
    data = json.loads(Path(path).read_text())
    calibration = tr.calibrate_threshold(
        tuple(tr.CalibrationLabel.from_dict(r) for r in data["calibration"]),
        minimum_examples=data["minimum_examples"],
        target_accuracy=data["target_accuracy"],
    )
    report = benchmark_typed_calibration(
        calibration, tuple(tr.CalibrationLabel.from_dict(r) for r in data["held_out"]), enabled=True
    )
    assert report["eligible"] > 0 and report["calibration_opportunities"] > 0
