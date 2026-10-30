"""Matched private reports retain failures and separate model/harness effects."""

from dataclasses import replace

import pytest

from evaluation.model_harness_attribution import (
    AttributionObservation,
    AttributionProtocol,
    report_model_harness_attribution,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.private_promotion import ComputeBudget
from gepa_mindfulness.training.eligibility import require_training_eligible

EVALUATOR = TrustedEvaluatorContract("outcomes", "v1", "host-private-protocol")


def protocol():
    return AttributionProtocol(
        protocol_id="factorial:1",
        task_manifest_digest="a" * 64,
        original_model="m0",
        updated_model="m1",
        original_harness="h0",
        candidate_harness="h1",
        task_ids=("task:1", "task:2"),
        seeds=(11, 23),
        budget=ComputeBudget(100, 10, 1000),
        evaluator=EVALUATOR,
    )


def observations():
    return tuple(
        AttributionObservation(
            task_id=task,
            seed=seed,
            model_version=model,
            harness_version=harness,
            task_manifest_digest="a" * 64,
            evaluator=EVALUATOR,
            score=score,
            task_success=score > 0.5,
            protected_pass=True,
            tokens=10,
            tool_calls=1,
            wall_time_ms=10,
            evidence_refs=(
                EvidenceReference(
                    f"outcome:{task}:{seed}:{model}:{harness}", EvidenceSourceKind.EXTERNAL_RECORD
                ),
            ),
        )
        for task in protocol().task_ids
        for seed in protocol().seeds
        for model, harness, score in (
            ("m0", "h0", 0.25),
            ("m0", "h1", 0.5),
            ("m1", "h0", 0.5),
            ("m1", "h1", 0.875),
        )
    )


def report(rows=None):
    return report_model_harness_attribution(protocol(), rows or observations(), enabled=True)


def test_complete_matched_effects_are_private_diagnostics_not_admission():
    result = report()
    assert result["paired_samples"] == 4
    assert result["effects"] == {
        "model_only": 0.25,
        "harness_only": 0.25,
        "combined": 0.625,
        "interaction": 0.125,
    }
    assert len(result["paired_rows"]) == 4
    assert len(result["per_seed_effects"]) == 2
    assert result["authority_granted"] is False
    assert result["promote_candidate"] is False
    with pytest.raises(ValueError, match="HIDDEN_EVAL"):
        require_training_eligible(result)
    assert AttributionProtocol.from_dict(protocol().to_dict()) == protocol()
    assert AttributionObservation.from_dict(observations()[0].to_dict()) == observations()[0]


def test_failure_and_protected_regression_stay_in_denominator():
    rows = list(observations())
    rows[3] = replace(rows[3], score=0.0, task_success=False, protected_pass=False)
    result = report(tuple(rows))
    assert result["paired_samples"] == 4
    assert result["protected_regressions"] == 1
    assert result["protected_non_regression"] is False
    assert result["cells"]["combined"]["task_successes"] == 3
    assert result["effects"]["combined"] == 0.40625


@pytest.mark.parametrize("enabled", [False, None, 1, "yes"])
def test_disabled_before_inputs(enabled):
    with pytest.raises(ValueError, match="enabled"):
        report_model_harness_attribution(None, None, enabled=enabled)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "model",
        "harness",
        "manifest",
        "evaluator",
        "budget",
        "seed",
        "task",
        "reused_evidence",
    ],
)
def test_incomplete_or_unmatched_experiments_fail_closed(fault):
    rows = list(observations())
    if fault == "missing":
        rows.pop()
    if fault == "duplicate":
        rows[-1] = rows[0]
    if fault == "model":
        rows[0] = replace(rows[0], model_version="other")
    if fault == "harness":
        rows[0] = replace(rows[0], harness_version="other")
    if fault == "manifest":
        rows[0] = replace(rows[0], task_manifest_digest="b" * 64)
    if fault == "evaluator":
        rows[0] = replace(rows[0], evaluator=replace(EVALUATOR, evaluator_version="v2"))
    if fault == "budget":
        rows[0] = replace(rows[0], tokens=91)
    if fault == "seed":
        rows[0] = replace(rows[0], seed=2)
    if fault == "task":
        rows[0] = replace(rows[0], task_id="unknown")
    if fault == "reused_evidence":
        rows[-1] = replace(rows[-1], evidence_refs=rows[0].evidence_refs)
    with pytest.raises(ValueError):
        report(tuple(rows))


@pytest.mark.parametrize(
    "field,value",
    [("score", float("nan")), ("score", 1.1), ("score", True), ("tokens", -1), ("seed", True)],
)
def test_nonfinite_or_malformed_observations(field, value):
    with pytest.raises(ValueError):
        replace(observations()[0], **{field: value})


def test_timeout_and_unknown_protected_result_remain_visible():
    rows = list(observations())
    rows[3] = replace(
        rows[3], execution_status="timeout", score=0, task_success=False, protected_pass=None
    )
    result = report(tuple(rows))
    assert result["paired_samples"] == 4
    assert result["cells"]["combined"]["execution_failures"] == 1
    assert result["protected_unknown"] == 1
    assert result["protected_non_regression"] is False
