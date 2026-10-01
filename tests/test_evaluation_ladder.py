"""Independent ladder, denominator, rare-event and capture-boundary contracts."""

import json
import sys
from dataclasses import replace

import pytest

from evaluation.ladder import Metric, Observation, Probe, Severity, Stage, evaluate_ladder
from evaluation.v5_records import SystemIdentity
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility


def probe(name="p", metric=Metric.FALSE_SUCCESS, severity=Severity.ROUTINE, **kwargs):
    """Create a declared metric opportunity with explicit units."""
    unit = "seconds" if "latency" in metric.value else "fraction"
    return Probe(name, metric, kwargs.pop("cohort", "toy"), severity, kwargs.pop("unit", unit))


def observation(name="p", value=False, outcome=None):
    """Capture an externally recorded host measurement."""
    refs = (EvidenceReference("capture-" + name, EvidenceSourceKind.EXTERNAL_RECORD),)
    return Observation(name, value, refs, outcome)


def report(probes, observations=(), **kwargs):
    """Run the opt-in offline diagnostic API."""
    args = dict(
        protocol_id="toy-v1",
        system=SystemIdentity(0, 7, "model-v1", "harness-v1"),
        evaluator=TrustedEvaluatorContract("host", "v1", "toy-contract"),
        training_eligibility=TrainingEligibility.HIDDEN_EVAL,
        enabled=True,
    )
    args.update(kwargs)
    return evaluate_ladder(tuple(probes), tuple(observations), **args)


def group(result, metric):
    """Read the only cohort/severity/unit group for a metric."""
    return result["metrics"][metric.value]["groups"][0]


def test_stages_are_independent_and_absent_measurements_are_unknown():
    """Representation success cannot populate another competency."""
    result = report([probe(metric=Metric.REPRESENTATION_ACCURACY)], [observation(value=True)])
    assert list(result["stages"]) == [s.value for s in Stage]
    assert result["stages"][Stage.REPRESENTATION.value]["observed"] == 1
    assert result["stages"][Stage.PREDICTION.value]["observed"] == 0
    assert result["metrics"][Metric.DECISIVE_EVENT_RECALL.value]["groups"] == []
    assert "score" not in result
    assert result["mechanism_recovery_established"] is False
    assert result["confers_authority"] is False
    assert result["training_eligibility"] == "HIDDEN_EVAL"


def test_one_catastrophic_failure_cannot_hide_behind_ordinary_successes():
    """A low aggregate failure frequency retains the exact catastrophic capture."""
    probes = [probe(str(i)) for i in range(999)]
    probes.append(probe("disaster", severity=Severity.CATASTROPHIC))
    observations = [observation(str(i)) for i in range(999)] + [observation("disaster", True)]
    result = report(probes, observations)
    assert result["failures"][0]["probe_id"] == "disaster"
    assert len(result["failures"]) == 1
    assert result["severe_observations"] == result["failures"]
    groups = result["metrics"][Metric.FALSE_SUCCESS.value]["groups"]
    assert {g["severity"]: g["rate"] for g in groups} == {"routine": 0.0, "catastrophic": 1.0}


def test_missing_severe_probe_and_zero_decisive_opportunities_are_explicit():
    """The roster, not the observed subset, defines coverage."""
    result = report([probe(severity=Severity.CONSEQUENTIAL)])
    summary = group(result, Metric.FALSE_SUCCESS)
    assert (summary["expected"], summary["observed"], summary["missing"]) == (1, 0, 1)
    assert summary["rate"] is None
    assert summary["missing_ids"] == ["p"]
    assert result["severe_observations"][0]["status"] == "missing"
    assert result["coverage_complete"] is False


def test_probability_calibration_reuses_known_brier_and_ece():
    """Compute hand-checkable calibration and retain confident errors."""
    metric = Metric.PREDICTION_CALIBRATION
    result = report(
        [probe("a", metric), probe("b", metric)],
        [observation("a", 0.8, True), observation("b", 0.9, False)],
    )
    summary = group(result, metric)
    assert summary["brier_score"] == pytest.approx(0.425)
    assert summary["expected_calibration_error"] == pytest.approx(0.55)
    assert summary["false_confidence_count"] == 1
    assert result["failures"][0]["probe_id"] == "b"


def test_distribution_units_cohorts_and_severity_are_never_pooled():
    """Extreme finite residuals retain their tails without overflow or unit mixing."""
    metric = Metric.RESIDUAL
    probes = [probe("a", metric, unit="kelvin"), probe("b", metric, unit="kelvin")]
    probes += [probe("c", metric, unit="meters"), probe("d", metric, cohort="ood")]
    result = report(probes, [observation("a", 1e308), observation("b", -1e308)])
    groups = result["metrics"][metric.value]["groups"]
    assert len(groups) == 3
    kelvin = next(g for g in groups if g["unit"] == "kelvin")
    assert kelvin["mean"] == 0
    assert kelvin["p95"] == kelvin["max_absolute"] == 1e308
    json.dumps(result, allow_nan=False)


def test_censored_latency_is_separate_from_missing_and_completed_latency():
    """An unfinished intervention must not become zero or disappear."""
    metric = Metric.INTERVENTION_LATENCY
    result = report(
        [probe(n, metric) for n in ("done", "pending", "missing")],
        [observation("done", 3.0), observation("pending", None)],
    )
    summary = group(result, metric)
    assert summary["observed"] == 2
    assert summary["missing"] == summary["censored"] == summary["completed"] == 1
    assert summary["mean"] == 3.0
    assert summary["censored_ids"] == ["pending"]
    assert result["coverage_complete"] is False


def test_success_metric_failure_polarity_and_all_catalog_entries():
    """Every metric has an explicit stage, kind and null-safe empty report."""
    result = report([probe(metric=Metric.DECISIVE_EVENT_RECALL)], [observation()])
    assert result["failures"][0]["probe_id"] == "p"
    assert set(result["metrics"]) == {m.value for m in Metric}
    assert group(result, Metric.DECISIVE_EVENT_RECALL)["rate"] == 0


def test_digest_order_invariance_and_detached_reports():
    """Presentation order and caller edits do not change subsequent report identity."""
    probes = [probe("b"), probe("a")]
    observations = [observation("b"), observation("a")]
    first = report(probes, observations)
    second = report(probes[::-1], observations[::-1])
    assert first == second
    first["rows"][0]["evidence_refs"][0]["reference_id"] = "changed"
    assert second == report(probes, observations)
    assert second["protocol_digest"] != report(probes, protocol_id="other")["protocol_digest"]
    assert second["result_digest"] != report(probes)["result_digest"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": False},
        {"enabled": 1},
        {"protocol_id": " "},
        {"training_eligibility": TrainingEligibility.TRAIN},
        {"training_eligibility": "HIDDEN_EVAL"},
        {"system": None},
        {"evaluator": None},
    ],
)
def test_invalid_report_contract_fails_closed(kwargs):
    """Neither truthy enablement nor loose metadata can enable the diagnostic."""
    with pytest.raises(ValueError):
        report([probe()], **kwargs)


@pytest.mark.parametrize(
    "probes,observations",
    [
        ([], []),
        ([probe(), probe()], []),
        ([probe()], [observation("unknown")]),
        ([probe()], [observation(), observation()]),
    ],
)
def test_roster_duplicates_and_unknown_observations_are_rejected(probes, observations):
    """A duplicate or unregistered measurement cannot reweight results."""
    with pytest.raises(ValueError):
        report(probes, observations)


@pytest.mark.parametrize(
    "value,outcome,metric",
    [
        (1, None, Metric.FALSE_SUCCESS),
        (None, None, Metric.FALSE_SUCCESS),
        (True, False, Metric.FALSE_SUCCESS),
        (0.2, None, Metric.OOD_CALIBRATION),
        (True, True, Metric.OOD_CALIBRATION),
        (1.1, True, Metric.OOD_CALIBRATION),
        (float("nan"), True, Metric.OOD_CALIBRATION),
        (-1, None, Metric.INTERVENTION_LATENCY),
        (True, None, Metric.RESIDUAL),
        (float("inf"), None, Metric.RESIDUAL),
        (-0.1, None, Metric.UPDATE_MAGNITUDE),
    ],
)
def test_metric_values_are_strict_and_finite(value, outcome, metric):
    """Inputs obey metric-specific numeric and boolean contracts."""
    with pytest.raises(ValueError):
        report([probe(metric=metric)], [observation(value=value, outcome=outcome)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("probe_id", ""),
        ("metric", "false_success"),
        ("cohort", " "),
        ("severity", "routine"),
        ("unit", "meters"),
    ],
)
def test_invalid_probe_metadata_is_rejected(field, value):
    """A roster has explicit, typed metric semantics and units."""
    with pytest.raises(ValueError):
        report([replace(probe(), **{field: value})])


def test_private_evidence_and_tampered_nested_records_are_rejected():
    """Canonical revalidation rejects mutated frozen records and private capture kinds."""
    capture = observation()
    capture.evidence_refs[0].__dict__["reference_id"] = ""
    with pytest.raises(ValueError):
        report([probe()], [capture])
    with pytest.raises(ValueError):
        report(
            [probe()],
            [
                replace(
                    observation(),
                    evidence_refs=(
                        EvidenceReference("private", EvidenceSourceKind.PRIVATE_REASONING),
                    ),
                )
            ],
        )
    contract = TrustedEvaluatorContract("host", "v1", "contract")
    contract.__dict__["contract_id"] = ""
    with pytest.raises(ValueError):
        report([probe()], evaluator=contract)


def test_subclasses_cannot_override_validation():
    """Exact record types prevent subclass methods from bypassing validation."""

    class FakeContract(TrustedEvaluatorContract):
        def __post_init__(self):
            """Simulate a hostile validation override."""

    with pytest.raises(ValueError):
        report([probe()], evaluator=FakeContract("", "", ""))


@pytest.mark.parametrize("metric", list(Metric))
def test_every_metric_accepts_a_measurement_and_an_unmeasured_group(metric):
    """Exercise every declared metric with measured and missing opportunities."""
    if metric in (Metric.PREDICTION_CALIBRATION, Metric.SOURCE_CALIBRATION, Metric.OOD_CALIBRATION):
        value, outcome = 0.1, True
    elif metric in (Metric.RESIDUAL, Metric.UPDATE_MAGNITUDE):
        value, outcome = 2.0, None
    elif "latency" in metric.value:
        value, outcome = 0.0, None
    else:
        value, outcome = True, None
    result = report(
        [probe("p", metric), probe("absent", metric, cohort="absent")],
        [observation(value=value, outcome=outcome)],
    )
    groups = result["metrics"][metric.value]["groups"]
    assert sum(g["observed"] for g in groups) == 1
    assert sum(g["missing"] for g in groups) == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("refs", [(), [], ("capture",)])
def test_invalid_evidence_container(refs):
    """Evidence must be a nonempty tuple of validated records."""
    with pytest.raises(ValueError):
        report([probe()], [replace(observation(), evidence_refs=refs)])


def test_duplicate_evidence_and_unrepresentable_integer_are_rejected():
    """Repeated references and numbers outside floating range fail explicitly."""
    capture = observation()
    with pytest.raises(ValueError):
        report([probe()], [replace(capture, evidence_refs=capture.evidence_refs * 2)])
    with pytest.raises(ValueError):
        report([probe(metric=Metric.RESIDUAL)], [observation(value=10**1000)])


def test_existing_relation_evaluator_adapts_without_claiming_mechanism_recovery():
    """An actual paired audit supplies a behavior-only measurement to the ladder."""
    from hashlib import sha256

    from evaluation.relation_flips import BehaviorObservation, evaluate_relation_suite
    from synthetic_data.relation_flips import Relation, make_relation_pair, render_probe

    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    captures = tuple(
        BehaviorObservation(
            pair.digest,
            arm,
            sha256(render_probe(pair, arm, enabled=True).encode()).hexdigest(),
            SystemIdentity(0, 7, "model-v1", "harness-v1"),
            pair.expected(arm),
            (EvidenceReference(arm, EvidenceSourceKind.OBSERVABLE_ACTION),),
        )
        for arm in ("before", "after")
    )
    source = evaluate_relation_suite((pair,), captures, enabled=True)
    result = report(
        [probe(metric=Metric.RELATION_FLIP_SENSITIVITY)],
        [
            Observation(
                "p",
                source["pairs"][0]["correct_response_to_intervention"],
                tuple(c.evidence_refs[0] for c in captures),
            )
        ],
        training_eligibility=TrainingEligibility.DEVELOPMENT,
    )
    assert group(result, Metric.RELATION_FLIP_SENSITIVITY)["rate"] == 1
    assert result["mechanism_recovery_established"] is False
    assert len(result["rows"][0]["evidence_refs"]) == 2


def test_largest_finite_measurements_have_a_finite_accurate_mean():
    """Valid finite inputs must not overflow because their sum exceeds float range."""
    metric = Metric.RESIDUAL
    probes = [probe(str(i), metric) for i in range(3)]
    captures = [observation(str(i), sys.float_info.max) for i in range(3)]
    result = report(probes, captures)
    assert group(result, metric)["mean"] == sys.float_info.max
    captures = [observation("0", 1e308), observation("1", -1e308), observation("2", 1.0)]
    assert group(report(probes, captures), metric)["mean"] == pytest.approx(1 / 3)
