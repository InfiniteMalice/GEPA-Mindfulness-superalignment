"""Deterministic numerical and causal checks for the experimental scalar estimator."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import isfinite

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_reconciliation import OutcomeMeasurementBinding
from gepa_mindfulness.verification.epistemic_state import (
    DiagonalState,
    EpistemicContext,
    EpistemicMeasurement,
    EpistemicStateEstimate,
    InnovationRecord,
    MismatchStatus,
)
from gepa_mindfulness.verification.temporal_estimator import (
    ScalarEstimatorConfig,
    ScalarTemporalEstimator,
)
from mindful_trace_gepa.action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
    make_action_event,
    make_outcome_observation_event,
    make_prediction_commit_event,
    make_verification_result_event,
)
from mindful_trace_gepa.confidence import ConfidenceSource
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion, validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

CONTEXT = EpistemicContext("run", 0, EvaluatedSystemVersion("model", "harness"))


def initial(**changes: object) -> EpistemicStateEstimate:
    values = dict(
        estimate_id="initial",
        context=CONTEXT,
        estimator_version="initial-v1",
        world_uncertainty=None,
        model_uncertainty=None,
        monitor_uncertainty=None,
        state=DiagonalState("sensor-units-v1", ("value",), (0,), (1,)),
        evidence_refs=(EvidenceReference("initial-evidence", EvidenceSourceKind.EXTERNAL_RECORD),),
        provenance=("initial-producer",),
    )
    return EpistemicStateEstimate(**(values | changes))


def config(**changes: object) -> ScalarEstimatorConfig:
    return ScalarEstimatorConfig(
        **(
            dict(
                enabled=True,
                independent_noise=True,
                measurement_provenance_ref="sensor-v1",
                process_variance=1.0,
                process_variance_floor=0.01,
                uncertainty_scale=1.0,
            )
            | changes
        )
    )


def inputs(
    estimator: ScalarTemporalEstimator,
    value: float,
    index: int = 0,
    history: tuple[EventEnvelope, ...] = (),
    variance: float = 1.0,
) -> tuple[tuple[EventEnvelope, ...], dict]:
    prior = estimator.estimate
    mean = prior.state.values[0]
    start = datetime(2026, 9, 30, 12, tzinfo=timezone.utc) + timedelta(seconds=index * 10)

    def metadata(name: str, second: int, parents: tuple[str, ...] = ()) -> dict:
        return dict(
            event_id=f"{name}{index}",
            timestamp=(start + timedelta(seconds=second)).isoformat(),
            run_id="run",
            repeat_id=0,
            model_version="model",
            harness_version="harness",
            parent_event_ids=parents,
        )

    evidence = EvidenceReference(f"evidence-{index}", EvidenceSourceKind.EXTERNAL_RECORD)
    action = ActionRecord(f"action-{index}", "read", True, "read-only", f"prediction-{index}")
    prediction = PredictionCommit(
        action.prediction_commit_id,
        {"value": mean},
        0.5,
        tuple(ref.reference_id for ref in prior.evidence_refs),
    )
    observation = OutcomeObservation(
        f"observation-{index}", action.action_id, {"value": value}, (evidence.reference_id,)
    )
    events = (
        *history,
        make_prediction_commit_event(prediction, **metadata("p", 0)),
        make_action_event(
            action, StructuredEventType.ACTION_PROPOSED, **metadata("a", 1, (f"p{index}",))
        ),
        make_action_event(
            action, StructuredEventType.ACTION_EXECUTED, **metadata("x", 2, (f"a{index}",))
        ),
        make_outcome_observation_event(observation, **metadata("o", 3, (f"x{index}",))),
        make_verification_result_event(
            VerificationResult(
                f"verifier-{index}", "v1", observation.observation_id, True, ("verifier-contract",)
            ),
            **metadata("v", 4, (f"o{index}",)),
        ),
    )
    measurement = EpistemicMeasurement(
        measurement_id=f"measurement-{index}",
        context=CONTEXT,
        source=ConfidenceSource.TOOL_RESULT,
        representation_id="sensor-units-v1",
        target_dimension="value",
        value=value,
        uncertainty=None,
        variance=variance,
        evidence_refs=(evidence,),
        provenance=("sensor-v1", "verifier-contract"),
    )
    binding = OutcomeMeasurementBinding(
        measurement_id=measurement.measurement_id,
        innovation=InnovationRecord(
            innovation_id=f"innovation-{index}",
            context=CONTEXT,
            prediction_commit_id=prediction.prediction_commit_id,
            observation_id=observation.observation_id,
            predicted_measurement=mean,
            actual_measurement=value,
            evidence_refs=(evidence,),
            provenance=("comparison-v1",),
        ),
        prediction_path=("value",),
        observation_path=("value",),
        verifier_event_id=f"v{index}",
    )
    return events, dict(
        measurement=measurement,
        binding=binding,
        prediction_event_id=f"p{index}",
        observation_event_id=f"o{index}",
        verification_event_ids=(f"v{index}",),
        update_id=f"update-{index}",
        estimate_id=f"estimate-{index}",
        event_id=f"r{index}",
        timestamp=(start + timedelta(seconds=5)).isoformat(),
    )


def test_disabled_by_default_is_inert() -> None:
    estimator = ScalarTemporalEstimator(initial())
    before = estimator.estimate
    events, kwargs = inputs(estimator, 1)
    assert estimator.reconcile((), **kwargs) is None
    assert estimator.estimate == before
    validate_action_bound_sequence(events)


def test_enable_requires_explicit_noise_assumption() -> None:
    with pytest.raises(ValueError, match="independent_noise"):
        ScalarEstimatorConfig(enabled=True)


def test_hand_computed_joseph_update_and_causal_output() -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 1)
    result = estimator.reconcile(events, **kwargs)
    assert result.assimilated
    assert result.gain == pytest.approx(2 / 3)
    assert result.estimate.state.values == pytest.approx((2 / 3,))
    assert result.estimate.state.variances == pytest.approx((2 / 3,))
    assert result.estimate.world_uncertainty == pytest.approx(0.4)
    assert result.estimate.monitor_uncertainty == pytest.approx(0.5)
    assert result.estimate.model_uncertainty is None
    assert result.reconciliation.update.model_mismatch is MismatchStatus.NONE
    assert result.innovation.innovation_variance == 3
    assert result.innovation.normalized_innovation == pytest.approx(1 / 3**0.5)
    validate_action_bound_sequence((*events, result.event))
    assert estimator.estimate == result.estimate


def test_fixed_noise_covariance_does_not_depend_on_residual_magnitude() -> None:
    variances = []
    for value in (0.1, 100.0):
        estimator = ScalarTemporalEstimator(initial(), config(innovation_gate=1000))
        events, kwargs = inputs(estimator, value)
        variances.append(estimator.reconcile(events, **kwargs).estimate.state.variances)
    assert variances[0] == variances[1]


def test_outlier_inflates_noise_and_covariance_without_assimilating_mean() -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 100)
    result = estimator.reconcile(events, **kwargs)
    assert not result.assimilated
    assert result.gain == 0
    assert result.estimate.state.values == (0,)
    assert result.process_variance == 4
    assert result.measurement_variance == 4
    assert result.estimate.state.variances == (5,)
    assert result.reconciliation.update.model_mismatch is MismatchStatus.MODEL_MISMATCH
    assert result.innovation.innovation_variance == 3  # Gate used pre-adaptation covariance.


def test_persistent_shift_reaches_latched_insufficient_model() -> None:
    estimator = ScalarTemporalEstimator(
        initial(),
        config(
            regime_shift_after=2,
            insufficient_model_after=3,
        ),
    )
    history = ()
    statuses = []
    for index, value in enumerate((1000, 1000, 1000, 0)):
        events, kwargs = inputs(estimator, value, index, history)
        result = estimator.reconcile(events, **kwargs)
        statuses.append(result.reconciliation.update.model_mismatch)
        assert not result.assimilated
        history = (*events, result.event)
    assert statuses == [
        MismatchStatus.MODEL_MISMATCH,
        MismatchStatus.REGIME_SHIFT_SUSPECTED,
        MismatchStatus.INSUFFICIENT_MODEL,
        MismatchStatus.INSUFFICIENT_MODEL,
    ]


def test_inlier_resets_streak_and_decays_adaptive_noise() -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    history = ()
    results = []
    for index, value in enumerate((100, 0, 0)):
        events, kwargs = inputs(estimator, value, index, history)
        result = estimator.reconcile(events, **kwargs)
        history = (*events, result.event)
        results.append(result)
    assert results[1].assimilated and results[1].outlier_count == 0
    assert results[1].process_variance == 4
    assert results[2].process_variance == 2
    assert results[1].measurement_variance == 4
    assert results[2].measurement_variance == 2


@pytest.mark.parametrize("change", ["missing-parent", "false-verifier", "bad-number", "source"])
def test_invalid_step_is_atomic(change: str) -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 100)
    original_events, original_kwargs = events, dict(kwargs)
    if change == "missing-parent":
        events = events[1:]
    elif change == "false-verifier":
        events = (
            *events[:-1],
            replace(
                events[-1],
                payload=dict(events[-1].payload)
                | {
                    "verified": False,
                },
            ),
        )
    elif change == "bad-number":
        kwargs["measurement"] = replace(kwargs["measurement"], value=101)
    else:
        kwargs["measurement"] = replace(kwargs["measurement"], source=ConfidenceSource.FUSED)
    with pytest.raises(ValueError):
        estimator.reconcile(events, **kwargs)
    assert estimator.estimate == initial()
    result = estimator.reconcile(original_events, **original_kwargs)
    assert result.outlier_count == 1
    assert result.process_variance == 4


@pytest.mark.parametrize("verified", [True, False])
def test_estimator_requires_explicit_verifier_binding_before_assimilation(verified: bool) -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 1)
    original_events, original_kwargs = events, dict(kwargs)
    events = (
        *events[:-1],
        replace(
            events[-1],
            payload=dict(events[-1].payload)
            | {
                "verified": verified,
            },
        ),
    )
    kwargs["binding"] = replace(kwargs["binding"], verifier_event_id=None)
    with pytest.raises(ValueError, match="verifier.*binding"):
        estimator.reconcile(events, **kwargs)
    assert estimator.estimate == initial()
    assert estimator.reconcile(original_events, **original_kwargs).assimilated


@pytest.mark.parametrize(
    "field",
    [
        "process_variance",
        "innovation_gate",
        "uncertainty_scale",
        "process_variance_floor",
        "process_noise_growth",
    ],
)
@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf"), -1, 10**400])
def test_numeric_configuration_fails_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        config(**{field: value})


@pytest.mark.parametrize(
    "state",
    [
        None,
        DiagonalState("sensor-units-v1", ("value",), (0,), None),
        DiagonalState("sensor-units-v1", ("value", "other"), (0, 0), (1, 1)),
    ],
)
def test_only_scalar_with_available_covariance_is_supported(state: object) -> None:
    with pytest.raises(ValueError):
        ScalarTemporalEstimator(initial(state=state, world_uncertainty=0.5), config())


def test_history_must_retain_prior_reconciliation_and_cannot_replay() -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 1)
    result = estimator.reconcile(events, **kwargs)
    before = estimator.estimate
    with pytest.raises(ValueError):
        estimator.reconcile(events, **kwargs)
    events2, kwargs2 = inputs(estimator, 2, 1)
    with pytest.raises(ValueError, match="history"):
        estimator.reconcile(events2, **kwargs2)
    assert estimator.estimate == before
    events2, kwargs2 = inputs(estimator, 2, 1, (*events, result.event))
    assert estimator.reconcile(events2, **kwargs2).assimilated


@pytest.mark.parametrize("variance", [None, 0])
def test_measurement_variance_must_be_available_and_positive(variance: object) -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 1)
    kwargs["measurement"] = replace(kwargs["measurement"], variance=variance)
    with pytest.raises(ValueError, match="positive finite variance"):
        estimator.reconcile(events, **kwargs)
    assert estimator.estimate == initial()


@pytest.mark.parametrize("change", ["context", "dimension", "representation", "provenance"])
def test_measurement_contract_must_match(change: str) -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 1)
    changes = {
        "context": {"context": replace(CONTEXT, run_id="another-run")},
        "dimension": {"target_dimension": "other"},
        "representation": {"representation_id": "other"},
        "provenance": {"provenance": ("another-sensor",)},
    }
    kwargs["measurement"] = replace(kwargs["measurement"], **changes[change])
    with pytest.raises(ValueError):
        estimator.reconcile(events, **kwargs)
    assert estimator.estimate == initial()


def test_overflow_rolls_back_before_a_valid_retry() -> None:
    estimator = ScalarTemporalEstimator(
        initial(), config(process_variance=1e308, max_process_variance=1e308)
    )
    events, kwargs = inputs(estimator, 1, variance=1e308)
    with pytest.raises(ValueError, match="innovation variance"):
        estimator.reconcile(events, **kwargs)
    assert estimator.estimate == initial()
    events, kwargs = inputs(estimator, 1)
    result = estimator.reconcile(events, **kwargs)
    assert result.assimilated and result.outlier_count == 0
    assert result.estimate.state.variances == (1,)


def test_saturated_process_noise_cannot_underflow_to_zero_on_outlier() -> None:
    estimator = ScalarTemporalEstimator(
        initial(state=DiagonalState("sensor-units-v1", ("value",), (0,), (0,))),
        config(
            process_variance=1e-300,
            process_variance_floor=1e-300,
            max_process_variance=1e-300,
            process_noise_growth=1e100,
        ),
    )
    events, kwargs = inputs(estimator, 1, variance=1e-300)
    result = estimator.reconcile(events, **kwargs)
    assert not result.assimilated
    assert result.process_variance == 1e-300
    assert result.estimate.state.variances == (1e-300,)


@pytest.mark.parametrize("change", ["prefix", "timestamp", "measurement", "evidence", "estimate"])
def test_temporal_tampering_and_identity_reuse_are_atomic(change: str) -> None:
    estimator = ScalarTemporalEstimator(initial(), config())
    first_events, first_kwargs = inputs(estimator, 1)
    first = estimator.reconcile(first_events, **first_kwargs)
    history = (*first_events, first.event)
    events, kwargs = inputs(estimator, 2, 1, history)
    original_events, original_kwargs = events, dict(kwargs)
    if change == "prefix":
        events = (replace(events[0], timestamp="2026-09-30T11:59:59Z"), *events[1:])
    elif change == "timestamp":
        events, kwargs = inputs(estimator, 2, -1, history)
    elif change == "measurement":
        kwargs["measurement"] = replace(kwargs["measurement"], measurement_id="measurement-0")
    elif change == "evidence":
        kwargs["measurement"] = replace(
            kwargs["measurement"], evidence_refs=first_kwargs["measurement"].evidence_refs
        )
    else:
        kwargs["estimate_id"] = "initial"
    with pytest.raises(ValueError):
        estimator.reconcile(events, **kwargs)
    assert estimator.estimate == first.estimate
    assert estimator.reconcile(original_events, **original_kwargs).assimilated


def test_scalar_zero_covariance_is_valid_and_configuration_round_trips() -> None:
    cfg = config(process_variance=0)
    assert ScalarEstimatorConfig.from_dict(cfg.to_dict()) == cfg
    estimator = ScalarTemporalEstimator(
        initial(state=DiagonalState("sensor-units-v1", ("value",), (0,), (0,))), cfg
    )
    events, kwargs = inputs(estimator, 0)
    result = estimator.reconcile(events, **kwargs)
    assert result.gain == 0 and result.estimate.state.variances == (0,)
    assert result.estimate.world_uncertainty == 0


def test_caller_snapshots_do_not_mutate_estimator_state_or_configuration() -> None:
    cfg = config()
    estimator = ScalarTemporalEstimator(initial(), cfg)
    object.__setattr__(cfg, "enabled", False)
    snapshot = estimator.estimate
    object.__setattr__(snapshot.state, "values", (999,))
    events, kwargs = inputs(estimator, 1)
    result = estimator.reconcile(events, **kwargs)
    expected = estimator.estimate
    object.__setattr__(result.estimate.state, "values", (999,))
    object.__setattr__(result.configuration, "process_variance", 999)
    assert estimator.estimate == expected
    events, kwargs = inputs(estimator, 1, 1, (*events, result.event))
    assert estimator.reconcile(events, **kwargs).process_variance == 1


def test_synthetic_matched_then_shift_comparison() -> None:
    runs = []
    for gate in (1_000_000, 3):
        estimator = ScalarTemporalEstimator(
            initial(),
            config(innovation_gate=gate, regime_shift_after=2, insufficient_model_after=3),
        )
        history = ()
        results = []
        for index, value in enumerate((0, 0, 1000, 1000, 1000)):
            events, kwargs = inputs(estimator, value, index, history)
            result = estimator.reconcile(events, **kwargs)
            history = (*events, result.event)
            results.append(result)
            assert all(isfinite(v) and v >= 0 for v in result.estimate.state.variances)
        runs.append(results)
    baseline, gated = runs
    assert baseline[1].estimate.state == gated[1].estimate.state
    assert baseline[-1].estimate.state.values[0] > 900
    assert baseline[-1].estimate.state.variances[0] < 1
    assert gated[-1].estimate.state.values == (0,)
    assert gated[-1].estimate.state.variances == pytest.approx((84.625,))
    assert gated[-1].reconciliation.update.model_mismatch is MismatchStatus.INSUFFICIENT_MODEL
    for index, (left, right) in enumerate(zip(baseline, gated)):
        print(
            index,
            left.estimate.state.values[0],
            left.estimate.state.variances[0],
            right.estimate.state.values[0],
            right.estimate.state.variances[0],
            right.reconciliation.update.model_mismatch.value,
        )
