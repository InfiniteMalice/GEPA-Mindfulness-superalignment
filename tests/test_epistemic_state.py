"""Temporal record contracts remain diagnostic and fail closed on malformed inputs."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_state import (
    Availability,
    CorrelationTreatment,
    DiagonalState,
    EpistemicContext,
    EpistemicMeasurement,
    EpistemicStateEstimate,
    InnovationRecord,
    MismatchStatus,
    UncertaintyUpdateRecord,
)
from mindful_trace_gepa.action_bound_events import PredictionCommit, make_prediction_commit_event
from mindful_trace_gepa.confidence import ConfidenceSource
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion, validate_action_bound_sequence

EVIDENCE = EvidenceReference("evidence-1", EvidenceSourceKind.EXTERNAL_RECORD)
CONTEXT = EpistemicContext("run-1", None, EvaluatedSystemVersion("model-1", "harness-1"))


def estimate(**changes: object) -> EpistemicStateEstimate:
    values = dict(
        estimate_id="prior",
        context=CONTEXT,
        estimator_version="declared-v1",
        world_uncertainty=0.7,
        model_uncertainty=None,
        monitor_uncertainty=0.3,
        evidence_refs=(EVIDENCE,),
        provenance=("producer-1",),
        state=DiagonalState("temperature-v1", ("temperature",), (20.0,), (4.0,)),
    )
    return EpistemicStateEstimate(**(values | changes))


def measurement(**changes: object) -> EpistemicMeasurement:
    values = dict(
        measurement_id="measurement-1",
        context=CONTEXT,
        source=ConfidenceSource.TOOL_RESULT,
        target_dimension="temperature",
        representation_id="temperature-v1",
        value=22.0,
        uncertainty=0.2,
        variance=1.0,
        evidence_refs=(EVIDENCE,),
        provenance=("sensor-1",),
    )
    return EpistemicMeasurement(**(values | changes))


def innovation(**changes: object) -> InnovationRecord:
    values = dict(
        innovation_id="innovation-1",
        context=CONTEXT,
        prediction_commit_id="prediction-1",
        observation_id="observation-1",
        predicted_measurement=20.0,
        actual_measurement=22.0,
        evidence_refs=(EVIDENCE,),
        provenance=("comparison-1",),
    )
    return InnovationRecord(**(values | changes))


def update(**changes: object) -> UncertaintyUpdateRecord:
    values = dict(
        update_id="update-1",
        prior_state=estimate(),
        measurements=(measurement(),),
        posterior_state=estimate(estimate_id="posterior", world_uncertainty=0.5),
        update_method="external-diagnostic-v1",
        evidence_refs=(EVIDENCE,),
        provenance=("estimator-1",),
    )
    return UncertaintyUpdateRecord(**(values | changes))


@pytest.mark.parametrize("factory", [estimate, measurement, innovation, update])
def test_exact_json_round_trip(factory) -> None:
    record = factory()
    payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
    assert type(record).from_dict(payload) == record
    for key in payload:
        malformed = dict(payload)
        del malformed[key]
        with pytest.raises(ValueError):
            type(record).from_dict(malformed)
    with pytest.raises(ValueError):
        type(record).from_dict(payload | {"authorized": True})


@pytest.mark.parametrize("value", [True, "0.5", float("nan"), float("inf"), -0.1, 1.1, 10**400])
@pytest.mark.parametrize("field", ["world_uncertainty", "model_uncertainty", "monitor_uncertainty"])
def test_uncertainties_reject_invalid_numbers(value, field) -> None:
    with pytest.raises(ValueError):
        estimate(**{field: value})


@pytest.mark.parametrize("factory", [estimate, measurement, innovation, update])
@pytest.mark.parametrize("provenance", [(), "producer", ("",), (True,), {"producer": 1}])
def test_provenance_is_required_and_typed(factory, provenance) -> None:
    with pytest.raises(ValueError):
        factory(provenance=provenance)


@pytest.mark.parametrize("factory", [estimate, measurement, innovation, update])
def test_evidence_is_typed_and_not_relabelled(factory) -> None:
    with pytest.raises(ValueError):
        factory(evidence_refs=("evidence-1",))
    internal = EvidenceReference("latent-1", EvidenceSourceKind.LATENT_STATE)
    if factory is not update:
        record = factory(evidence_refs=(internal,))
        assert not record.evidence_refs[0].is_observable


def test_unavailable_is_null_and_never_zero() -> None:
    missing = estimate(
        status=Availability.UNAVAILABLE,
        world_uncertainty=None,
        model_uncertainty=None,
        monitor_uncertainty=None,
        state=None,
        evidence_refs=(),
    )
    assert missing.to_dict()["world_uncertainty"] is None
    with pytest.raises(ValueError):
        replace(missing, world_uncertainty=0.0)
    absent = measurement(
        status=Availability.UNAVAILABLE,
        value=None,
        uncertainty=None,
        variance=None,
        evidence_refs=(),
    )
    assert absent.to_dict()["value"] is None
    with pytest.raises(ValueError):
        replace(absent, value=0.0)
    assert measurement(value=0.0, uncertainty=0.0, variance=0.0).value == 0.0
    assert measurement(variance=None).variance is None
    with pytest.raises(ValueError):
        measurement(value=None)


@pytest.mark.parametrize(
    "variances",
    [(-1.0,), (True,), (float("nan"),), (), (1.0, 2.0), ((1.0,),), ((1.0, 2.0), (2.0, 1.0))],
)
def test_malformed_diagonal_covariance_fails_closed(variances) -> None:
    with pytest.raises(ValueError):
        DiagonalState("v1", ("x",), (1.0,), variances)


def test_diagonal_shape_and_immutable_snapshots() -> None:
    dimensions, values, variances = ["x"], [1.0], [0.0]
    state = DiagonalState("v1", dimensions, values, variances)
    dimensions[0], values[0], variances[0] = "y", 9.0, 5.0
    assert state.dimensions == ("x",)
    assert state.values == (1.0,) and state.variances == (0.0,)
    for dims, vals in [((), ()), (("x", "x"), (1.0, 2.0)), (("x",), (1.0, 2.0))]:
        with pytest.raises(ValueError):
            DiagonalState("v1", dims, vals, None)
    with pytest.raises(FrozenInstanceError):
        state.values = (9.0,)
    refs, provenance = [EVIDENCE], ["origin"]
    record = estimate(evidence_refs=refs, provenance=provenance)
    refs.clear()
    provenance.clear()
    assert record.evidence_refs == (EVIDENCE,) and record.provenance == ("origin",)


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", True),
        ("value", float("inf")),
        ("variance", -1.0),
        ("variance", True),
        ("uncertainty", 2.0),
        ("source", "TOOL_RESULT"),
        ("status", "available"),
        ("correlation_group", ""),
    ],
)
def test_measurements_validate_fields(field, value) -> None:
    with pytest.raises(ValueError):
        measurement(**{field: value})


def test_innovation_normalization_requires_explicit_basis() -> None:
    raw = innovation()
    assert raw.residual == 2.0 and raw.normalized_innovation is None
    normalized = innovation(innovation_variance=4.0, normalization_basis="linear-model-v1")
    assert normalized.normalized_innovation == 1.0
    for changes in [
        dict(innovation_variance=4.0),
        dict(normalization_basis="assumed"),
        dict(innovation_variance=0.0, normalization_basis="zero"),
        dict(predicted_measurement=True),
        dict(actual_measurement=float("nan")),
        dict(predicted_measurement=-1e308, actual_measurement=1e308),
    ]:
        with pytest.raises(ValueError):
            innovation(**changes)


def test_update_preserves_unresolved_correlation_and_hypotheses() -> None:
    record = update(
        unresolved_hypotheses=("sensor-drift", "world-change"),
        model_mismatch=MismatchStatus.MODEL_MISMATCH,
    )
    assert record.correlation_treatment is CorrelationTreatment.UNRESOLVED_CORRELATION
    assert record.measurements[0].correlation_group is None
    assert record.unresolved_hypotheses == ("sensor-drift", "world-change")
    assert UncertaintyUpdateRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError):
        update(measurements=(measurement(), measurement()))
    with pytest.raises(ValueError):
        update(posterior_state=estimate())
    with pytest.raises(ValueError):
        update(measurements=())
    with pytest.raises(ValueError):
        update(evidence_refs=())


@pytest.mark.parametrize(
    "context",
    [
        EpistemicContext("run-2", None, CONTEXT.system),
        EpistemicContext("run-1", 0, CONTEXT.system),
        EpistemicContext("run-1", None, EvaluatedSystemVersion("model-2", "harness-1")),
        EpistemicContext("run-1", None, EvaluatedSystemVersion("model-1", "harness-2")),
    ],
)
def test_update_rejects_evaluation_identity_drift(context) -> None:
    with pytest.raises(ValueError, match="context"):
        update(measurements=(measurement(context=context),))
    with pytest.raises(ValueError, match="context"):
        update(posterior_state=estimate(estimate_id="posterior", context=context))


def test_update_rejects_dimension_and_representation_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension"):
        update(measurements=(measurement(target_dimension="pressure"),))
    with pytest.raises(ValueError, match="representation"):
        update(measurements=(measurement(representation_id="fahrenheit-v1"),))
    with pytest.raises(ValueError):
        update(
            posterior_state=estimate(
                estimate_id="posterior",
                state=DiagonalState("temperature-v2", ("temperature",), (20.0,), None),
            )
        )


def test_existing_prediction_event_identity_and_json_are_unchanged() -> None:
    prediction = PredictionCommit("prediction-1", {"temperature": 20}, 0.8, ("evidence-1",))
    event = make_prediction_commit_event(
        prediction,
        run_id="run-1",
        model_version="model-1",
        harness_version="harness-1",
    )
    before = event.to_dict()
    CONTEXT.validate_event(event)
    validate_action_bound_sequence((event,))
    assert event.to_dict() == before
    for changes in [
        dict(run_id="other"),
        dict(repeat_id=0),
        dict(model_version="other"),
        dict(harness_version="other"),
    ]:
        with pytest.raises(ValueError, match="identity"):
            CONTEXT.validate_event(replace(event, **changes))


def test_records_do_not_implement_optimizer_or_authority_contracts() -> None:
    from gepa_mindfulness.core.epistemic_process import EpistemicProcessAssessment

    for record in (estimate(), measurement(), innovation(), update()):
        assert not hasattr(record, "optimizer_score")
        assert not hasattr(record, "authorize_action")
        with pytest.raises(ValueError):
            EpistemicProcessAssessment(verified_components=(record,))


@pytest.mark.parametrize("repeat_id", [True, -1, 0.5, "0", 9_007_199_254_740_992])
def test_context_rejects_invalid_repeat_identity(repeat_id) -> None:
    with pytest.raises(ValueError, match="repeat_id"):
        EpistemicContext("run-1", repeat_id, CONTEXT.system)


@pytest.mark.parametrize("factory", [estimate, measurement, innovation, update])
def test_serialized_payload_is_detached_and_version_is_checked(factory) -> None:
    record = factory()
    payload = record.to_dict()
    payload["provenance"].clear()
    payload["evidence_refs"][0]["source_kind"] = "private_reasoning"
    assert record.provenance and record.evidence_refs == (EVIDENCE,)
    for version in ("epistemic-state-v2", True, None):
        with pytest.raises(ValueError, match="schema_version"):
            type(record).from_dict(record.to_dict() | {"schema_version": version})


def test_update_detaches_measurement_list() -> None:
    measurements = [measurement()]
    record = update(measurements=measurements)
    measurements.clear()
    assert len(record.measurements) == 1
