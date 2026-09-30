"""Reconciliation binds diagnostic numbers to the recorded causal trajectory."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from evaluation import validate_v5_record_provenance
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_reconciliation import (
    EpistemicReconciliation,
    OutcomeMeasurementBinding,
    make_epistemic_reconciliation_event,
)
from gepa_mindfulness.verification.epistemic_state import (
    EpistemicContext,
    EpistemicMeasurement,
    EpistemicStateEstimate,
    InnovationRecord,
    UncertaintyUpdateRecord,
)
from gepa_mindfulness.verification.interfaces import (
    LocalVerificationResult,
    RelationalVerificationResult,
    VerificationEvidenceBinding,
    make_local_verification_event,
    make_relational_verification_event,
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
EVIDENCE = EvidenceReference("sensor-log", EvidenceSourceKind.EXTERNAL_RECORD)


def reconciliation() -> EpistemicReconciliation:
    prior = EpistemicStateEstimate(
        estimate_id="prior",
        context=CONTEXT,
        estimator_version="diagnostic-v1",
        world_uncertainty=0.6,
        model_uncertainty=None,
        monitor_uncertainty=None,
        evidence_refs=(EVIDENCE,),
        provenance=("producer",),
    )
    measurement = EpistemicMeasurement(
        measurement_id="measurement",
        context=CONTEXT,
        source=ConfidenceSource.EXTERNAL_VERIFIER,
        target_dimension="temperature",
        representation_id="celsius-v1",
        value=22,
        uncertainty=None,
        evidence_refs=(EVIDENCE,),
        provenance=("verifier-log",),
    )
    update = UncertaintyUpdateRecord(
        update_id="update",
        prior_state=prior,
        measurements=(measurement,),
        posterior_state=replace(
            prior, estimate_id="posterior", action_id="action", prediction_commit_id="prediction"
        ),
        update_method="recorded-comparison",
        evidence_refs=(EVIDENCE,),
        provenance=("producer",),
    )
    innovation = InnovationRecord(
        innovation_id="innovation",
        context=CONTEXT,
        prediction_commit_id="prediction",
        observation_id="observation",
        predicted_measurement=20,
        actual_measurement=22,
        evidence_refs=(EVIDENCE,),
        provenance=("comparison",),
    )
    return EpistemicReconciliation(
        update=update,
        prediction_event_id="p",
        observation_event_id="o",
        verification_event_ids=("v",),
        bindings=(
            OutcomeMeasurementBinding(
                measurement_id="measurement",
                innovation=innovation,
                prediction_path=("temperature", 0),
                observation_path=("temperature", 0),
                verifier_event_id="v",
            ),
        ),
    )


def metadata(event_id: str, second: int, parents: tuple[str, ...] = ()) -> dict:
    return dict(
        event_id=event_id,
        timestamp=f"2026-09-30T12:00:{second:02d}Z",
        run_id="run",
        repeat_id=0,
        model_version="model",
        harness_version="harness",
        parent_event_ids=parents,
    )


def sequence(record: EpistemicReconciliation | None = None) -> list[EventEnvelope]:
    action = ActionRecord("action", "read_sensor", True, "read-only", "prediction")
    return [
        make_prediction_commit_event(
            PredictionCommit("prediction", {"temperature": [20]}, 0.7, ("sensor-log",)),
            **metadata("p", 0),
        ),
        make_action_event(action, StructuredEventType.ACTION_PROPOSED, **metadata("a", 1, ("p",))),
        make_action_event(action, StructuredEventType.ACTION_EXECUTED, **metadata("x", 2, ("a",))),
        make_outcome_observation_event(
            OutcomeObservation("observation", "action", {"temperature": [22]}, ("sensor-log",)),
            **metadata("o", 3, ("x",)),
        ),
        make_verification_result_event(
            VerificationResult("verifier", "v1", "observation", True, ("verifier-log",)),
            **metadata("v", 4, ("o",)),
        ),
        make_epistemic_reconciliation_event(
            record or reconciliation(),
            event_id="r",
            timestamp="2026-09-30T12:00:05Z",
        ),
    ]


def test_happy_path_round_trip_and_world_residual() -> None:
    record = reconciliation()
    assert EpistemicReconciliation.from_dict(json.loads(json.dumps(record.to_dict()))) == record
    events = sequence(record)
    validate_action_bound_sequence(events)
    assert record.bindings[0].innovation.residual == 2
    assert events[-1].parent_event_ids == ("p", "o", "v")
    assert events[-1].action_id == "action"
    validate_action_bound_sequence([EventEnvelope(**e.to_dict()) for e in events])


@pytest.mark.parametrize("index", range(5))
def test_missing_causal_input_fails(index: int) -> None:
    events = sequence()
    del events[index]
    with pytest.raises(ValueError):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize("index", range(5))
def test_reconciliation_cannot_precede_inputs(index: int) -> None:
    events = sequence()
    events.insert(index, events.pop())
    with pytest.raises(ValueError):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    "index,timestamp",
    [
        (0, "2026-09-30T12:00:02Z"),
        (2, "2026-09-30T11:59:59Z"),
        (3, "2026-09-30T12:00:01Z"),
        (4, "2026-09-30T12:00:02Z"),
        (5, "2026-09-30T12:00:03Z"),
        (0, "2026-09-30T12:00:00"),
    ],
)
def test_timestamp_chronology_fails(index: int, timestamp: str) -> None:
    events = sequence()
    events[index] = replace(events[index], timestamp=timestamp)
    with pytest.raises(ValueError):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_id", "other"),
        ("run_id", "other"),
        ("repeat_id", 1),
        ("model_version", "other"),
        ("harness_version", "other"),
        ("parent_event_ids", ("o", "v")),
        ("evidence_refs", ("fabricated",)),
    ],
)
def test_envelope_cannot_override_payload(field: str, value: object) -> None:
    events = sequence()
    events[-1] = replace(events[-1], **{field: value})
    with pytest.raises(ValueError):
        validate_action_bound_sequence(events)
    with pytest.raises(ValueError):
        make_epistemic_reconciliation_event(reconciliation(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("predicted_measurement", 19),
        ("actual_measurement", 23),
        ("prediction_commit_id", "other"),
        ("observation_id", "other"),
    ],
)
def test_innovation_cannot_rewrite_recorded_outcome(field: str, value: object) -> None:
    record = reconciliation()
    binding = record.bindings[0]
    with pytest.raises(ValueError):
        altered = replace(binding, innovation=replace(binding.innovation, **{field: value}))
        validate_action_bound_sequence(sequence(replace(record, bindings=(altered,))))


@pytest.mark.parametrize("path", [("missing",), ("temperature", "0"), ("temperature", 8)])
def test_outcome_paths_fail_closed(path: tuple) -> None:
    record = reconciliation()
    record = replace(record, bindings=(replace(record.bindings[0], prediction_path=path),))
    with pytest.raises(ValueError):
        validate_action_bound_sequence(sequence(record))


def test_external_label_requires_successful_verifier() -> None:
    record = reconciliation()
    with pytest.raises(ValueError):
        absent = replace(record, bindings=(replace(record.bindings[0], verifier_event_id=None),))
        validate_action_bound_sequence(sequence(absent))
    events = sequence()
    events[4] = replace(events[4], payload=dict(events[4].payload) | {"verified": False})
    with pytest.raises(ValueError):
        validate_action_bound_sequence(events)


def test_predictions_and_reconciliations_are_append_only() -> None:
    events = sequence()
    with pytest.raises(ValueError, match="duplicate prediction_commit_id"):
        validate_action_bound_sequence(events + [replace(events[0], event_id="p-again")])
    with pytest.raises(ValueError, match="duplicate.*update"):
        validate_action_bound_sequence(events + [replace(events[-1], event_id="r-again")])
    with pytest.raises(ValueError, match="superseded_by"):
        validate_action_bound_sequence(
            events[:-1]
            + [replace(events[-1], superseded_by="r-next"), replace(events[-1], event_id="r-next")]
        )


def test_payload_is_detached_and_frozen() -> None:
    event = sequence()[-1]
    detached = event.to_dict()
    detached["payload"]["update"]["measurements"][0]["value"] = 999
    assert event.payload["update"]["measurements"][0]["value"] == 22
    with pytest.raises(TypeError):
        event.payload["update"]["measurements"][0]["value"] = 999


def relational_verifier(**changes: object) -> EventEnvelope:
    values = dict(
        action_id="action",
        task_fit=False,
        dependencies_satisfied=False,
        contradiction_status="unknown",
        provenance_intact=True,
        authorization_scope_valid=False,
        claimed_outcome_supported=True,
        repeated_failed_route=False,
        evidence_refs=(EVIDENCE,),
        evidence_bindings=tuple(
            VerificationEvidenceBinding(name, (EVIDENCE,))
            for name in ("provenance_intact", "claimed_outcome_supported")
        ),
    )
    return make_relational_verification_event(
        RelationalVerificationResult(**(values | changes)),
        verifier_refs=("verifier-log",),
        **metadata("v", 4, ("o",)),
    )


def test_relational_verifier_field_evidence() -> None:
    events = sequence()
    events[4] = relational_verifier()
    validate_action_bound_sequence(events)
    unrelated = EvidenceReference("unrelated", EvidenceSourceKind.EXTERNAL_RECORD)
    for field in ("provenance_intact", "claimed_outcome_supported"):
        events[4] = relational_verifier(**{field: False})
        with pytest.raises(ValueError, match="supported outcome"):
            validate_action_bound_sequence(events)
        events[4] = relational_verifier(
            evidence_refs=(EVIDENCE, unrelated),
            evidence_bindings=tuple(
                VerificationEvidenceBinding(name, (unrelated if name == field else EVIDENCE,))
                for name in ("provenance_intact", "claimed_outcome_supported")
            ),
        )
        with pytest.raises(ValueError, match="bind verifier field"):
            validate_action_bound_sequence(events)


def test_local_execution_is_not_outcome_verification() -> None:
    result = LocalVerificationResult(
        action_id="action",
        executed=False,
        arguments_valid=False,
        schema_valid=False,
        authorization_valid=False,
        intended_operation_observed=False,
        irreversible_action_permitted=None,
        evidence_refs=(),
    )
    events = sequence()
    events[4] = make_local_verification_event(
        result,
        verifier_refs=("verifier-log",),
        **metadata("v", 4, ("o",)),
    )
    with pytest.raises(ValueError, match="local execution"):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize("change", ["provenance", "private", "unrecorded", "kind"])
def test_external_measurement_cannot_launder_evidence(change: str) -> None:
    record = reconciliation()
    measurement = record.update.measurements[0]
    evidence = EVIDENCE
    if change == "private":
        evidence = replace(evidence, source_kind=EvidenceSourceKind.PRIVATE_REASONING)
    elif change == "unrecorded":
        evidence = replace(evidence, reference_id="unrecorded")
    elif change == "kind":
        evidence = replace(evidence, source_kind=EvidenceSourceKind.OBSERVABLE_ACTION)
    measurement = replace(
        measurement,
        evidence_refs=(evidence,),
        provenance=("forged",) if change == "provenance" else measurement.provenance,
    )
    update = replace(
        record.update,
        measurements=(measurement,),
        evidence_refs=tuple(dict.fromkeys((EVIDENCE, evidence))),
    )
    binding = replace(
        record.bindings[0],
        innovation=replace(
            record.bindings[0].innovation,
            evidence_refs=(evidence,),
        ),
    )
    events = sequence(replace(record, update=update, bindings=(binding,)))
    if change == "kind":
        events[4] = relational_verifier()
    with pytest.raises(ValueError):
        validate_action_bound_sequence(events)


def test_failed_verification_can_remain_an_unverified_diagnostic() -> None:
    record = reconciliation()
    measurement = replace(record.update.measurements[0], source=ConfidenceSource.TOOL_RESULT)
    record = replace(
        record,
        update=replace(record.update, measurements=(measurement,)),
        bindings=(replace(record.bindings[0], verifier_event_id=None),),
    )
    events = sequence(record)
    events[4] = replace(events[4], payload=dict(events[4].payload) | {"verified": False})
    validate_action_bound_sequence(events)


@pytest.mark.parametrize("path", [(True,), (-1,), (1.0,), (2**53,), "temperature", {}])
def test_invalid_path_types_rejected(path: object) -> None:
    with pytest.raises(ValueError):
        replace(reconciliation().bindings[0], prediction_path=path)


@pytest.mark.parametrize("value", [True, None, "22", {"value": 22}, [22]])
def test_selected_outcome_must_be_a_number(value: object) -> None:
    events = sequence()
    payload = dict(events[3].payload) | {"actual_outcome": {"temperature": [value]}}
    events[3] = replace(events[3], payload=payload)
    with pytest.raises(ValueError, match="selected outcome"):
        validate_action_bound_sequence(events)


def test_scalar_outcomes_and_timezone_offsets() -> None:
    record = reconciliation()
    binding = replace(record.bindings[0], prediction_path=(), observation_path=())
    events = sequence(replace(record, bindings=(binding,)))
    events[0] = replace(events[0], payload=dict(events[0].payload) | {"predicted_outcome": 20})
    events[3] = replace(events[3], payload=dict(events[3].payload) | {"actual_outcome": 22})
    events[4] = replace(events[4], timestamp="2026-09-30T08:00:04-04:00")
    validate_action_bound_sequence(events)


def test_a_different_committed_prediction_cannot_replace_action_ancestry() -> None:
    record = reconciliation()
    binding = replace(
        record.bindings[0],
        innovation=replace(
            record.bindings[0].innovation,
            prediction_commit_id="prediction-other",
        ),
    )
    update = replace(
        record.update,
        posterior_state=replace(
            record.update.posterior_state,
            prediction_commit_id="prediction-other",
        ),
    )
    events = sequence(
        replace(record, prediction_event_id="p-other", bindings=(binding,), update=update)
    )
    other = replace(
        events[0],
        event_id="p-other",
        payload=dict(events[0].payload)
        | {
            "prediction_commit_id": "prediction-other",
        },
    )
    with pytest.raises(ValueError, match="executed prediction"):
        validate_action_bound_sequence([other, *events])


def test_exact_schema_and_complete_bindings() -> None:
    record = reconciliation()
    payload = record.to_dict()
    for key in payload:
        malformed = dict(payload)
        del malformed[key]
        with pytest.raises(ValueError):
            EpistemicReconciliation.from_dict(malformed)
    for extra in ({"schema_version": "future"}, {"authorized": True}):
        with pytest.raises(ValueError):
            EpistemicReconciliation.from_dict(payload | extra)
    for bindings in ((), record.bindings * 2):
        with pytest.raises(ValueError, match="cover each"):
            replace(record, bindings=bindings)
    assert EpistemicReconciliation.from_dict(sequence()[-1].payload) == record


def test_prediction_must_strictly_predate_execution_even_with_equal_earlier_times() -> None:
    events = sequence()
    for index in (0, 1, 2):
        events[index] = replace(events[index], timestamp="2026-09-30T12:00:00Z")
    with pytest.raises(ValueError, match="predate execution"):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_id", 13),
        ("case_version", "other"),
        ("seed", 99),
        ("stripe_id", "other"),
        ("stripe_subtype", "other"),
    ],
)
def test_v5_cell_includes_reconciliation_without_changing_scores(field: str, value: object) -> None:
    # Reuse the established Boolean V5 outcome fixture; numeric telemetry is a separate action.
    from test_v5_provenance import _base_metadata, _record, _verified_sequence

    v5_record = _record()
    legacy = _verified_sequence()
    cell = _base_metadata()
    context = EpistemicContext(
        cell["run_id"],
        cell["repeat_id"],
        EvaluatedSystemVersion(cell["model_version"], cell["harness_version"]),
    )
    record = reconciliation()
    update = replace(
        record.update,
        prior_state=replace(record.update.prior_state, context=context),
        posterior_state=replace(record.update.posterior_state, context=context),
        measurements=tuple(replace(m, context=context) for m in record.update.measurements),
    )
    binding = replace(
        record.bindings[0],
        innovation=replace(
            record.bindings[0].innovation,
            context=context,
        ),
    )
    events = [
        replace(e, **cell) for e in sequence(replace(record, update=update, bindings=(binding,)))
    ]
    baseline = validate_v5_record_provenance(v5_record, legacy).optimizer_scores()
    assert (
        validate_v5_record_provenance(v5_record, [*legacy, *events]).optimizer_scores() == baseline
    )
    events[-1] = replace(events[-1], **{field: value})
    with pytest.raises(ValueError, match=field):
        validate_v5_record_provenance(v5_record, [*legacy, *events])


def with_prior_links(**links: object) -> EpistemicReconciliation:
    record = reconciliation()
    return replace(
        record,
        update=replace(
            record.update,
            prior_state=replace(record.update.prior_state, **links),
        ),
    )


def historical_action() -> list[EventEnvelope]:
    action = ActionRecord("past-action", "read_sensor", True, "read-only", "past-prediction")
    events = [
        make_prediction_commit_event(
            PredictionCommit("past-prediction", 18, 0.7, ("sensor-log",)),
            **metadata("past-p", 0),
        ),
        make_action_event(
            action, StructuredEventType.ACTION_PROPOSED, **metadata("past-a", 1, ("past-p",))
        ),
        make_action_event(
            action, StructuredEventType.ACTION_EXECUTED, **metadata("past-x", 2, ("past-a",))
        ),
    ]
    return [replace(e, timestamp=e.timestamp.replace("12:00", "11:59")) for e in events]


@pytest.mark.parametrize(
    "links",
    [
        {"action_id": "never-recorded"},
        {"prediction_commit_id": "never-recorded"},
        {"action_id": "action", "prediction_commit_id": "contradicts-actions-prediction"},
        {"action_id": "past-action", "prediction_commit_id": "prediction"},
    ],
)
def test_prior_causal_labels_must_resolve_and_agree(links: dict) -> None:
    with pytest.raises(ValueError, match="prior"):
        validate_action_bound_sequence([*historical_action(), *sequence(with_prior_links(**links))])


@pytest.mark.parametrize(
    "links",
    [
        {},
        {"action_id": "action"},
        {"prediction_commit_id": "prediction"},
        {"action_id": "action", "prediction_commit_id": "prediction"},
        {"action_id": "past-action"},
        {"prediction_commit_id": "past-prediction"},
        {"action_id": "past-action", "prediction_commit_id": "past-prediction"},
    ],
)
def test_initial_current_and_historical_prior_links(links: dict) -> None:
    validate_action_bound_sequence([*historical_action(), *sequence(with_prior_links(**links))])


@pytest.mark.parametrize("location", ["after-reconciliation", "after-prediction", "future-time"])
def test_historical_prior_must_precede_current_prediction(location: str) -> None:
    events = sequence(with_prior_links(action_id="past-action"))
    history = historical_action()
    if location == "after-reconciliation":
        events = [*events, *history]
    elif location == "after-prediction":
        events = [events[0], *history, *events[1:]]
    else:
        history = [replace(e, timestamp=e.timestamp.replace("11:59", "12:00")) for e in history]
        events = [*history, *events]
    with pytest.raises(ValueError, match="prior"):
        validate_action_bound_sequence(events)


def test_prior_link_cannot_cross_evaluation_units() -> None:
    history = [replace(e, run_id="other-run") for e in historical_action()]
    with pytest.raises(ValueError, match="context"):
        validate_action_bound_sequence(
            [
                *history,
                *sequence(with_prior_links(action_id="past-action")),
            ]
        )
