"""Contract tests for causal action-bound event sequences."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mindful_trace_gepa.action_bound_events import PredictionCommit, make_prediction_commit_event
from mindful_trace_gepa.event_sequence import (
    EvaluatedSystemVersion,
    validate_action_bound_sequence,
)
from mindful_trace_gepa.logging_schema import (
    EventEnvelope,
    StructuredEventType,
    make_event_envelope,
)


def _metadata(event_id: str, timestamp: str, **extra: object) -> dict[str, object]:
    """Create literal metadata for one event in the same evaluation unit."""

    return {
        "event_id": event_id,
        "timestamp": timestamp,
        "run_id": "run-1",
        "repeat_id": 0,
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        **extra,
    }


def _valid_sequence(*, repeat_id: int | None = 0) -> list[EventEnvelope]:
    """Build literal envelopes that exercise validator decoding without event helpers."""

    base = {
        "run_id": "run-1",
        "repeat_id": repeat_id,
        "model_version": "model-v1",
        "harness_version": "harness-v1",
    }
    prediction = EventEnvelope(
        schema_version="1.0",
        event_id="prediction-event-1",
        event_type="prediction_commit",
        timestamp="2026-09-10T12:00:00Z",
        payload={
            "prediction_commit_id": "prediction-1",
            "predicted_outcome": {"answer": "safe"},
            "confidence": 0.9,
            "evidence_refs": ["evidence-1"],
        },
        evidence_refs=("evidence-1",),
        **base,
    )
    proposed = EventEnvelope(
        schema_version="1.0",
        event_id="proposed-event-1",
        event_type="action_proposed",
        timestamp="2026-09-10T12:00:01Z",
        parent_event_ids=(prediction.event_id,),
        action_id="action-1",
        authorization_scope="sandbox",
        payload={
            "action_id": "action-1",
            "action_class": "read",
            "reversible": True,
            "authorization_scope": "sandbox",
            "prediction_commit_id": "prediction-1",
        },
        **base,
    )
    executed = EventEnvelope(
        schema_version="1.0",
        event_id="executed-event-1",
        event_type="action_executed",
        timestamp="2026-09-10T12:00:02Z",
        parent_event_ids=(proposed.event_id,),
        action_id="action-1",
        authorization_scope="sandbox",
        payload={
            "action_id": "action-1",
            "action_class": "read",
            "reversible": True,
            "authorization_scope": "sandbox",
            "prediction_commit_id": "prediction-1",
        },
        **base,
    )
    observation = EventEnvelope(
        schema_version="1.0",
        event_id="observation-event-1",
        event_type="outcome_observed",
        timestamp="2026-09-10T12:00:03Z",
        parent_event_ids=(executed.event_id,),
        action_id="action-1",
        evidence_refs=("evidence-2",),
        payload={
            "observation_id": "observation-1",
            "action_id": "action-1",
            "actual_outcome": {"answer": "safe"},
            "evidence_refs": ["evidence-2"],
        },
        **base,
    )
    verification = EventEnvelope(
        schema_version="1.0",
        event_id="verification-event-1",
        event_type="verification_result",
        timestamp="2026-09-10T12:00:04Z",
        parent_event_ids=(observation.event_id,),
        verifier_refs=("verifier-ref-1",),
        payload={
            "verifier_id": "verifier-1",
            "verifier_version": "v1",
            "observation_id": "observation-1",
            "verified": True,
            "verifier_refs": ["verifier-ref-1"],
        },
        **base,
    )
    epistemic = EventEnvelope(
        schema_version="1.0",
        event_id="epistemic-event-1",
        event_type="epistemic_assessment",
        timestamp="2026-09-10T12:00:05Z",
        parent_event_ids=(verification.event_id,),
        payload={"assessment": "well-supported"},
        **base,
    )
    case = EventEnvelope(
        schema_version="1.0",
        event_id="case-event-1",
        event_type="case_assessment",
        timestamp="2026-09-10T12:00:06Z",
        parent_event_ids=(epistemic.event_id,),
        payload={"assessment": "pass"},
        **base,
    )
    return [prediction, proposed, executed, observation, verification, epistemic, case]


def test_valid_full_action_bound_sequence_has_a_frozen_system_version() -> None:
    """Catch validator changes that reject the documented causal happy path."""

    events = _valid_sequence()

    assert EvaluatedSystemVersion("model-v1", "harness-v1").model_version == "model-v1"
    validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda events: [events[1], events[0], *events[2:]], "forward"),
        (lambda events: [*events[:3], events[4], events[3], *events[5:]], "forward"),
        (lambda events: [*events[:2], events[3], events[2], *events[4:]], "forward"),
    ],
)
def test_sequence_rejects_forward_causal_dependencies(
    mutator: object,
    match: str,
) -> None:
    """Catch actions, outcomes, or verifications recorded before their cause exists."""

    events = _valid_sequence()

    with pytest.raises(ValueError, match=match):
        validate_action_bound_sequence(mutator(events))  # type: ignore[operator]


@pytest.mark.parametrize(
    ("index", "parents", "match"),
    [
        (1, (), "parent_event_ids"),
        (2, ("prediction-event-1",), "action_proposed"),
        (3, ("prediction-event-1",), "action_executed"),
        (4, ("prediction-event-1",), "outcome_observed"),
        (5, (), "parent_event_ids"),
        (6, ("verification-event-1",), "epistemic_assessment"),
    ],
)
def test_sequence_rejects_missing_or_wrong_causal_parent_links(
    index: int,
    parents: tuple[str, ...],
    match: str,
) -> None:
    """Catch a causal event whose declared parent is absent or has the wrong event kind."""

    events = _valid_sequence()
    events[index] = replace(events[index], parent_event_ids=parents)

    with pytest.raises(ValueError, match=match):
        validate_action_bound_sequence(events)


def test_sequence_rejects_duplicate_unknown_or_forward_parent_references() -> None:
    """Catch parent references that do not name one unique earlier event."""

    events = _valid_sequence()
    events[1] = replace(events[1], parent_event_ids=("prediction-event-1", "prediction-event-1"))
    with pytest.raises(ValueError, match="duplicate parent"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[1] = replace(events[1], parent_event_ids=("missing-event",))
    with pytest.raises(ValueError, match="unknown parent"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[1] = replace(events[1], parent_event_ids=("proposed-event-1",))
    with pytest.raises(ValueError, match="forward parent"):
        validate_action_bound_sequence(events)


def test_sequence_rejects_duplicate_event_and_semantic_identifiers() -> None:
    """Catch append-only identity rewrites of envelopes or payload records."""

    events = _valid_sequence()
    events.append(replace(events[-1], event_id="prediction-event-1"))
    with pytest.raises(ValueError, match="duplicate event_id"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    rewritten_prediction = make_prediction_commit_event(
        PredictionCommit("prediction-1", {"answer": "unsafe"}, 0.1, ("evidence-3",)),
        **_metadata("prediction-event-2", "2026-09-10T12:00:07Z"),
    )
    events.append(rewritten_prediction)
    with pytest.raises(ValueError, match="prediction_commit_id"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events.append(replace(events[3], event_id="observation-event-2"))
    with pytest.raises(ValueError, match="observation_id"):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    ("index", "namespace"),
    [
        (0, "prediction_commit_id"),
        (1, "proposed action_id"),
        (2, "executed action_id"),
        (3, "observation_id"),
        (4, "verifier_id"),
    ],
)
def test_sequence_rejects_each_duplicate_semantic_identifier_namespace(
    index: int,
    namespace: str,
) -> None:
    """Catch duplicate semantic IDs independently for every action-bound payload namespace."""

    events = _valid_sequence()
    events.append(replace(events[index], event_id=f"duplicate-event-{index}"))

    with pytest.raises(ValueError, match=namespace):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("run_id", "run-2"), ("repeat_id", 1)],
)
def test_sequence_rejects_causal_parents_from_another_run_or_repeat(
    field_name: str,
    value: str | int,
) -> None:
    """Catch an execution that claims a direct proposal from a distinct evaluation unit."""

    events = _valid_sequence()
    events[2] = replace(events[2], **{field_name: value})

    with pytest.raises(ValueError, match="same evaluation unit"):
        validate_action_bound_sequence(events)


def test_sequence_treats_none_repeat_as_isolated_from_numbered_repetitions() -> None:
    """Catch a numbered action proposal that cites a prediction from the None-repeat unit."""

    events = _valid_sequence(repeat_id=None)
    events[1] = replace(events[1], repeat_id=0)

    with pytest.raises(ValueError, match="same evaluation unit"):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    ("run_id", "repeat_id"),
    [("run-2", 0), ("run-1", 1), ("run-1", None)],
)
def test_sequence_rejects_prediction_semantic_id_reuse_across_evaluation_units(
    run_id: str,
    repeat_id: int | None,
) -> None:
    """Catch semantic prediction identity reuse even when envelope IDs and units differ."""

    events = _valid_sequence()
    events.append(
        replace(
            events[0],
            event_id=f"prediction-event-{run_id}-{repeat_id}",
            run_id=run_id,
            repeat_id=repeat_id,
        )
    )

    with pytest.raises(ValueError, match="prediction_commit_id"):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize(
    ("run_id", "repeat_id"),
    [("run-2", 0), ("run-1", 1), ("run-1", None)],
)
def test_sequence_rejects_cross_unit_action_semantic_links(
    run_id: str,
    repeat_id: int | None,
) -> None:
    """Catch a new unit reusing an existing action semantic ID after a distinct prediction."""

    events = _valid_sequence()
    prediction = EventEnvelope(
        schema_version="1.0",
        event_id=f"cross-unit-prediction-{run_id}-{repeat_id}",
        event_type="prediction_commit",
        timestamp="2026-09-10T12:00:07Z",
        run_id=run_id,
        repeat_id=repeat_id,
        model_version="model-v2",
        harness_version="harness-v2",
        evidence_refs=("evidence-cross-unit",),
        payload={
            "prediction_commit_id": "prediction-cross-unit",
            "predicted_outcome": {"answer": "safe"},
            "confidence": 0.9,
            "evidence_refs": ["evidence-cross-unit"],
        },
    )
    proposal = EventEnvelope(
        schema_version="1.0",
        event_id=f"cross-unit-proposal-{run_id}-{repeat_id}",
        event_type="action_proposed",
        timestamp="2026-09-10T12:00:08Z",
        run_id=run_id,
        repeat_id=repeat_id,
        model_version="model-v2",
        harness_version="harness-v2",
        parent_event_ids=(prediction.event_id,),
        action_id="action-1",
        authorization_scope="sandbox",
        payload={
            "action_id": "action-1",
            "action_class": "read",
            "reversible": True,
            "authorization_scope": "sandbox",
            "prediction_commit_id": "prediction-cross-unit",
        },
    )

    with pytest.raises(ValueError, match="duplicate proposed action_id"):
        validate_action_bound_sequence([*events, prediction, proposal])


def test_sequence_rejects_payload_type_and_envelope_link_mismatches() -> None:
    """Catch malformed payloads or semantic IDs that disagree with their envelope."""

    events = _valid_sequence()
    events[1] = replace(events[1], action_id="other-action")
    with pytest.raises(ValueError, match="action_id"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[3] = replace(events[3], payload={"action_id": "action-1"})
    with pytest.raises(ValueError, match="OutcomeObservation"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[4] = replace(events[4], verifier_refs=("different-ref",))
    with pytest.raises(ValueError, match="verifier_refs"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[3] = replace(events[3], action_id="action-2")
    with pytest.raises(ValueError, match="action_id"):
        validate_action_bound_sequence(events)


@pytest.mark.parametrize("field", ["model_version", "harness_version"])
def test_sequence_requires_nonblank_version_identity_for_action_bound_events(field: str) -> None:
    """Catch events that cannot identify the evaluated model and harness pair."""

    events = _valid_sequence()
    if field == "model_version":
        events[0] = replace(events[0], model_version=None)
    else:
        events[0] = replace(events[0], harness_version=None)

    with pytest.raises(ValueError, match=field):
        validate_action_bound_sequence(events)


def test_sequence_rejects_version_drift_only_inside_the_same_run_and_repeat() -> None:
    """Catch model or harness changes within one evaluation unit, not across units."""

    events = _valid_sequence()
    events[3] = replace(events[3], model_version="model-v2")
    with pytest.raises(ValueError, match="model_version"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    unrelated_unit = make_prediction_commit_event(
        PredictionCommit("prediction-2", {"answer": "safe"}, 0.9, ("evidence-2",)),
        **_metadata(
            "prediction-event-2",
            "2026-09-10T12:00:07Z",
            run_id="run-2",
            repeat_id=1,
            model_version="model-v2",
            harness_version="harness-v2",
        ),
    )
    validate_action_bound_sequence([*events, unrelated_unit])


def test_sequence_treats_a_none_repeat_as_one_distinct_evaluation_unit() -> None:
    """Catch None repeat IDs being coupled to numbered repetitions or left unvalidated."""

    events = _valid_sequence(repeat_id=None)
    validate_action_bound_sequence(events)

    numbered = _valid_sequence(repeat_id=1)
    numbered[0] = replace(numbered[0], model_version="model-v2")
    with pytest.raises(ValueError, match="model_version"):
        validate_action_bound_sequence(numbered)


def test_only_later_same_unit_derived_events_may_supersede() -> None:
    """Catch raw-event rewrites and invalid derived supersession targets."""

    events = _valid_sequence()
    events[5] = replace(events[5], superseded_by="epistemic-event-2")
    replacement = replace(
        events[5],
        event_id="epistemic-event-2",
        timestamp="2026-09-10T12:00:07Z",
        superseded_by=None,
    )
    events.append(replacement)
    validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[0] = replace(events[0], superseded_by="case-event-1")
    with pytest.raises(ValueError, match="superseded_by"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[5] = replace(events[5], superseded_by="case-event-1")
    with pytest.raises(ValueError, match="same event type"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[5] = replace(events[5], superseded_by="epistemic-event-1")
    with pytest.raises(ValueError, match="later"):
        validate_action_bound_sequence(events)

    events = _valid_sequence()
    events[5] = replace(events[5], superseded_by="missing-event")
    with pytest.raises(ValueError, match="unknown"):
        validate_action_bound_sequence(events)


def test_supersession_rejects_cross_unit_chains_and_multiple_sources() -> None:
    """Catch ambiguous assessment rewrites outside a single derived event lineage."""

    events = _valid_sequence()
    events[5] = replace(events[5], superseded_by="epistemic-event-2")
    replacement = replace(
        events[5],
        event_id="epistemic-event-2",
        timestamp="2026-09-10T12:00:07Z",
        superseded_by="epistemic-event-3",
    )
    terminal = replace(
        events[5],
        event_id="epistemic-event-3",
        timestamp="2026-09-10T12:00:08Z",
        superseded_by=None,
    )
    with pytest.raises(ValueError, match="chain"):
        validate_action_bound_sequence([*events, replacement, terminal])

    events = _valid_sequence()
    alternate = replace(events[5], event_id="epistemic-event-2", timestamp="2026-09-10T12:00:07Z")
    events[5] = replace(events[5], superseded_by="epistemic-event-3")
    alternate = replace(alternate, superseded_by="epistemic-event-3")
    terminal = replace(
        events[5],
        event_id="epistemic-event-3",
        timestamp="2026-09-10T12:00:08Z",
        superseded_by=None,
    )
    with pytest.raises(ValueError, match="multiple"):
        validate_action_bound_sequence([*events, alternate, terminal])

    events = _valid_sequence()
    replacement = replace(
        events[5],
        event_id="epistemic-event-2",
        timestamp="2026-09-10T12:00:07Z",
        run_id="run-2",
    )
    events[5] = replace(events[5], superseded_by="epistemic-event-2")
    with pytest.raises(ValueError, match="evaluation unit"):
        validate_action_bound_sequence([*events, replacement])


def test_legacy_and_unrelated_events_remain_valid_and_ignored() -> None:
    """Catch legacy event rows accidentally routed through action-bound validation."""

    legacy = EventEnvelope("1.0", "legacy-event", "legacy_event", "2026-09-10T11:59:59Z")
    unrelated = make_event_envelope(
        StructuredEventType.REWARD_BREAKDOWN,
        {"reward": 1.0},
        event_id="reward-event",
        timestamp="2026-09-10T12:00:00Z",
        parent_event_ids=("future-event",),
    )

    validate_action_bound_sequence([legacy, unrelated, *_valid_sequence()])
