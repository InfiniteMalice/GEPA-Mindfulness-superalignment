"""Contract tests for causal action-bound event sequences."""

from __future__ import annotations

from dataclasses import replace

import pytest

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
    """Build a literal causal sequence with all seven action-bound event kinds."""

    base = {
        "run_id": "run-1",
        "repeat_id": repeat_id,
        "model_version": "model-v1",
        "harness_version": "harness-v1",
    }
    prediction = make_prediction_commit_event(
        PredictionCommit("prediction-1", {"answer": "safe"}, 0.9, ("evidence-1",)),
        event_id="prediction-event-1",
        timestamp="2026-09-10T12:00:00Z",
        **base,
    )
    proposed = make_action_event(
        ActionRecord("action-1", "read", True, "sandbox", "prediction-1"),
        StructuredEventType.ACTION_PROPOSED,
        event_id="proposed-event-1",
        timestamp="2026-09-10T12:00:01Z",
        parent_event_ids=(prediction.event_id,),
        **base,
    )
    executed = make_action_event(
        ActionRecord("action-1", "read", True, "sandbox", "prediction-1"),
        StructuredEventType.ACTION_EXECUTED,
        event_id="executed-event-1",
        timestamp="2026-09-10T12:00:02Z",
        parent_event_ids=(proposed.event_id,),
        **base,
    )
    observation = make_outcome_observation_event(
        OutcomeObservation("observation-1", "action-1", {"answer": "safe"}, ("evidence-2",)),
        event_id="observation-event-1",
        timestamp="2026-09-10T12:00:03Z",
        parent_event_ids=(executed.event_id,),
        **base,
    )
    verification = make_verification_result_event(
        VerificationResult("verifier-1", "v1", "observation-1", True, ("verifier-ref-1",)),
        event_id="verification-event-1",
        timestamp="2026-09-10T12:00:04Z",
        parent_event_ids=(observation.event_id,),
        **base,
    )
    epistemic = make_event_envelope(
        StructuredEventType.EPISTEMIC_ASSESSMENT,
        {"assessment": "well-supported"},
        **_metadata(
            "epistemic-event-1",
            "2026-09-10T12:00:05Z",
            parent_event_ids=(verification.event_id,),
            **base,
        ),
    )
    case = make_event_envelope(
        StructuredEventType.CASE_ASSESSMENT,
        {"assessment": "pass"},
        **_metadata(
            "case-event-1",
            "2026-09-10T12:00:06Z",
            parent_event_ids=(epistemic.event_id,),
            **base,
        ),
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
