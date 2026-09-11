"""Contract tests for causal action-bound event sequences."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.interfaces import (
    LocalVerificationResult,
    RelationalVerificationResult,
    VerificationEvidenceBinding,
    make_local_verification_event,
    make_relational_verification_event,
)
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


def _metadata(event_id: str, timestamp: str, **extra: object) -> dict[str, Any]:
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

    base: dict[str, Any] = {
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


def _second_action_chain() -> list[EventEnvelope]:
    """Build a second independent action ancestry in the default evaluation unit."""

    base: dict[str, Any] = {
        "run_id": "run-1",
        "repeat_id": 0,
        "model_version": "model-v1",
        "harness_version": "harness-v1",
    }
    prediction = EventEnvelope(
        "1.0",
        "prediction-event-2",
        "prediction_commit",
        "2026-09-10T12:00:07Z",
        evidence_refs=("evidence-3",),
        payload={
            "prediction_commit_id": "prediction-2",
            "predicted_outcome": {"answer": "also-safe"},
            "confidence": 0.8,
            "evidence_refs": ["evidence-3"],
        },
        **base,
    )
    proposed = EventEnvelope(
        "1.0",
        "proposed-event-2",
        "action_proposed",
        "2026-09-10T12:00:08Z",
        parent_event_ids=(prediction.event_id,),
        action_id="action-2",
        authorization_scope="sandbox",
        payload={
            "action_id": "action-2",
            "action_class": "read",
            "reversible": True,
            "authorization_scope": "sandbox",
            "prediction_commit_id": "prediction-2",
        },
        **base,
    )
    executed = replace(
        proposed,
        event_id="executed-event-2",
        event_type="action_executed",
        timestamp="2026-09-10T12:00:09Z",
        parent_event_ids=(proposed.event_id,),
    )
    observation = EventEnvelope(
        "1.0",
        "observation-event-2",
        "outcome_observed",
        "2026-09-10T12:00:10Z",
        parent_event_ids=(executed.event_id,),
        action_id="action-2",
        evidence_refs=("evidence-4",),
        payload={
            "observation_id": "observation-2",
            "action_id": "action-2",
            "actual_outcome": {"answer": "also-safe"},
            "evidence_refs": ["evidence-4"],
        },
        **base,
    )
    verification = EventEnvelope(
        "1.0",
        "verification-event-2",
        "verification_result",
        "2026-09-10T12:00:11Z",
        parent_event_ids=(observation.event_id,),
        action_id="action-2",
        verifier_refs=("verifier-ref-2",),
        payload={
            "verifier_id": "verifier-2",
            "verifier_version": "v1",
            "observation_id": "observation-2",
            "verified": True,
            "verifier_refs": ["verifier-ref-2"],
        },
        **base,
    )
    return [prediction, proposed, executed, observation, verification]


def _sequence_with_both_leveled_verifications() -> list[EventEnvelope]:
    """Replace the legacy result with local and relational verification of one observation."""

    events = _valid_sequence()
    observation = events[3]
    evidence = EvidenceReference("evidence-2", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    local = LocalVerificationResult(
        "action-1",
        True,
        False,
        False,
        False,
        False,
        None,
        (evidence,),
        (VerificationEvidenceBinding("executed", (evidence,)),),
    )
    relational = RelationalVerificationResult(
        "action-1",
        False,
        False,
        "none",
        False,
        False,
        True,
        False,
        (evidence,),
        (
            VerificationEvidenceBinding("contradiction_status", (evidence,)),
            VerificationEvidenceBinding("claimed_outcome_supported", (evidence,)),
        ),
    )
    common = {
        "run_id": "run-1",
        "repeat_id": 0,
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        "parent_event_ids": (observation.event_id,),
    }
    local_event = make_local_verification_event(
        local,
        verifier_refs=("verifier:local-1",),
        event_id="verification-local-1",
        timestamp="2026-09-10T12:00:04Z",
        **common,
    )
    relational_event = make_relational_verification_event(
        relational,
        verifier_refs=("verifier:relational-1",),
        event_id="verification-relational-1",
        timestamp="2026-09-10T12:00:05Z",
        **common,
    )
    epistemic = replace(
        events[5],
        parent_event_ids=(local_event.event_id, relational_event.event_id),
    )
    return [*events[:4], local_event, relational_event, epistemic, events[6]]


def test_valid_full_action_bound_sequence_has_a_frozen_system_version() -> None:
    """Catch validator changes that reject the documented causal happy path."""

    events = _valid_sequence()

    assert EvaluatedSystemVersion("model-v1", "harness-v1").model_version == "model-v1"
    validate_action_bound_sequence(events)


def test_valid_sequence_accepts_both_structured_verification_levels() -> None:
    """Catch canonical validation rejecting either new typed verification level."""

    validate_action_bound_sequence(_sequence_with_both_leveled_verifications())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda event: replace(
                event,
                payload={**event.to_dict()["payload"], "verification_level": "unknown"},
            ),
            "verification_level",
        ),
        (
            lambda event: replace(
                event,
                payload={**event.to_dict()["payload"], "verified": True},
            ),
            "payload fields",
        ),
        (lambda event: replace(event, action_id="other-action"), "action_id"),
        (lambda event: replace(event, evidence_refs=("other-evidence",)), "evidence_refs"),
        (lambda event: replace(event, verifier_refs=("verifier:other",)), "verifier_refs"),
        (
            lambda event: replace(
                event,
                payload={**event.to_dict()["payload"], "verifier_refs": []},
            ),
            "verifier_refs",
        ),
    ],
)
def test_sequence_rejects_malformed_or_mismatched_leveled_verification(
    mutator: Any,
    match: str,
) -> None:
    """Catch new verification envelopes bypassing typed payload and linkage validation."""

    events = _sequence_with_both_leveled_verifications()
    events[4] = mutator(events[4])

    with pytest.raises(ValueError, match=match):
        validate_action_bound_sequence(events)


def test_sequence_rejects_result_type_that_disagrees_with_verification_level() -> None:
    """Catch a relational result relabeled as local execution evidence."""

    events = _sequence_with_both_leveled_verifications()
    local_event = events[4]
    relational_payload = events[5].to_dict()["payload"]
    events[4] = replace(
        local_event,
        payload={
            "verification_level": "local_execution",
            "result": relational_payload["result"],
            "verifier_refs": ["verifier:local-1"],
        },
    )

    with pytest.raises(ValueError, match="LocalVerificationResult"):
        validate_action_bound_sequence(events)


def test_sequence_rejects_leveled_verification_without_observation_parent() -> None:
    """Catch a structured verification result attached to another verifier instead of evidence."""

    events = _sequence_with_both_leveled_verifications()
    events[5] = replace(events[5], parent_event_ids=(events[4].event_id,))

    with pytest.raises(ValueError, match="outcome_observed"):
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
    update: dict[str, Any] = {field_name: value}
    events[2] = replace(events[2], **update)

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


def test_sequence_accepts_matching_action_ids_on_verification_and_assessments() -> None:
    """Catch ancestry propagation that rejects an explicitly correct optional action ID."""

    events = _valid_sequence()
    for index in (4, 5, 6):
        events[index] = replace(events[index], action_id="action-1")

    validate_action_bound_sequence(events)


@pytest.mark.parametrize("index", [4, 5, 6])
def test_sequence_rejects_action_id_that_conflicts_with_resolved_ancestry(index: int) -> None:
    """Catch a verification or assessment claiming an action outside its causal ancestry."""

    events = _valid_sequence()
    events[index] = replace(events[index], action_id="action-2")

    with pytest.raises(ValueError, match="action_id.*ancestry"):
        validate_action_bound_sequence(events)


def test_sequence_accepts_same_action_multi_parent_derived_events() -> None:
    """Catch same-action evidence aggregation being mistaken for causal ambiguity."""

    events = _valid_sequence()
    second_verification = EventEnvelope(
        "1.0",
        "verification-event-1b",
        "verification_result",
        "2026-09-10T12:00:04.500000Z",
        run_id="run-1",
        repeat_id=0,
        model_version="model-v1",
        harness_version="harness-v1",
        parent_event_ids=("observation-event-1",),
        action_id="action-1",
        verifier_refs=("verifier-ref-1b",),
        payload={
            "verifier_id": "verifier-1b",
            "verifier_version": "v1",
            "observation_id": "observation-1",
            "verified": True,
            "verifier_refs": ["verifier-ref-1b"],
        },
    )
    epistemic = replace(
        events[5],
        parent_event_ids=("verification-event-1", "verification-event-1b"),
        action_id="action-1",
    )
    second_epistemic = replace(
        events[5],
        event_id="epistemic-event-1b",
        timestamp="2026-09-10T12:00:05.500000Z",
        action_id="action-1",
    )
    case = replace(
        events[6],
        parent_event_ids=("epistemic-event-1", "epistemic-event-1b"),
        action_id="action-1",
    )

    validate_action_bound_sequence(
        [*events[:5], second_verification, epistemic, second_epistemic, case]
    )


def test_sequence_rejects_cross_action_multi_parent_assessment() -> None:
    """Catch a derived assessment with parents that resolve to different actions."""

    events = _valid_sequence()
    second_chain = _second_action_chain()
    epistemic = replace(
        events[5],
        timestamp="2026-09-10T12:00:12Z",
        parent_event_ids=("verification-event-1", "verification-event-2"),
    )
    case = replace(events[6], timestamp="2026-09-10T12:00:13Z")

    with pytest.raises(ValueError, match="parents.*one action"):
        validate_action_bound_sequence([*events[:5], *second_chain, epistemic, case])


def test_sequence_rejects_cross_action_derived_supersession() -> None:
    """Catch an assessment claiming to replace one from a different action ancestry."""

    events = _valid_sequence()
    second_chain = _second_action_chain()
    first_epistemic = replace(
        events[5],
        timestamp="2026-09-10T12:00:12Z",
        action_id="action-1",
        superseded_by="epistemic-event-2",
    )
    second_epistemic = replace(
        events[5],
        event_id="epistemic-event-2",
        timestamp="2026-09-10T12:00:13Z",
        parent_event_ids=("verification-event-2",),
        action_id="action-2",
    )

    with pytest.raises(ValueError, match="superseded_by.*action ancestry"):
        validate_action_bound_sequence(
            [*events[:5], *second_chain, first_epistemic, second_epistemic]
        )


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
