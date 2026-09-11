"""Contract tests for immutable action-bound event payloads."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from math import inf, nan
from typing import Any, cast

import pytest

from mindful_trace_gepa import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
    make_action_event,
    make_outcome_observation_event,
    make_prediction_commit_event,
    make_verification_result_event,
)
from mindful_trace_gepa.logging_schema import StructuredEventType


def test_prediction_commit_is_frozen_and_snapshots_nested_json_outcomes() -> None:
    """Catch callers rewriting a committed prediction through mutable input containers."""

    labels = ["safe"]
    details = {"score": 1}
    outcome: dict[str, object] = {"labels": labels, "details": details}
    commit = PredictionCommit("prediction-1", outcome, 0.75, cast(Any, [" evidence-1 "]))
    labels.append("rewritten")
    details["score"] = 0

    assert commit.evidence_refs == ("evidence-1",)
    expected = {
        "prediction_commit_id": "prediction-1",
        "predicted_outcome": {"details": {"score": 1}, "labels": ["safe"]},
        "confidence": 0.75,
        "evidence_refs": ["evidence-1"],
    }
    serialized = commit.to_dict()
    serialized_outcome = cast(dict[str, object], serialized["predicted_outcome"])
    serialized_details = cast(dict[str, int], serialized_outcome["details"])
    serialized_details["score"] = 0

    assert serialized != expected
    assert commit.to_dict() == expected
    assert json.loads(json.dumps(commit.to_dict())) == expected
    with pytest.raises(FrozenInstanceError):
        commit.confidence = 0.25  # type: ignore[misc]


@pytest.mark.parametrize("confidence", [-0.01, 1.01, nan, inf, True, "0.5"])
def test_prediction_commit_rejects_invalid_confidence(confidence: object) -> None:
    """Catch unbounded, nonfinite, or nonnumeric confidence scores."""

    with pytest.raises(ValueError, match="confidence"):
        PredictionCommit("prediction-1", "safe", cast(Any, confidence), ())


@pytest.mark.parametrize(
    ("constructor", "args", "match"),
    [
        (PredictionCommit, (" ", "safe", 0.5, ()), "prediction_commit_id"),
        (ActionRecord, (" ", "read", True, "sandbox", "prediction-1"), "action_id"),
        (ActionRecord, ("action-1", " ", True, "sandbox", "prediction-1"), "action_class"),
        (ActionRecord, ("action-1", "read", True, " ", "prediction-1"), "authorization_scope"),
        (ActionRecord, ("action-1", "read", True, "sandbox", " "), "prediction_commit_id"),
        (OutcomeObservation, (" ", "action-1", "safe", ("evidence-1",)), "observation_id"),
        (OutcomeObservation, ("observation-1", " ", "safe", ("evidence-1",)), "action_id"),
        (VerificationResult, (" ", "v1", "observation-1", True, ("ref-1",)), "verifier_id"),
        (
            VerificationResult,
            ("verifier-1", " ", "observation-1", True, ("ref-1",)),
            "verifier_version",
        ),
        (VerificationResult, ("verifier-1", "v1", " ", True, ("ref-1",)), "observation_id"),
    ],
)
def test_action_bound_payloads_reject_blank_required_strings(
    constructor: type[Any],
    args: tuple[Any, ...],
    match: str,
) -> None:
    """Catch payloads with identifiers or authorization values that cannot be audited."""

    with pytest.raises(ValueError, match=match):
        constructor(*args)


@pytest.mark.parametrize("reversible", [1, 0, "true", None])
def test_action_record_requires_exact_boolean_reversibility(reversible: object) -> None:
    """Catch truthy values that would blur the action's reversibility contract."""

    with pytest.raises(ValueError, match="reversible"):
        ActionRecord("action-1", "read", cast(Any, reversible), "sandbox", "prediction-1")


@pytest.mark.parametrize("verified", [1, 0, "true", None])
def test_verification_result_requires_exact_boolean_status(verified: object) -> None:
    """Catch truthy values that would blur an independently verified result."""

    with pytest.raises(ValueError, match="verified"):
        VerificationResult("verifier-1", "v1", "observation-1", cast(Any, verified), ("ref-1",))


@pytest.mark.parametrize(
    ("constructor", "args", "match"),
    [
        (OutcomeObservation, ("observation-1", "action-1", "safe", ()), "evidence_refs"),
        (OutcomeObservation, ("observation-1", "action-1", "safe", [" "]), "evidence_refs"),
        (VerificationResult, ("verifier-1", "v1", "observation-1", True, ()), "verifier_refs"),
        (VerificationResult, ("verifier-1", "v1", "observation-1", True, [" "]), "verifier_refs"),
    ],
)
def test_observations_and_verifications_require_provenance_references(
    constructor: type[Any],
    args: tuple[Any, ...],
    match: str,
) -> None:
    """Catch observed facts or verification claims without required supporting references."""

    with pytest.raises(ValueError, match=match):
        constructor(*args)


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string key"},
        {"nonfinite": nan},
        {"nonfinite": inf},
        {"unsupported": object()},
    ],
)
def test_outcome_payloads_reject_non_json_compatible_values(value: object) -> None:
    """Catch values that cannot be losslessly represented in JSON event records."""

    with pytest.raises(ValueError, match="outcomes"):
        PredictionCommit("prediction-1", value, 0.5, ())


def test_outcome_payloads_reject_cycles() -> None:
    """Catch recursive payloads before a serializer recurses indefinitely."""

    outcome: list[object] = []
    outcome.append(outcome)

    with pytest.raises(ValueError, match="cycle"):
        OutcomeObservation("observation-1", "action-1", outcome, ("evidence-1",))


def test_helpers_preserve_payloads_semantic_links_and_envelope_metadata() -> None:
    """Catch helper events that lose their typed payload linkage or run metadata."""

    common = {
        "run_id": "run-1",
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        "case_version": "17case-v5",
        "case_id": 1,
        "stripe_id": "none",
        "repeat_id": 0,
        "seed": 42,
        "parent_event_ids": ("parent-event-1",),
    }
    prediction = PredictionCommit("prediction-1", {"answer": "safe"}, 0.9, ("evidence-1",))
    action = ActionRecord("action-1", "read", True, "sandbox", "prediction-1")
    observation = OutcomeObservation(
        "observation-1", "action-1", {"answer": "safe"}, ("evidence-2",)
    )
    verification = VerificationResult(
        "verifier-1", "v1", "observation-1", True, ("verifier-ref-1",)
    )

    prediction_event = make_prediction_commit_event(prediction, **common)
    action_event = make_action_event(action, StructuredEventType.ACTION_EXECUTED, **common)
    observation_event = make_outcome_observation_event(observation, **common)
    verification_event = make_verification_result_event(verification, **common)

    assert prediction_event.event_type == "prediction_commit"
    assert prediction_event.evidence_refs == ("evidence-1",)
    assert action_event.event_type == "action_executed"
    assert action_event.action_id == "action-1"
    assert observation_event.event_type == "outcome_observed"
    assert observation_event.action_id == "action-1"
    assert observation_event.evidence_refs == ("evidence-2",)
    assert verification_event.event_type == "verification_result"
    assert verification_event.verifier_refs == ("verifier-ref-1",)
    expected_payloads = (
        {
            "prediction_commit_id": "prediction-1",
            "predicted_outcome": {"answer": "safe"},
            "confidence": 0.9,
            "evidence_refs": ["evidence-1"],
        },
        {
            "action_id": "action-1",
            "action_class": "read",
            "reversible": True,
            "authorization_scope": "sandbox",
            "prediction_commit_id": "prediction-1",
        },
        {
            "observation_id": "observation-1",
            "action_id": "action-1",
            "actual_outcome": {"answer": "safe"},
            "evidence_refs": ["evidence-2"],
        },
        {
            "verifier_id": "verifier-1",
            "verifier_version": "v1",
            "observation_id": "observation-1",
            "verified": True,
            "verifier_refs": ["verifier-ref-1"],
        },
    )
    for event, expected_payload in zip(
        (prediction_event, action_event, observation_event, verification_event),
        expected_payloads,
        strict=True,
    ):
        assert event.parent_event_ids == ("parent-event-1",)
        assert event.run_id == "run-1"
        assert event.payload == expected_payload
        assert json.loads(json.dumps(event.to_dict()))["payload"] == expected_payload


def test_action_event_requires_an_action_event_type() -> None:
    """Catch action records accidentally wrapped as an unrelated event category."""

    action = ActionRecord("action-1", "read", True, "sandbox", "prediction-1")

    with pytest.raises(ValueError, match="ACTION_PROPOSED or ACTION_EXECUTED"):
        make_action_event(action, StructuredEventType.PREDICTION_COMMIT)


def test_helper_rejects_conflicting_semantic_linkage_metadata() -> None:
    """Catch a caller attaching an event to a different payload action or evidence reference."""

    observation = OutcomeObservation("observation-1", "action-1", "safe", ("evidence-1",))

    with pytest.raises(ValueError, match="action_id"):
        make_outcome_observation_event(observation, action_id="other-action")
    with pytest.raises(ValueError, match="evidence_refs"):
        make_outcome_observation_event(observation, evidence_refs=("other-evidence",))
