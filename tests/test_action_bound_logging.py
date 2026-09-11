"""Contract tests for action-bound structured event envelopes."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from mindful_trace_gepa.logging_schema import (
    EventEnvelope,
    StructuredEventType,
    make_event_envelope,
    normalize_trace_event,
)


def test_action_bound_event_types_and_linkage_fields_round_trip() -> None:
    """Catch omitted event types or linkage metadata at the event boundary."""

    expected_types = {
        "PREDICTION_COMMIT": "prediction_commit",
        "ACTION_PROPOSED": "action_proposed",
        "ACTION_EXECUTED": "action_executed",
        "OUTCOME_OBSERVED": "outcome_observed",
        "VERIFICATION_RESULT": "verification_result",
        "EPISTEMIC_ASSESSMENT": "epistemic_assessment",
        "CASE_ASSESSMENT": "case_assessment",
    }
    assert {
        name: getattr(StructuredEventType, name).value for name in expected_types
    } == expected_types
    for event_type in expected_types:
        row = make_event_envelope(
            getattr(StructuredEventType, event_type),
            {"summary": event_type},
        ).to_dict()
        assert normalize_trace_event(row)["event_type"] == expected_types[event_type]

    event = make_event_envelope(
        StructuredEventType.ACTION_EXECUTED,
        {"summary": "executed after a prediction"},
        action_id="action-1",
        parent_event_ids=["prediction-event-1"],
        evidence_refs=["evidence-1"],
        model_version="model-v1",
        harness_version="harness-v1",
        case_version="17case-v5",
        case_id=17,
        stripe_id="baseline",
        repeat_id=0,
        seed=42,
        authorization_scope="sandbox",
        verifier_refs=["verifier-1"],
        valid_from="2026-09-10T12:00:00Z",
        valid_until="2026-09-10T12:05:00+00:00",
        superseded_by="assessment-event-2",
    )

    assert event.parent_event_ids == ("prediction-event-1",)
    assert event.evidence_refs == ("evidence-1",)
    assert event.verifier_refs == ("verifier-1",)
    row = event.to_dict()
    assert normalize_trace_event(row)["action_id"] == "action-1"
    assert json.loads(json.dumps(row))["parent_event_ids"] == ["prediction-event-1"]
    assert row["valid_until"] == "2026-09-10T12:05:00+00:00"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_id", " "),
        ("parent_event_ids", [" "]),
        ("evidence_refs", [""]),
        ("model_version", ""),
        ("harness_version", " "),
        ("case_version", ""),
        ("stripe_id", " "),
        ("authorization_scope", ""),
        ("verifier_refs", [" "]),
        ("superseded_by", ""),
    ],
)
def test_event_envelope_rejects_blank_action_bound_references(field: str, value: object) -> None:
    """Catch records that cannot be linked or audited reliably."""

    with pytest.raises(ValueError, match=field):
        invalid_kwargs: dict[str, Any] = {field: value}
        EventEnvelope("1.0", "event-1", "action_executed", "2026-09-10T12:00:00Z", **invalid_kwargs)


@pytest.mark.parametrize("repeat_id", [-1, True, 1.5])
def test_event_envelope_rejects_invalid_repeat_ids(repeat_id: object) -> None:
    """Catch invalid repetition identity before it reaches evaluation logs."""

    with pytest.raises(ValueError, match="repeat_id"):
        EventEnvelope(
            "1.0",
            "event-1",
            "action_executed",
            "2026-09-10T12:00:00Z",
            repeat_id=cast(Any, repeat_id),
        )


@pytest.mark.parametrize("case_id", [-1, 18, True, 1.5])
def test_event_envelope_rejects_noncanonical_case_ids(case_id: object) -> None:
    """Catch case identities outside the compatibility range of zero through seventeen."""

    with pytest.raises(ValueError, match="case_id"):
        EventEnvelope(
            "1.0",
            "event-1",
            "action_executed",
            "2026-09-10T12:00:00Z",
            case_id=cast(Any, case_id),
        )


@pytest.mark.parametrize("seed", [True, 1.5, "42"])
def test_event_envelope_requires_builtin_integer_seed(seed: object) -> None:
    """Catch non-reproducible seed representations."""

    with pytest.raises(ValueError, match="seed"):
        EventEnvelope(
            "1.0",
            "event-1",
            "action_executed",
            "2026-09-10T12:00:00Z",
            seed=cast(Any, seed),
        )


def test_event_envelope_requires_aware_ordered_validity_bounds() -> None:
    """Catch ambiguous or reversed validity windows."""

    base: dict[str, Any] = {
        "schema_version": "1.0",
        "event_id": "event-1",
        "event_type": "action_executed",
        "timestamp": "2026-09-10T12:00:00Z",
    }

    with pytest.raises(ValueError, match="valid_from"):
        EventEnvelope(**base, valid_from="2026-09-10T12:00:00")
    with pytest.raises(ValueError, match="valid_until"):
        EventEnvelope(
            **base,
            valid_from="2026-09-10T12:00:00Z",
            valid_until="2026-09-10T11:59:59Z",
        )

    EventEnvelope(
        **base,
        valid_from="2026-09-10T12:00:00Z",
        valid_until="2026-09-10T12:00:00+00:00",
    )


@pytest.mark.parametrize(
    "valid_from",
    [
        "2026-09-10T12:00:00Z",
        "2026-09-10T12:00:00+00:00",
        "2026-09-10T12:00:00.123456+05:30",
    ],
)
def test_event_envelope_accepts_rfc3339_offset_validity_bounds(valid_from: str) -> None:
    """Catch valid offset timestamps rejected before the validity comparison."""

    event = EventEnvelope(
        "1.0",
        "event-1",
        "action_executed",
        "2026-09-10T12:00:00Z",
        valid_from=valid_from,
    )

    assert event.valid_from == valid_from


@pytest.mark.parametrize(
    "valid_from",
    [
        "2026-09-10X12:00:00+00:00",
        "2026-09-10 12:00:00+00:00",
        "2026-09-10T12:00+00:00",
        "2026-09-10T12:00:00+0000",
        "2026-09-10T12:00:00z",
    ],
)
def test_event_envelope_rejects_non_rfc3339_validity_syntax(valid_from: str) -> None:
    """Catch parser-permitted datetime spellings outside the event contract."""

    with pytest.raises(ValueError, match="valid_from"):
        EventEnvelope(
            "1.0",
            "event-1",
            "action_executed",
            "2026-09-10T12:00:00Z",
            valid_from=valid_from,
        )


def test_action_bound_linkage_round_trips_through_current_normalizers() -> None:
    """Catch linkage metadata omitted or altered by current event serialization paths."""

    expected_linkage: dict[str, Any] = {
        "action_id": "action-1",
        "parent_event_ids": ("prediction-event-1",),
        "evidence_refs": ("evidence-1",),
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        "case_version": "17case-v5",
        "case_id": 17,
        "stripe_id": "baseline",
        "repeat_id": 0,
        "seed": 42,
        "authorization_scope": "sandbox",
        "verifier_refs": ("verifier-1",),
        "valid_from": "2026-09-10T12:00:00Z",
        "valid_until": "2026-09-10T12:05:00+00:00",
        "superseded_by": "assessment-event-2",
    }
    linkage_kwargs: dict[str, Any] = {
        **expected_linkage,
        "parent_event_ids": list(expected_linkage["parent_event_ids"]),
        "evidence_refs": list(expected_linkage["evidence_refs"]),
        "verifier_refs": list(expected_linkage["verifier_refs"]),
    }
    event = make_event_envelope(
        StructuredEventType.ACTION_EXECUTED,
        {"summary": "executed after a prediction"},
        **linkage_kwargs,
    )

    row = event.to_dict()
    normalized = normalize_trace_event(row)
    assert {field: row[field] for field in expected_linkage} == expected_linkage
    assert {field: normalized[field] for field in expected_linkage} == expected_linkage

    json_normalized = normalize_trace_event(json.loads(json.dumps(row)))
    assert json_normalized["parent_event_ids"] == ["prediction-event-1"]
    assert json_normalized["evidence_refs"] == ["evidence-1"]
    assert json_normalized["verifier_refs"] == ["verifier-1"]


def test_empty_and_none_action_bound_linkage_fields_have_stable_serialization() -> None:
    """Catch absent scalar linkage or empty reference collection serialization drift."""

    row = EventEnvelope("1.0", "event-1", "legacy_event", "2026-09-10T12:00:00Z").to_dict()

    for field_name in (
        "action_id",
        "model_version",
        "harness_version",
        "case_version",
        "case_id",
        "stripe_id",
        "repeat_id",
        "seed",
        "authorization_scope",
        "valid_from",
        "valid_until",
        "superseded_by",
    ):
        assert field_name not in row
    assert row["parent_event_ids"] == ()
    assert row["evidence_refs"] == ()
    assert row["verifier_refs"] == ()

    json_row = json.loads(json.dumps(row))
    assert json_row["parent_event_ids"] == []
    assert json_row["evidence_refs"] == []
    assert json_row["verifier_refs"] == []


def test_event_envelope_snapshots_empty_and_mutable_reference_collections() -> None:
    """Catch mutable caller collections leaking into immutable event records."""

    parents = ["prediction-event-1"]
    event = EventEnvelope(
        "1.0",
        "event-1",
        "action_executed",
        "2026-09-10T12:00:00Z",
        parent_event_ids=cast(Any, parents),
    )
    parents.append("unexpected-parent")

    assert event.parent_event_ids == ("prediction-event-1",)
    assert event.to_dict()["evidence_refs"] == ()


def test_event_envelope_retains_legacy_positional_payload_argument() -> None:
    """Catch a new field shifting the established positional payload slot."""

    legacy_args = cast(
        Any,
        (
            "1.0",
            "event-1",
            "legacy_event",
            "2026-09-10T12:00:00Z",
            *([None] * 12),
            {"summary": "legacy payload"},
        ),
    )
    event = EventEnvelope(*legacy_args)

    assert event.payload == {"summary": "legacy payload"}
    assert event.action_id is None
