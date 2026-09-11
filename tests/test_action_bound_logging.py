"""Contract tests for action-bound structured event envelopes."""

from __future__ import annotations

import json
from math import nan
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


@pytest.mark.parametrize(
    ("field_name", "valid_value", "invalid_value"),
    [
        ("seed", 9_007_199_254_740_991, 9_007_199_254_740_992),
        ("seed", -9_007_199_254_740_991, -9_007_199_254_740_992),
        ("repeat_id", 9_007_199_254_740_991, 9_007_199_254_740_992),
        ("seed", 0, 10**5000),
        ("repeat_id", 0, 10**5000),
    ],
    ids=["seed-max", "seed-min", "repeat-max", "seed-digit-overflow", "repeat-digit-overflow"],
)
def test_envelope_integer_metadata_uses_serialization_safe_json_range(
    field_name: str,
    valid_value: int,
    invalid_value: int,
) -> None:
    """Catch metadata integers that fail or lose precision in supported JSON consumers."""

    valid_kwargs: dict[str, Any] = {field_name: valid_value}
    EventEnvelope(
        "1.0",
        "event-1",
        StructuredEventType.PREDICTION_COMMIT.value,
        "2026-09-10T12:00:00Z",
        **valid_kwargs,
    )
    invalid_kwargs: dict[str, Any] = {field_name: invalid_value}
    with pytest.raises(ValueError, match=f"{field_name}.*serialization-safe"):
        EventEnvelope(
            "1.0",
            "event-1",
            StructuredEventType.PREDICTION_COMMIT.value,
            "2026-09-10T12:00:00Z",
            **invalid_kwargs,
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


def test_event_envelope_range_checks_rfc3339_numeric_offsets() -> None:
    """Catch offset components that parsers normalize beyond RFC3339 lexical bounds."""

    EventEnvelope(
        "1.0",
        "event-1",
        "action_executed",
        "2026-09-10T12:00:00Z",
        valid_from="2026-09-10T12:00:00+23:59",
    )
    for invalid_offset in ("+24:00", "+00:60", "-00:60"):
        with pytest.raises(ValueError, match="valid_from"):
            EventEnvelope(
                "1.0",
                "event-1",
                "action_executed",
                "2026-09-10T12:00:00Z",
                valid_from=f"2026-09-10T12:00:00{invalid_offset}",
            )


def test_event_envelope_caps_fractional_seconds_before_ordering() -> None:
    """Catch sub-microsecond truncation that can hide a reversed validity interval."""

    EventEnvelope(
        "1.0",
        "event-1",
        "action_executed",
        "2026-09-10T12:00:00Z",
        valid_from="2026-09-10T12:00:00.123456Z",
    )
    with pytest.raises(ValueError, match="valid_from"):
        EventEnvelope(
            "1.0",
            "event-1",
            "action_executed",
            "2026-09-10T12:00:00Z",
            valid_from="2026-09-10T12:00:00.0000009Z",
            valid_until="2026-09-10T12:00:00.0000001Z",
        )


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


def test_action_bound_envelope_deep_snapshots_and_thaws_payload() -> None:
    """Catch caller or returned-container mutation rewriting an action-bound event."""

    scores = [1]
    assessment: dict[str, object] = {"scores": scores}
    payload: dict[str, object] = {"assessment": assessment}
    event = EventEnvelope(
        "1.0",
        "event-1",
        StructuredEventType.EPISTEMIC_ASSESSMENT.value,
        "2026-09-10T12:00:00Z",
        payload=payload,
    )
    scores.append(2)
    assessment["status"] = "rewritten"
    payload["extra"] = True

    with pytest.raises(TypeError):
        cast(Any, event.payload)["assessment"] = "rewritten"
    with pytest.raises(TypeError):
        cast(Any, event.payload["assessment"])["scores"] = ()

    expected = {"assessment": {"scores": [1]}}
    first = event.to_dict()
    assert event.payload == expected
    assert not event.payload != expected
    frozen_scores = cast(Any, event.payload["assessment"])["scores"]
    assert frozen_scores == [1]
    assert not frozen_scores != [1]
    assert first["payload"] == expected
    assert type(first["payload"]) is dict
    assert type(cast(dict[str, object], first["payload"])["assessment"]) is dict
    cast(dict[str, Any], first["payload"])["assessment"]["scores"].append(3)

    assert event.to_dict()["payload"] == expected
    assert json.loads(json.dumps(event.to_dict()))["payload"] == expected


@pytest.mark.parametrize(
    "event_type",
    [StructuredEventType.EPISTEMIC_ASSESSMENT, StructuredEventType.CASE_ASSESSMENT],
)
def test_derived_assessment_envelopes_require_mapping_payloads(
    event_type: StructuredEventType,
) -> None:
    """Catch derived assessments that do not expose named, auditable fields."""

    with pytest.raises(ValueError, match="payload"):
        EventEnvelope(
            "1.0",
            "event-1",
            event_type.value,
            "2026-09-10T12:00:00Z",
            payload=cast(Any, ["not", "a", "mapping"]),
        )


def test_action_bound_helper_normalizes_nonmapping_payload_failure() -> None:
    """Catch a public helper leaking an incidental ``dict`` conversion TypeError."""

    with pytest.raises(ValueError, match="payload"):
        make_event_envelope(
            StructuredEventType.EPISTEMIC_ASSESSMENT,
            cast(Any, object()),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {1: "non-string key"},
        {"nonfinite": nan},
        {"unsupported": object()},
    ],
)
def test_derived_assessment_envelopes_reject_non_json_values(payload: object) -> None:
    """Catch assessment data that cannot cross the structured JSON logging boundary."""

    with pytest.raises(ValueError, match="payload"):
        EventEnvelope(
            "1.0",
            "event-1",
            StructuredEventType.EPISTEMIC_ASSESSMENT.value,
            "2026-09-10T12:00:00Z",
            payload=cast(Any, payload),
        )


def test_derived_assessment_envelopes_reject_cyclic_payloads() -> None:
    """Catch recursive assessment data before serialization recurses indefinitely."""

    payload: dict[str, object] = {}
    payload["cycle"] = payload

    with pytest.raises(ValueError, match="payload.*cycle"):
        EventEnvelope(
            "1.0",
            "event-1",
            StructuredEventType.CASE_ASSESSMENT.value,
            "2026-09-10T12:00:00Z",
            payload=payload,
        )


def test_derived_assessment_envelopes_reject_nested_noninteroperable_integers() -> None:
    """Catch huge assessment integers before an eventual serializer raises incidentally."""

    with pytest.raises(ValueError, match="payload.*serialization-safe"):
        EventEnvelope(
            "1.0",
            "event-1",
            StructuredEventType.EPISTEMIC_ASSESSMENT.value,
            "2026-09-10T12:00:00Z",
            payload={"assessment": {"sample_count": 10**5000}},
        )


def test_generic_envelope_retains_legacy_opaque_mutable_payload_behavior() -> None:
    """Catch action-bound hardening leaking into the legacy generic event contract."""

    opaque = object()
    payload: dict[str, object] = {"opaque": opaque, "values": []}
    event = EventEnvelope(
        "1.0",
        "event-1",
        "legacy_event",
        "2026-09-10T12:00:00Z",
        payload=payload,
    )
    cast(list[str], payload["values"]).append("still-shared")

    assert event.payload is payload
    assert event.payload["opaque"] is opaque
    assert event.payload["values"] == ["still-shared"]
    assert normalize_trace_event(event.to_dict())["event_type"] == "legacy_event"


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
