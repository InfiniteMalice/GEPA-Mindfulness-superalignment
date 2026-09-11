"""Typed routing and immutable proposal tests for controlled learning surfaces."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from gepa_mindfulness import (
    LearningSurface,
    LessonCharacteristics,
    LessonKind,
    LessonProposal,
    LessonReviewStatus,
    classify_learning_surface,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind


def _observable_ref(reference_id: str = "output:lesson-1") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _proposal(**overrides: object) -> LessonProposal:
    values: dict[str, object] = {
        "lesson_id": "lesson-1",
        "summary": "The stable action convention reduced execution errors.",
        "primary_destination": LearningSurface.HARNESS,
        "evidence_refs": (_observable_ref(),),
        "rationale": "Repeated executions provide observable support for the convention.",
        "reversible": True,
        "review_status": LessonReviewStatus.PENDING,
    }
    values.update(overrides)
    return LessonProposal(**cast(Any, values))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (LessonKind.ONE_OFF_OBSERVATION, LearningSurface.TRACE_ONLY),
        (LessonKind.EPISODE_FACT, LearningSurface.MEMORY),
        (LessonKind.STABLE_PROCEDURAL_CONVENTION, LearningSurface.HARNESS),
        (LessonKind.REUSABLE_DEPENDENCY, LearningSurface.SKILL_GRAPH),
        (LessonKind.PERSISTENT_INTRINSIC_BEHAVIOR, LearningSurface.MODEL),
    ],
)
def test_classifier_uses_explicit_typed_decision_table(
    kind: LessonKind,
    expected: LearningSurface,
) -> None:
    characteristics = LessonCharacteristics(kind=kind)

    assert classify_learning_surface(characteristics) is expected


@pytest.mark.parametrize("flag", ["normative", "ambiguous", "difficult_to_reverse"])
def test_human_review_overrides_automatic_learning_destinations(flag: str) -> None:
    values = {
        "kind": LessonKind.REUSABLE_DEPENDENCY,
        "normative": False,
        "ambiguous": False,
        "difficult_to_reverse": False,
    }
    values[flag] = True

    characteristics = LessonCharacteristics(**cast(Any, values))

    assert classify_learning_surface(characteristics) is LearningSurface.HUMAN


def test_one_off_observation_cannot_route_automatically_to_model() -> None:
    characteristics = LessonCharacteristics(kind=LessonKind.ONE_OFF_OBSERVATION)

    destination = classify_learning_surface(characteristics)

    assert destination is LearningSurface.TRACE_ONLY
    assert destination is not LearningSurface.MODEL


def test_classifier_does_not_read_prose_keywords() -> None:
    characteristics = LessonCharacteristics(kind=LessonKind.ONE_OFF_OBSERVATION)

    first = classify_learning_surface(characteristics)
    second = classify_learning_surface(characteristics)

    assert first is second is LearningSurface.TRACE_ONLY


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("lesson_id", " "),
        ("summary", " "),
        ("rationale", " "),
        ("primary_destination", (LearningSurface.HARNESS, LearningSurface.MODEL)),
        ("primary_destination", [LearningSurface.HARNESS]),
        ("primary_destination", "harness"),
        ("reversible", 1),
        ("review_status", "pending"),
    ],
)
def test_proposal_rejects_missing_or_noncanonical_scalar_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _proposal(**{field_name: value})


@pytest.mark.parametrize("field_name", ["lesson_id", "summary", "rationale"])
def test_proposal_rejects_string_subclasses(field_name: str) -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match=field_name):
        _proposal(**{field_name: StringSubclass("apparently valid")})


def test_proposal_requires_observable_evidence() -> None:
    private_ref = EvidenceReference("reasoning:lesson-1", EvidenceSourceKind.PRIVATE_REASONING)

    with pytest.raises(ValueError, match="observable evidence_refs"):
        _proposal(evidence_refs=())
    with pytest.raises(ValueError, match="observable evidence_refs"):
        _proposal(evidence_refs=(private_ref,))


def test_proposal_rejects_duplicate_evidence_reference_ids() -> None:
    duplicate = _observable_ref()

    with pytest.raises(ValueError, match="unique reference_id"):
        _proposal(evidence_refs=(duplicate, duplicate))


def test_proposal_detaches_caller_collections_and_reference_aliases() -> None:
    original = _observable_ref()
    references = [original]

    proposal = _proposal(evidence_refs=references)
    references.clear()
    object.__setattr__(original, "reference_id", "corrupted")

    assert proposal.evidence_refs == (_observable_ref(),)
    assert proposal.evidence_refs[0] is not original


def test_proposal_rejects_hostile_container_and_reference_subclasses() -> None:
    class ListSubclass(list[EvidenceReference]):
        pass

    class ReferenceSubclass(EvidenceReference):
        pass

    with pytest.raises(ValueError, match="evidence_refs"):
        _proposal(evidence_refs=ListSubclass([_observable_ref()]))
    with pytest.raises(ValueError, match="exact EvidenceReference"):
        _proposal(
            evidence_refs=(
                ReferenceSubclass(
                    "output:lesson-1",
                    EvidenceSourceKind.OBSERVABLE_OUTPUT,
                ),
            )
        )


def test_proposal_rejects_hostile_reference_field_corruption() -> None:
    reference = _observable_ref()
    object.__setattr__(reference, "source_kind", "observable_output")

    with pytest.raises(ValueError, match="source_kind"):
        _proposal(evidence_refs=(reference,))


def test_proposal_revalidates_use_time_mutation_before_serializing() -> None:
    proposal = _proposal()
    object.__setattr__(proposal.evidence_refs[0], "reference_id", " ")

    with pytest.raises(ValueError, match="reference_id"):
        proposal.to_dict()


def test_lesson_characteristics_reject_ambiguous_or_hostile_values() -> None:
    with pytest.raises(ValueError, match="kind"):
        LessonCharacteristics(kind=cast(Any, "episode_fact"))
    with pytest.raises(ValueError, match="kind"):
        LessonCharacteristics(
            kind=cast(
                Any,
                (LessonKind.EPISODE_FACT, LessonKind.REUSABLE_DEPENDENCY),
            )
        )
    with pytest.raises(ValueError, match="normative"):
        LessonCharacteristics(kind=LessonKind.EPISODE_FACT, normative=cast(Any, 1))


def test_classifier_revalidates_use_time_characteristic_mutation() -> None:
    characteristics = LessonCharacteristics(kind=LessonKind.EPISODE_FACT)
    object.__setattr__(characteristics, "ambiguous", 1)

    with pytest.raises(ValueError, match="ambiguous"):
        classify_learning_surface(characteristics)


def test_proposal_is_frozen_slotted_and_json_round_trips() -> None:
    proposal = _proposal(review_status=LessonReviewStatus.APPROVED)
    payload = json.loads(json.dumps(proposal.to_dict()))

    restored = LessonProposal.from_dict(payload)

    assert restored == proposal
    assert restored.review_status is LessonReviewStatus.APPROVED
    assert restored.primary_destination is LearningSurface.HARNESS
    assert not hasattr(restored, "__dict__")
    with pytest.raises(FrozenInstanceError):
        restored.rationale = "rewritten"


@pytest.mark.parametrize(
    ("constructor", "payload"),
    [
        (LessonProposal.from_dict, []),
        (LessonProposal.from_dict, {"lesson_id": "lesson-1"}),
        (
            LessonProposal.from_dict,
            {
                "lesson_id": "lesson-1",
                "summary": "summary",
                "primary_destination": "harness",
                "evidence_refs": [],
                "rationale": "rationale",
                "reversible": True,
                "review_status": "pending",
                "extra": True,
            },
        ),
        (LessonCharacteristics.from_dict, []),
        (LessonCharacteristics.from_dict, {"kind": "episode_fact"}),
    ],
)
def test_learning_deserializers_reject_wrong_shapes_and_fields(
    constructor: Any,
    payload: object,
) -> None:
    with pytest.raises(ValueError):
        constructor(payload)


def test_lesson_characteristics_json_round_trip_is_deterministic() -> None:
    characteristics = LessonCharacteristics(
        kind=LessonKind.PERSISTENT_INTRINSIC_BEHAVIOR,
        normative=False,
        ambiguous=False,
        difficult_to_reverse=False,
    )

    payload = characteristics.to_dict()
    restored = LessonCharacteristics.from_dict(json.loads(json.dumps(payload)))

    assert restored == characteristics
    assert classify_learning_surface(restored) is LearningSurface.MODEL


def test_public_exports_preserve_existing_package_exports() -> None:
    from gepa_mindfulness import AggregateResult, PracticeSession

    assert AggregateResult.__name__ == "AggregateResult"
    assert PracticeSession.__name__ == "PracticeSession"
