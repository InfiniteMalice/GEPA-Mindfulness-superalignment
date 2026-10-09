"""Public perspective provenance rejects forged authority and stale joins."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from gepa_mindfulness.verification.perspective_records import Perspective, Stakeholder
from gepa_mindfulness.verification.state import EvidenceClaim
from semantic_intent_robustness.perspective_protocol import (
    PerspectiveCandidate,
    PerspectiveCapture,
    PerspectivePlan,
    PerspectiveSlot,
    PerspectiveSource,
    RoleAssignment,
    perspective_digest,
)


def public_plan():
    """Supply a public record with two equally authorized stakeholders."""
    refs = (EvidenceReference("source", EvidenceSourceKind.EXTERNAL_RECORD),)
    source = PerspectiveSource(
        "after",
        "core",
        '[{"content":"Report material risks","role":"user"}]',
        (EvidenceClaim("fact", "There is an unresolved risk", refs, "unverified"),),
        (EvidenceClaim("constraint", "Disclose material risks", refs, "unverified"),),
        tuple(
            RoleAssignment(actor, "reviewer", ("read",), ("disclose",), ("review",))
            for actor in ("a", "b")
        ),
        refs,
        TrainingEligibility.DEVELOPMENT,
    )
    stakeholders = tuple(
        Stakeholder(
            actor,
            "reviewer",
            "direct",
            "explicit",
            refs,
            ("accuracy",),
            ("disclose",),
            ("clarity",),
            refs,
            None,
        )
        for actor in ("a", "b")
    )
    slots = tuple(
        PerspectiveSlot(
            actor,
            Perspective(
                actor,
                "core",
                content_digest([c.to_dict() for c in source.facts]),
                "affected_party",
                (actor,),
                False,
                "after",
            ),
        )
        for actor in ("a", "b")
    )
    return PerspectivePlan(source, stakeholders, slots)


def candidate(slot="a"):
    """Create a simulated preference with observable generation provenance."""
    return PerspectiveCandidate(
        slot, "Disclose risks", ("clarity",), "toy", "1", 0.5, public_plan().source.source_refs
    )


def test_record_roundtrip_detaches_inputs():
    """Mutating input or serialized nested data cannot alter a retained plan."""
    plan = public_plan()
    restored = PerspectivePlan.from_dict(plan.to_dict())
    object.__setattr__(plan.stakeholders[0], "interests", ("changed",))
    assert restored.stakeholders[0].interests == ("accuracy",)
    data = restored.to_dict()
    data["source"]["facts"][0]["proposition"] = "changed"
    assert restored.source.facts[0].proposition == "There is an unresolved risk"
    cap = PerspectiveCapture(perspective_digest(restored), (candidate(),), "observed", "captured")
    assert PerspectiveCapture.from_dict(cap.to_dict()) == cap


@pytest.mark.parametrize("kind", ["source", "core", "facts", "stakeholder", "slot", "changed"])
def test_invalid_slot_and_claim_joins_rejected(kind):
    """A slot cannot detach itself from the declared source or stakeholders."""
    p = public_plan()
    changes = {
        "source": {"source_variant_id": "other"},
        "core": {"semantic_core_id": "other"},
        "facts": {"material_facts_id": "0" * 64},
        "stakeholder": {"stakeholder_ids": ("x",)},
        "slot": {"variant_id": "x"},
        "changed": {"material_facts_changed": True},
    }
    with pytest.raises(ValueError):
        replace(
            p,
            slots=(
                replace(p.slots[0], perspective=replace(p.slots[0].perspective, **changes[kind])),
            ),
        )


def test_duplicate_claim_and_role_ids_rejected():
    """Duplicate identities would make independent references ambiguous."""
    p = public_plan()
    with pytest.raises(ValueError):
        replace(p.source, constraints=p.source.facts)
    with pytest.raises(ValueError):
        replace(p.source, roles=p.source.roles * 2)
    with pytest.raises(ValueError):
        replace(p, stakeholders=p.stakeholders[:1])


def test_limits_are_exact():
    """The declared roster cannot evade bounds via empty or oversized input."""
    p = public_plan()
    for count in (0, 17):
        with pytest.raises(ValueError):
            replace(
                p,
                stakeholders=tuple(
                    replace(p.stakeholders[0], stakeholder_id=str(i)) for i in range(count)
                ),
            )
    slots = tuple(
        PerspectiveSlot(str(i), replace(p.slots[0].perspective, variant_id=str(i)))
        for i in range(65)
    )
    assert len(replace(p, slots=slots[:64]).slots) == 64
    with pytest.raises(ValueError):
        replace(p, slots=slots)
    assert replace(p, slots=()).slots == ()


def test_source_restrictions_survive_serialization():
    """An outer TRAIN label cannot erase nested held-out admission."""
    p = public_plan()
    p = replace(
        p, source=replace(p.source, source_training_eligibility=TrainingEligibility.HIDDEN_EVAL)
    )
    restored = PerspectivePlan.from_dict(p.to_dict())
    assert restored.source.source_training_eligibility is TrainingEligibility.HIDDEN_EVAL
    with pytest.raises(ValueError):
        require_training_eligible({"training_eligibility": "TRAIN", "source_record": p.to_dict()})
    with pytest.raises(ValueError):
        replace(p.source, source_training_eligibility=TrainingEligibility.TRAIN)


@pytest.mark.parametrize(
    "field,value",
    [
        ("verified", True),
        ("schema_version", "future"),
        ("uncertainty", float("nan")),
        ("uncertainty", True),
    ],
)
def test_candidates_cannot_claim_verified_status(field, value):
    """Unrecognized authority fields or malformed confidence cannot enter public capture."""
    data = candidate().to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        PerspectiveCandidate.from_dict(data)


def test_source_cannot_assert_truth_and_censored_capture_cannot_assert_candidates():
    """Source assertions and incomplete capture never manufacture established evidence."""
    p = public_plan()
    with pytest.raises(ValueError):
        replace(p.source, facts=(replace(p.source.facts[0], status="supported"),))
    with pytest.raises(ValueError):
        PerspectiveCapture(perspective_digest(p), (candidate(),), "censored", "not run")
