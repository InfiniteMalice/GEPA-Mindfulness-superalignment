"""Contracts separating observed world changes from evidence claims."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import (
    ArtifactObservation,
    EvidenceClaim,
    EvidenceState,
    WorldStateChange,
)

AFTER_DIGEST = "a" * 64
BEFORE_DIGEST = "b" * 64
OBSERVED_AT = "2026-09-10T12:00:00Z"


def _reference(reference_id: str = "output:bug-fix") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _observation(
    observation_id: str = "observation-after",
    *,
    artifact_ref: str = "artifact:fix.patch",
    digest: str = AFTER_DIGEST,
    observed_at: str = OBSERVED_AT,
) -> ArtifactObservation:
    return ArtifactObservation(observation_id, artifact_ref, digest, observed_at, (_reference(),))


def _claim(
    claim_id: str,
    *,
    status: str = "unverified",
    superseded_by: str | None = None,
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        proposition="I fixed the bug",
        evidence_refs=(),
        status=cast(Any, status),
        superseded_by=superseded_by,
    )


def test_success_statement_is_an_unverified_claim_not_a_world_change() -> None:
    """Catch a model success statement being promoted to observed world state."""

    claim = _claim("claim-1")
    state = EvidenceState((claim,))

    assert claim.proposition == "I fixed the bug"
    assert claim.status == "unverified"
    assert state.resolve("claim-1") == claim
    assert not isinstance(claim, WorldStateChange)
    assert not hasattr(state, "world")


def test_world_change_requires_action_artifact_digest_and_observation_time() -> None:
    """Catch purported world changes without complete observed artifact linkage."""

    change = WorldStateChange(
        "change-1",
        "action-1",
        _observation("observation-before", digest=BEFORE_DIGEST),
        _observation(),
    )

    assert change.action_id == "action-1"
    assert change.artifact_ref == "artifact:fix.patch"
    assert change.after_digest == AFTER_DIGEST
    assert change.observed_at == OBSERVED_AT


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("observation_id", " "),
        ("artifact_ref", " "),
        ("digest", " "),
        ("observed_at", " "),
    ],
)
def test_world_change_rejects_missing_required_observation_fields(
    field_name: str,
    value: object,
) -> None:
    """Catch incomplete world-state records that cannot establish an observation."""

    values: dict[str, object] = {
        "observation_id": "observation-after",
        "artifact_ref": "artifact:fix.patch",
        "digest": AFTER_DIGEST,
        "observed_at": OBSERVED_AT,
        "evidence_refs": (_reference(),),
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        ArtifactObservation(**cast(Any, values))


@pytest.mark.parametrize(
    "field_name",
    ["observation_id", "artifact_ref", "digest", "observed_at"],
)
def test_world_change_rejects_string_subclasses(field_name: str) -> None:
    """Catch subclasses bypassing exact typed validation at the state boundary."""

    class StringSubclass(str):
        pass

    values: dict[str, object] = {
        "observation_id": "observation-after",
        "artifact_ref": "artifact:fix.patch",
        "digest": AFTER_DIGEST,
        "observed_at": OBSERVED_AT,
        "evidence_refs": (_reference(),),
    }
    values[field_name] = StringSubclass(cast(str, values[field_name]))

    with pytest.raises(ValueError, match=field_name):
        ArtifactObservation(**cast(Any, values))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("digest", "A" * 64),
        ("digest", "a" * 63),
        ("digest", "g" * 64),
        ("digest", "a" * 65),
    ],
)
def test_world_change_requires_canonical_sha256_digests(
    field_name: str,
    value: object,
) -> None:
    """Catch artifact digests that are ambiguous across storage consumers."""

    values: dict[str, object] = {
        "observation_id": "observation-after",
        "artifact_ref": "artifact:fix.patch",
        "digest": AFTER_DIGEST,
        "observed_at": OBSERVED_AT,
        "evidence_refs": (_reference(),),
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        ArtifactObservation(**cast(Any, values))


@pytest.mark.parametrize(
    "observed_at",
    [
        "2026-09-10 12:00:00Z",
        "2026-09-10T12:00:00",
        "2026-09-10T12:00:00z",
        "2026-09-10T12:00:00+24:00",
        "2026-09-10T12:00:00.1234567Z",
        "2026-02-30T12:00:00Z",
    ],
)
def test_world_change_rejects_non_rfc3339_observation_time(observed_at: str) -> None:
    """Catch parser-permitted or impossible times outside the event timestamp contract."""

    with pytest.raises(ValueError, match="observed_at"):
        ArtifactObservation(
            "observation-after",
            "artifact:fix.patch",
            AFTER_DIGEST,
            observed_at,
            (_reference(),),
        )


def test_world_change_is_frozen_slotted_and_json_round_trips() -> None:
    """Catch mutable or lossy observed-world records at the serialization boundary."""

    after = ArtifactObservation(
        "observation-after",
        "artifact:fix.patch",
        AFTER_DIGEST,
        "2026-09-10T12:00:00.123456+23:59",
        (_reference(),),
    )
    change = WorldStateChange(
        "change-1",
        "action-1",
        None,
        after,
    )
    expected = {
        "change_id": "change-1",
        "action_id": "action-1",
        "before_observation": None,
        "after_observation": after.to_dict(),
    }

    assert not hasattr(change, "__dict__")
    assert change.to_dict() == expected
    assert WorldStateChange.from_dict(json.loads(json.dumps(expected))) == change
    with pytest.raises(FrozenInstanceError):
        change.action_id = "action-2"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("claim_id", " "),
        ("proposition", " "),
        ("status", "unknown"),
        ("status", 1),
        ("superseded_by", " "),
    ],
)
def test_evidence_claim_rejects_invalid_exact_fields(field_name: str, value: object) -> None:
    """Catch unauditable claim identifiers, propositions, status, or links."""

    values: dict[str, object] = {
        "claim_id": "claim-1",
        "proposition": "I fixed the bug",
        "evidence_refs": (),
        "status": "unverified",
        "superseded_by": None,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        EvidenceClaim(**cast(Any, values))


def test_evidence_claim_detaches_and_revalidates_canonical_evidence_refs() -> None:
    """Catch caller-owned collections or later reference corruption rewriting a claim."""

    original = _reference()
    references = [original]
    claim = EvidenceClaim(
        "claim-1",
        "I fixed the bug",
        cast(Any, references),
        "supported",
    )
    references.clear()
    object.__setattr__(original, "reference_id", "corrupted")

    assert claim.evidence_refs == (_reference(),)
    assert claim.evidence_refs[0] is not original
    assert not hasattr(claim, "__dict__")


def test_evidence_claim_rejects_reference_subclasses_and_corruption() -> None:
    """Catch subclass or object-level mutation bypassing evidence-reference validation."""

    class ReferenceSubclass(EvidenceReference):
        pass

    subclass = ReferenceSubclass("output:bug-fix", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    corrupted = _reference()
    object.__setattr__(corrupted, "source_kind", "observable_output")

    for reference in (subclass, corrupted):
        with pytest.raises(ValueError, match="evidence_refs"):
            EvidenceClaim(
                "claim-1",
                "I fixed the bug",
                cast(Any, (reference,)),
                "supported",
            )


def test_evidence_claim_rejects_hostile_reference_id_string_subclass() -> None:
    """Catch an empty string subclass whose overridden strip method claims content."""

    class HostileEmptyString(str):
        def strip(self, chars: str | None = None) -> str:
            return "pretend-valid"

    hostile_id = HostileEmptyString("")
    hostile_reference = EvidenceReference(
        hostile_id,
        EvidenceSourceKind.OBSERVABLE_OUTPUT,
    )
    assert json.loads(json.dumps(hostile_reference.to_dict()))["reference_id"] == ""

    with pytest.raises(ValueError, match="evidence_refs.*reference_id"):
        EvidenceClaim(
            "claim-1",
            "I fixed the bug",
            (hostile_reference,),
            "supported",
        )


def test_evidence_claim_from_dict_rejects_hostile_reference_id_string_subclass() -> None:
    """Catch deserialization preserving a deceptive noncanonical reference identifier."""

    class HostileEmptyString(str):
        def strip(self, chars: str | None = None) -> str:
            return "pretend-valid"

    payload = {
        "claim_id": "claim-1",
        "proposition": "I fixed the bug",
        "evidence_refs": [
            {
                "reference_id": HostileEmptyString(""),
                "source_kind": "observable_output",
            }
        ],
        "status": "supported",
        "superseded_by": None,
    }

    with pytest.raises(ValueError, match="evidence_refs.*reference_id"):
        EvidenceClaim.from_dict(payload)


def test_evidence_claim_refuses_to_serialize_later_hostile_reference_corruption() -> None:
    """Catch object-level mutation smuggling an empty identifier into persisted evidence."""

    class HostileEmptyString(str):
        def strip(self, chars: str | None = None) -> str:
            return "pretend-valid"

    claim = EvidenceClaim(
        "claim-1",
        "I fixed the bug",
        (_reference(),),
        "supported",
    )
    object.__setattr__(claim.evidence_refs[0], "reference_id", HostileEmptyString(""))

    with pytest.raises(ValueError, match="evidence_refs.*reference_id"):
        claim.to_dict()


def test_supersession_preserves_original_claim_and_resolves_terminal_claim() -> None:
    """Catch supersession rewriting history or stopping before the terminal claim."""

    original = _claim("claim-1", status="superseded", superseded_by="claim-2")
    correction = EvidenceClaim(
        "claim-2",
        "The bug remains reproducible",
        (_reference("output:reproduction"),),
        "contradicted",
    )
    state = EvidenceState(cast(Any, [original, correction]))

    assert state.claims == (original, correction)
    assert state.claims[0].proposition == "I fixed the bug"
    assert state.resolve("claim-1") == correction
    assert state.resolve("claim-2") == correction


def test_evidence_state_snapshots_claims_before_validating_graph() -> None:
    """Catch later object-level mutation of caller claims changing the evidence graph."""

    original = _claim("claim-1", status="superseded", superseded_by="claim-2")
    correction = _claim("claim-2")
    state = EvidenceState((original, correction))
    object.__setattr__(original, "superseded_by", "missing")
    object.__setattr__(correction, "proposition", "rewritten")

    resolved = state.resolve("claim-1")
    assert resolved.claim_id == "claim-2"
    assert resolved.proposition == "I fixed the bug"
    assert resolved is not correction


def test_evidence_state_rejects_duplicate_dangling_self_and_cyclic_links() -> None:
    """Catch ambiguous or invalid supersession graphs before resolution."""

    duplicate = (_claim("claim-1"), _claim("claim-1"))
    dangling = (_claim("claim-1", status="superseded", superseded_by="missing"),)
    cycle = (
        _claim("claim-1", status="superseded", superseded_by="claim-2"),
        _claim("claim-2", status="superseded", superseded_by="claim-1"),
    )

    with pytest.raises(ValueError, match="unique"):
        EvidenceState(duplicate)
    with pytest.raises(ValueError, match="dangling"):
        EvidenceState(dangling)
    with pytest.raises(ValueError, match="itself"):
        _claim("claim-1", status="superseded", superseded_by="claim-1")
    with pytest.raises(ValueError, match="cycle"):
        EvidenceState(cycle)


@pytest.mark.parametrize(
    ("status", "superseded_by"),
    [
        ("superseded", None),
        ("unverified", "claim-2"),
        ("supported", "claim-2"),
        ("contradicted", "claim-2"),
    ],
)
def test_evidence_claim_requires_status_and_supersession_link_to_agree(
    status: str,
    superseded_by: str | None,
) -> None:
    """Catch links whose active/superseded status is internally contradictory."""

    with pytest.raises(ValueError, match="superseded"):
        _claim("claim-1", status=status, superseded_by=superseded_by)


@pytest.mark.parametrize("status", ["supported", "contradicted"])
def test_evidentiary_claim_status_requires_at_least_one_reference(status: str) -> None:
    """Catch affirmative evidence states that do not identify supporting evidence."""

    with pytest.raises(ValueError, match="requires evidence_refs"):
        _claim("claim-1", status=status)


@pytest.mark.parametrize("status", ["supported", "contradicted"])
def test_evidence_state_rejects_coherent_empty_evidence_escalation(status: str) -> None:
    """Catch use-time mutation promoting an unverified claim without evidence."""

    state = EvidenceState((_claim("claim-1"),))
    object.__setattr__(state.claims[0], "status", status)

    with pytest.raises(ValueError, match="requires evidence_refs"):
        state.resolve("claim-1")
    with pytest.raises(ValueError, match="requires evidence_refs"):
        state.to_dict()


def test_evidence_state_resolve_rejects_unknown_claim_without_mutating() -> None:
    """Catch implicit claim creation or state mutation during failed resolution."""

    state = EvidenceState((_claim("claim-1"),))
    before = state.to_dict()

    with pytest.raises(KeyError, match="missing"):
        state.resolve("missing")

    assert state.to_dict() == before


def test_evidence_state_is_frozen_slotted_and_json_round_trips() -> None:
    """Catch mutable or lossy evidence-state records at the serialization boundary."""

    state = EvidenceState(
        (
            EvidenceClaim(
                "claim-1",
                "I fixed the bug",
                (_reference(),),
                "superseded",
                "claim-2",
            ),
            EvidenceClaim(
                "claim-2",
                "The fix is observed",
                (_reference("output:verification"),),
                "supported",
            ),
        )
    )
    payload = json.loads(json.dumps(state.to_dict()))
    restored = EvidenceState.from_dict(payload)

    assert restored == state
    assert restored.resolve("claim-1").claim_id == "claim-2"
    assert not hasattr(restored, "__dict__")
    with pytest.raises(FrozenInstanceError):
        restored.claims = ()


@pytest.mark.parametrize(
    ("constructor", "payload"),
    [
        (ArtifactObservation.from_dict, []),
        (ArtifactObservation.from_dict, {"observation_id": "observation-1"}),
        (WorldStateChange.from_dict, []),
        (WorldStateChange.from_dict, {"change_id": "change-1"}),
        (EvidenceClaim.from_dict, []),
        (EvidenceClaim.from_dict, {"claim_id": "claim-1"}),
        (EvidenceState.from_dict, []),
        (EvidenceState.from_dict, {"claims": [], "extra": True}),
    ],
)
def test_state_deserializers_reject_wrong_shapes_and_fields(
    constructor: Any,
    payload: object,
) -> None:
    """Catch partial or extension fields silently changing the persisted contract."""

    with pytest.raises(ValueError):
        constructor(payload)
