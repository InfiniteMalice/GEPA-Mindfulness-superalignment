"""Contracts separating observed world changes from evidence claims."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import EvidenceClaim, EvidenceState, WorldStateChange

AFTER_DIGEST = "a" * 64
BEFORE_DIGEST = "b" * 64
OBSERVED_AT = "2026-09-10T12:00:00Z"


def _reference(reference_id: str = "output:bug-fix") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)


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
        change_id="change-1",
        action_id="action-1",
        artifact_ref="artifact:fix.patch",
        before_digest=BEFORE_DIGEST,
        after_digest=AFTER_DIGEST,
        observed_at=OBSERVED_AT,
    )

    assert change.action_id == "action-1"
    assert change.artifact_ref == "artifact:fix.patch"
    assert change.after_digest == AFTER_DIGEST
    assert change.observed_at == OBSERVED_AT


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("change_id", " "),
        ("action_id", " "),
        ("artifact_ref", " "),
        ("after_digest", " "),
        ("observed_at", " "),
    ],
)
def test_world_change_rejects_missing_required_observation_fields(
    field_name: str,
    value: object,
) -> None:
    """Catch incomplete world-state records that cannot establish an observation."""

    values: dict[str, object] = {
        "change_id": "change-1",
        "action_id": "action-1",
        "artifact_ref": "artifact:fix.patch",
        "before_digest": None,
        "after_digest": AFTER_DIGEST,
        "observed_at": OBSERVED_AT,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        WorldStateChange(**cast(Any, values))


@pytest.mark.parametrize(
    "field_name",
    ["change_id", "action_id", "artifact_ref", "after_digest", "observed_at"],
)
def test_world_change_rejects_string_subclasses(field_name: str) -> None:
    """Catch subclasses bypassing exact typed validation at the state boundary."""

    class StringSubclass(str):
        pass

    values: dict[str, object] = {
        "change_id": "change-1",
        "action_id": "action-1",
        "artifact_ref": "artifact:fix.patch",
        "before_digest": None,
        "after_digest": AFTER_DIGEST,
        "observed_at": OBSERVED_AT,
    }
    values[field_name] = StringSubclass(cast(str, values[field_name]))

    with pytest.raises(ValueError, match=field_name):
        WorldStateChange(**cast(Any, values))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("before_digest", "A" * 64),
        ("before_digest", "a" * 63),
        ("after_digest", "g" * 64),
        ("after_digest", "a" * 65),
    ],
)
def test_world_change_requires_canonical_sha256_digests(
    field_name: str,
    value: object,
) -> None:
    """Catch artifact digests that are ambiguous across storage consumers."""

    values: dict[str, object] = {
        "change_id": "change-1",
        "action_id": "action-1",
        "artifact_ref": "artifact:fix.patch",
        "before_digest": BEFORE_DIGEST,
        "after_digest": AFTER_DIGEST,
        "observed_at": OBSERVED_AT,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        WorldStateChange(**cast(Any, values))


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
        WorldStateChange(
            "change-1",
            "action-1",
            "artifact:fix.patch",
            None,
            AFTER_DIGEST,
            observed_at,
        )


def test_world_change_is_frozen_slotted_and_json_round_trips() -> None:
    """Catch mutable or lossy observed-world records at the serialization boundary."""

    change = WorldStateChange(
        "change-1",
        "action-1",
        "artifact:fix.patch",
        None,
        AFTER_DIGEST,
        "2026-09-10T12:00:00.123456+23:59",
    )
    expected = {
        "change_id": "change-1",
        "action_id": "action-1",
        "artifact_ref": "artifact:fix.patch",
        "before_digest": None,
        "after_digest": AFTER_DIGEST,
        "observed_at": "2026-09-10T12:00:00.123456+23:59",
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
