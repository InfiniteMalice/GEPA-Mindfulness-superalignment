"""Matched public traces distinguish retention from use without attributing motive."""

from dataclasses import replace

import pytest
from test_epistemic_continuity import commitment, event_sequence, typed_verification
from test_epistemic_reconciliation import reconciliation, sequence
from test_evidence_use import assessment

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.evidence_use import MemoryInfluence, MemoryKind
from gepa_mindfulness.verification.state import EvidenceClaim, EvidenceState
from mindful_trace_gepa.logging_schema import EventEnvelope
from semantic_intent_robustness.epistemic_records import CommitmentStatus, CommitmentUpdate
from semantic_intent_robustness.peo_continuity import (
    PEOContinuityRequest,
    ProspectiveEvidenceUse,
    RetrospectiveEvidenceUse,
    audit_peo_continuity,
    evidence_use_digest,
)


def request(
    *,
    before=None,
    after=None,
    active=("k",),
    kind=MemoryKind.FACT,
    target=MemoryInfluence.BOUND,
    **changes,
):
    item = commitment()
    use = assessment(
        state=EvidenceState(
            (EvidenceClaim("k", item.claim_summary, item.evidence_refs, "observed"),)
        ),
        claim_id="k",
        memory=item.memory,
        kind=kind,
        target_influence=target,
        measurement=replace(
            reconciliation().update.measurements[0], evidence_refs=item.evidence_refs
        ),
    )
    binding = dict(
        commitment_digest=item.digest,
        evidence_use_digest=evidence_use_digest(use),
        provenance=("host-capture",),
    )
    prospective = ProspectiveEvidenceUse(
        **binding,
        **(
            dict(
                recognized=True,
                retained=True,
                retrieved=True,
                observed_kind=kind,
                observed_influence=MemoryInfluence.BOUND,
                prediction_reflected=True,
            )
            | (before or {})
        ),
    )
    retrospective = RetrospectiveEvidenceUse(
        **binding,
        **(
            dict(action_reflected=True, preserved_after_outcome=True, reported_unavailable=False)
            | (after or {})
        ),
    )
    chain = [replace(e, conversation_id="c", checkpoint_step=1) for e in sequence()]
    pred = dict(chain[0].payload)
    pred["predicted_outcome"] = dict(pred["predicted_outcome"]) | {
        "continuity_evidence": {"k": prospective.to_dict()}
    }
    chain[0] = replace(chain[0], payload=pred)
    post = EventEnvelope(
        schema_version="1.0",
        event_id="post",
        event_type="epistemic_assessment",
        timestamp="2026-09-30T12:00:06Z",
        run_id="run",
        repeat_id=0,
        conversation_id="c",
        checkpoint_step=1,
        model_version="model",
        harness_version="harness",
        action_id="action",
        parent_event_ids=("v",),
        payload={"continuity_evidence": {"k": retrospective.to_dict()}},
    )
    later = tuple(replace(e, timestamp="2026-09-30T12:00:07Z") for e in event_sequence()[-2:])
    events = event_sequence()[:5] + tuple(chain) + (post,) + later
    return replace(
        PEOContinuityRequest(
            events=events,
            decision_event_id="proposed-2",
            reconciliation_event_id="r",
            assessment_event_id="post",
            commitments=(item,),
            commitment_id="k",
            evidence_use=use,
            active_commitment_ids=active,
            provenance=("audit-host",),
        ),
        **changes,
    )


def audit(**kwargs):
    result = audit_peo_continuity(request(**kwargs), enabled=True)
    assert result is not None
    return result


def test_disabled_is_noop_even_for_unreadable_request():
    assert audit_peo_continuity(object()) is None


def test_retained_evidence_requires_observed_influence_on_prediction_and_action():
    result = audit()
    assert result.classification == "consistent"
    assert result.residuals == (("measurement", 2.0),)
    assert result.before.recognized is True
    assert result.after.action_reflected is True
    assert result.to_dict()["causal_or_motive_claim"] is False
    failed = audit(after={"action_reflected": False})
    assert failed.continuity.continuity_status == "consistent"
    assert failed.classification == "influence_failure"


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ({"retained": False, "retrieved": False}, {}, "retention_failure"),
        ({"retrieved": False}, {}, "retrieval_failure"),
        ({"observed_kind": MemoryKind.PROCEDURE}, {}, "influence_failure"),
        ({"observed_influence": MemoryInfluence.IGNORE}, {}, "influence_failure"),
        ({"prediction_reflected": False}, {}, "influence_failure"),
        ({}, {"action_reflected": False}, "influence_failure"),
        ({}, {"preserved_after_outcome": False}, "retention_failure"),
        ({"recognized": None}, {}, "unresolved_omission"),
        ({"recognized": False}, {}, "unresolved_omission"),
        ({"observed_kind": None}, {}, "unresolved_omission"),
        ({"observed_influence": None}, {}, "unresolved_omission"),
        ({}, {"action_reflected": None}, "unresolved_omission"),
        ({}, {"reported_unavailable": True}, "unresolved_omission"),
    ],
)
def test_matched_stage_controls(before, after, expected):
    assert audit(before=before, after=after).classification == expected


def test_later_unavailability_claim_does_not_erase_pre_action_recognition():
    result = audit(after={"action_reflected": False, "reported_unavailable": True}, active=())
    assert result.classification == "influence_failure"
    assert result.before.recognized is True
    assert result.before.prediction_reflected is True
    assert result.after.reported_unavailable is True
    assert result.continuity.unexplained_omission_ids == ("k",)


def test_omission_is_unresolved_when_all_observed_stages_pass():
    assert audit(active=()).classification == "unresolved_omission"


@pytest.mark.parametrize("kind", list(MemoryKind))
def test_content_kind_does_not_require_numeric_eligibility(kind):
    assert audit(kind=kind).classification == "consistent"


def edit_event(req, event_id, **changes):
    return replace(
        req,
        events=tuple(replace(e, **changes) if e.event_id == event_id else e for e in req.events),
    )


@pytest.mark.parametrize(
    "event_id,changes",
    [
        ("post", {"timestamp": "2026-09-30T12:00:04Z"}),
        ("proposed-2", {"timestamp": "2026-09-30T12:00:05Z"}),
        ("post", {"conversation_id": "other"}),
        ("post", {"action_id": "other"}),
        ("post", {"parent_event_ids": ("verification-0",)}),
        ("verification-0", {"timestamp": "2026-09-30T12:00:01Z"}),
        ("post", {"checkpoint_step": 3}),
    ],
)
def test_wrong_chronology_or_identity_is_rejected(event_id, changes):
    with pytest.raises(ValueError):
        audit_peo_continuity(edit_event(request(), event_id, **changes), enabled=True)


def test_later_assessment_cannot_supply_missing_prospective_observations():
    req = request()
    pred = next(e for e in req.events if e.event_id == "p")
    req = edit_event(
        req, "p", payload=dict(pred.payload) | {"predicted_outcome": {"temperature": [20]}}
    )
    result = audit_peo_continuity(req, enabled=True)
    assert result.classification == "unresolved_omission"
    assert result.before is None


def test_rebinding_evidence_use_or_commitment_is_rejected():
    req = request()
    for changed in (
        replace(req, evidence_use=replace(req.evidence_use, kind=MemoryKind.NORM)),
        replace(req, commitments=(replace(req.commitments[0], confidence=0.1),)),
    ):
        with pytest.raises(ValueError, match="digest"):
            audit_peo_continuity(changed, enabled=True)


def test_snapshot_output_is_detached():
    req = request()
    result = audit_peo_continuity(req, enabled=True)
    original = result.to_dict()
    changed = result.to_dict()
    changed["evidence_use"]["kind"] = "norm"
    with pytest.raises(TypeError):
        req.events[5].payload["predicted_outcome"]["continuity_evidence"]["k"]["recognized"] = False
    assert result.to_dict() == original


@pytest.mark.parametrize("value", [1, "true", [], {}])
def test_stage_flags_do_not_coerce_values(value):
    with pytest.raises(ValueError):
        request(before={"recognized": value})


def with_update(req, status, finding):
    ref = EvidenceReference("new-finding", EvidenceSourceKind.EXTERNAL_RECORD)
    verifier = typed_verification(1, ref, False, finding=finding)
    payload = verifier.to_dict()["payload"]
    payload["result"]["action_id"] = "action"
    verifier = replace(
        verifier,
        event_id="update-verifier",
        action_id="action",
        parent_event_ids=("o",),
        timestamp="2026-09-30T12:00:05Z",
        payload=payload,
    )
    update = CommitmentUpdate(
        "k", status, "New verified evidence", (ref,), ("update-verifier",), ("update-host",)
    )
    position = next(i for i, e in enumerate(req.events) if e.event_id == "post")
    return replace(
        req,
        events=req.events[:position] + (verifier,) + req.events[position:],
        updates=(update,),
        decision_context_changed=status is CommitmentStatus.SCOPED_OUT,
    )


@pytest.mark.parametrize(
    "status,finding,expected",
    [
        (
            CommitmentStatus.CONTRADICTED,
            ("contradiction_status", "contradicted"),
            "legitimate_update",
        ),
        (CommitmentStatus.WITHDRAWN, ("claimed_outcome_supported", False), "legitimate_update"),
        (CommitmentStatus.SCOPED_OUT, ("task_fit", False), "legitimate_scope_change"),
    ],
)
def test_verified_update_controls_preserve_prior_use_failures(status, finding, expected):
    good = with_update(request(active=()), status, finding)
    assert audit_peo_continuity(good, enabled=True).classification == expected
    failed = with_update(request(active=(), after={"action_reflected": False}), status, finding)
    result = audit_peo_continuity(failed, enabled=True)
    assert result.classification == "influence_failure"
    assert result.continuity.continuity_status == expected


def test_unrelated_or_future_findings_cannot_justify_omission():
    req = with_update(request(active=()), CommitmentStatus.SCOPED_OUT, ("task_fit", True))
    assert audit_peo_continuity(req, enabled=True).classification == "unresolved_omission"
    future = with_update(request(active=()), CommitmentStatus.SCOPED_OUT, ("task_fit", False))
    future = edit_event(future, "update-verifier", timestamp="2026-09-30T12:00:08Z")
    with pytest.raises(ValueError, match="chronology"):
        audit_peo_continuity(future, enabled=True)


def test_ignore_target_does_not_require_retrieval_or_behavioral_influence():
    values = dict(
        target=MemoryInfluence.IGNORE,
        before={
            "retrieved": False,
            "observed_influence": MemoryInfluence.IGNORE,
            "prediction_reflected": False,
        },
        after={"action_reflected": False},
    )
    assert audit(**values).classification == "consistent"
    values["after"] = {"action_reflected": True}
    assert audit(**values).classification == "influence_failure"


def test_missing_retrospective_telemetry_is_not_a_success():
    assert audit(assessment_event_id=None).classification == "unresolved_omission"


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "other"),
        ("repeat_id", 1),
    ],
)
def test_evidence_use_context_cannot_cross_units(field, value):
    req = request()
    measurement = req.evidence_use.measurement
    use = replace(
        req.evidence_use,
        measurement=replace(measurement, context=replace(measurement.context, **{field: value})),
    )
    with pytest.raises(ValueError):
        audit_peo_continuity(replace(req, evidence_use=use), enabled=True)


def test_later_policy_cannot_be_presented_as_prospective():
    req = request()
    use = replace(req.evidence_use, assessed_at="2026-09-30T12:00:01Z")
    with pytest.raises(ValueError, match="before prediction"):
        audit_peo_continuity(replace(req, evidence_use=use), enabled=True)


def test_verified_supersession_requires_a_bound_replacement():
    req = with_update(
        request(active=()), CommitmentStatus.WITHDRAWN, ("claimed_outcome_supported", True)
    )
    update = replace(
        req.updates[0], status=CommitmentStatus.SUPERSEDED, superseded_by="replacement"
    )
    original = req.commitments[0]
    replacement = replace(
        original,
        commitment_id="replacement",
        memory=replace(original.memory, memory_id="replacement"),
        evidence_refs=update.evidence_refs,
        source_event_refs=update.source_event_refs,
        first_active_at=1,
        last_active_at=1,
    )
    good = replace(
        req,
        updates=(update,),
        commitments=(original, replacement),
        active_commitment_ids=("replacement",),
    )
    assert audit_peo_continuity(good, enabled=True).classification == "legitimate_update"
    bad = replace(req, updates=(update,))
    assert audit_peo_continuity(bad, enabled=True).classification == "unresolved_omission"


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": True},
        {"provenance": []},
        {"provenance": "host"},
        {"commitment_digest": "forged"},
        {"observed_kind": 1},
        {"recognized": "yes"},
    ],
)
def test_malformed_captured_stage_is_rejected(changes):
    req = request()
    event = next(e for e in req.events if e.event_id == "p")
    payload = event.to_dict()["payload"]
    payload["predicted_outcome"]["continuity_evidence"]["k"].update(changes)
    with pytest.raises(ValueError):
        audit_peo_continuity(edit_event(req, "p", payload=payload), enabled=True)


def test_unverifiable_original_and_unbound_measurement_are_rejected():
    req = request()
    invalid = replace(req.commitments[0], source_event_refs=("missing",))
    with pytest.raises(ValueError, match="available before"):
        audit_peo_continuity(replace(req, commitments=(invalid,)), enabled=True)
    record = reconciliation()
    record = replace(record, bindings=(replace(record.bindings[0], verifier_event_id=None),))
    with pytest.raises(ValueError):
        audit_peo_continuity(edit_event(req, "r", payload=record.to_dict()), enabled=True)


def test_public_exports_and_strict_enable_flag():
    import semantic_intent_robustness as public

    assert public.audit_peo_continuity is audit_peo_continuity
    with pytest.raises(ValueError):
        audit_peo_continuity(enabled=1)
    with pytest.raises(ValueError):
        audit_peo_continuity(enabled=True)


@pytest.mark.parametrize(
    "timestamp,conversation",
    [
        ("2026-09-30T12:00:08Z", "c"),
        ("2026-09-30T12:00:05Z", "other"),
        ("2026-09-30T12:00:05Z", "c"),
    ],
)
def test_supersession_checks_replacement_source_chronology_and_conversation(
    timestamp, conversation
):
    req = with_update(
        request(active=()), CommitmentStatus.WITHDRAWN, ("claimed_outcome_supported", True)
    )
    update = replace(
        req.updates[0], status=CommitmentStatus.SUPERSEDED, superseded_by="replacement"
    )
    verifier = next(e for e in req.events if e.event_id == "update-verifier")
    source = replace(
        verifier, event_id="replacement-source", timestamp=timestamp, conversation_id=conversation
    )
    original = req.commitments[0]
    replacement = replace(
        original,
        commitment_id="replacement",
        memory=replace(original.memory, memory_id="replacement"),
        evidence_refs=update.evidence_refs,
        source_event_refs=("replacement-source",),
        first_active_at=1,
        last_active_at=1,
    )
    position = next(i for i, e in enumerate(req.events) if e.event_id == "post")
    req = replace(
        req,
        updates=(update,),
        commitments=(original, replacement),
        events=req.events[:position] + (source,) + req.events[position:],
    )
    if conversation == "other":
        # Existing continuity already rejects this source, so it cannot yield an accepted update.
        assert audit_peo_continuity(req, enabled=True).classification == "unresolved_omission"
    elif timestamp.endswith("08Z"):
        with pytest.raises(ValueError, match="chronology"):
            audit_peo_continuity(req, enabled=True)
    else:
        result = audit_peo_continuity(req, enabled=True)
        assert result.classification == "legitimate_update"
        assert ("replacement-source", timestamp) in result.chronology
