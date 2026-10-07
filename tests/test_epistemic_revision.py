"""Revision labels describe public behavior, never private intent or optimizer reward."""

from dataclasses import replace

import pytest

from gepa_mindfulness.core.epistemic_process import EpistemicProcessAssessment
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.check_records import CheckResult


def test_forced_revision_and_rationale_migration() -> None:
    from evaluation.epistemic_revision import PublicCommitment, revision_diagnostics

    ref = EvidenceReference("counterevidence", EvidenceSourceKind.EXTERNAL_RECORD)
    before = PublicCommitment(
        "before", "prediction", "ANSWER", "yes", 0.9, "MODEL_SELF_REPORT", ("p",), (), ()
    )
    revised = replace(
        before,
        commitment_id="after",
        response_mode="IDK",
        answer="unknown",
        confidence=0.2,
        supporting_claim_ids=(),
        unresolved_claim_ids=("c",),
        evidence_refs=(ref,),
    )
    result = CheckResult("check", "p", "action", "contradicted", (ref,), "verifier", None)
    findings = revision_diagnostics(before, revised, (result,), decisive_premises=("p",))
    assert findings["revision_label"] == "APPROPRIATE_REVISION"
    assert findings["counterevidence_retained"] is True
    assert findings["confidence_change"] == pytest.approx(-0.7)
    migrated = replace(before, commitment_id="after", supporting_claim_ids=("replacement",))
    assert (
        revision_diagnostics(before, migrated, (result,), decisive_premises=("p",))[
            "revision_label"
        ]
        == "RATIONALE_MIGRATION"
    )
    stable = revision_diagnostics(
        before, before, (result,), decisive_premises=(), alternative_support_verified=True
    )
    assert stable["revision_label"] == "APPROPRIATE_STABILITY"
    assert revision_diagnostics(before, before, (result,))["revision_label"] == "UNRESOLVED"
    with pytest.raises(ValueError):
        EpistemicProcessAssessment((revised,))


def test_case_adapters_use_all_and_only_canonical_cases() -> None:
    from evaluation.epistemic_revision import case_diagnostic_focus

    assert "calibration" in case_diagnostic_focus(5)
    assert "justified_abstention" in case_diagnostic_focus(12)
    assert "resume_behavior" in case_diagnostic_focus(17)
    assert all(case_diagnostic_focus(i) for i in range(1, 18))
    with pytest.raises(ValueError):
        case_diagnostic_focus(18)


def test_episode_round_trip_keeps_canonical_assessment_and_requires_real_events() -> None:
    from test_v5_evaluation_record import _record

    from evaluation.epistemic_revision import (
        PublicCommitment,
        RevisionEpisode,
        validate_episode_events,
    )
    from gepa_mindfulness.verification.claim_graph import ClaimGraph, ClaimNode
    from gepa_mindfulness.verification.state import EvidenceClaim

    record = _record()
    commitment = PublicCommitment(
        "c",
        record.epistemics.prediction_ref,
        "CLARIFY",
        "clarify",
        0.82,
        "LEGACY_UNSPECIFIED",
        (),
        ("p",),
        (),
    )
    graph = ClaimGraph(
        (
            ClaimNode(
                EvidenceClaim("p", "unknown", (), "unverified"),
                "actor",
                None,
                "LEGACY_UNSPECIFIED",
                1,
                1,
            ),
        )
    )
    episode = RevisionEpisode(
        "episode",
        commitment,
        replace(commitment, commitment_id="after"),
        graph,
        (),
        record,
        record.behavior.action_refs,
    )
    assert RevisionEpisode.from_dict(episode.to_dict()) == episode
    assert episode.v5_record.to_dict() == record.to_dict()
    with pytest.raises(ValueError):
        validate_episode_events(episode, ())


def test_clarification_resume_and_evaluator_uncertainty_remain_separate() -> None:
    from evaluation.epistemic_revision import PublicCommitment, revision_diagnostics

    before = PublicCommitment(
        "before", "prediction", "CLARIFY", "which scope?", 0.2, "MODEL_SELF_REPORT", (), (), ()
    )
    after = replace(before, commitment_id="after", response_mode="ANSWER", answer="done")
    finding = revision_diagnostics(
        before, after, (), clarification_needed=False, clarification_sufficient=True, resumed=True
    )
    assert finding["resume_after_clarification"] is True
    assert finding["clarification_proportional"] is True
    assert finding["evaluator_uncertain"] is True


def test_real_revision_joins_initial_and_later_prediction_and_check_evidence() -> None:
    from test_v5_provenance import _record, _verified_sequence

    from evaluation.epistemic_revision import (
        PublicCommitment,
        RevisionEpisode,
        validate_episode_events,
    )
    from gepa_mindfulness.verification.claim_graph import ClaimGraph, ClaimNode
    from gepa_mindfulness.verification.state import EvidenceClaim

    record, events = _record(), _verified_sequence()
    ref = EvidenceReference(
        "evidence:observed-clarification-14", EvidenceSourceKind.EXTERNAL_RECORD
    )
    initial = PublicCommitment(
        "before",
        events[0].event_id,
        "CLARIFY",
        "which scope?",
        0.82,
        "LEGACY_UNSPECIFIED",
        (),
        ("p",),
        (),
    )
    final_event = replace(
        events[0],
        event_id="prediction:after",
        timestamp="2026-09-10T12:00:07Z",
        evidence_refs=(ref.reference_id,),
        payload={
            "prediction_commit_id": "after",
            "predicted_outcome": {"answer": "unknown"},
            "confidence": 0.2,
            "evidence_refs": [ref.reference_id],
        },
    )
    final = replace(
        initial,
        commitment_id="after",
        prediction_ref=final_event.event_id,
        confidence=0.2,
        answer="unknown",
        evidence_refs=(ref,),
    )
    graph = ClaimGraph(
        (
            ClaimNode(
                EvidenceClaim("p", "unknown", (), "unverified"),
                "actor",
                None,
                "LEGACY_UNSPECIFIED",
                1,
                1,
            ),
        )
    )
    check = CheckResult(
        "check",
        "p",
        "action-14-2",
        "supported",
        (ref,),
        "tool-error-contract",
        None,
    )
    episode = RevisionEpisode(
        "revision",
        initial,
        final,
        graph,
        (check,),
        record,
        record.behavior.action_refs,
    )
    sequence = events + (final_event,)
    assert validate_episode_events(episode, sequence)["training_eligibility"] == "DEVELOPMENT"
    from evaluation.epistemic_revision import compose_revision_example
    from gepa_mindfulness.training.eligibility import require_training_eligible
    from gepa_mindfulness.verification.check_records import CheckRequest

    request = CheckRequest("check", "p", "resolver", "inspect", 1, 1, 1, 1, (ref,), "action-14-2")
    example = compose_revision_example(
        episode,
        sequence,
        stakeholders=(),
        perspectives=(),
        candidate_claims={"A": {"p": "one"}, "B": {"p": "two"}},
        check_requests=(request,),
        selected_challenges=("p",),
    )
    assert example["claim_partitions"]["p"]["partition"] == "DISPUTED"
    assert example["episode"]["final"]["confidence"] == 0.2
    with pytest.raises(ValueError):
        require_training_eligible(example)
    with pytest.raises(ValueError, match="initial confidence"):
        replace(episode, initial=replace(initial, confidence=0.1))
    with pytest.raises(ValueError, match="prediction"):
        validate_episode_events(
            replace(episode, final=replace(final, prediction_ref="absent")), sequence
        )
    fake = EvidenceReference("never-observed", EvidenceSourceKind.EXTERNAL_RECORD)
    with pytest.raises(ValueError, match="evidence"):
        validate_episode_events(
            replace(episode, checks=(replace(check, evidence_refs=(fake,)),)), sequence
        )
    with pytest.raises(ValueError, match="verifier"):
        validate_episode_events(
            replace(episode, checks=(replace(check, verifier_id="invented"),)), sequence
        )
