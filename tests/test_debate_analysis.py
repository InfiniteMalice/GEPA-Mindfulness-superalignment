"""Observed graph transitions are distinct from independently established semantic events."""

from dataclasses import replace

import pytest
from test_debate_records import challenge, protocol, result, snapshot
from test_sensitive_debate import run_fixture

from evaluation.causal_records import MetricVerdict
from evaluation.debate_analysis import analyze_debate
from evaluation.debate_records import (
    DebateAssessment,
    DebateOpportunity,
    debate_opportunities_digest,
    debate_session_digest,
)
from evaluation.ladder import Severity


def assessment(p, session, verdicts):
    """Bind toy oracle labels to this exact transcript and declared metric roster."""
    return DebateAssessment(
        debate_session_digest(session),
        debate_opportunities_digest(p.opportunities),
        p.evaluator,
        "verified",
        False,
        verdicts,
        snapshot().evidence_refs,
        "fixture oracle",
    )


def report(p, session, **kwargs):
    """Analyze captured fixture data with a separately supplied check authority."""
    return analyze_debate(p, session, enabled=True, authenticate_check=lambda e: True, **kwargs)


def test_wrong_challenger_counts_as_false_challenge():
    p = protocol()
    r = report(p, run_fixture(p))
    assert r["false_challenges"]["numerator"] == 1
    assert r["false_challenges"]["denominator"] == 1
    assert r["evidence_coverage"]["rate"] == 0.5
    assert r["evidence_coverage"]["censored"] == ["check-1"]
    assert len(r["cases"]) == 17
    assert r["confers_authority"] is False
    assert r["mechanism_recovery_established"] is False


def test_rhetorical_change_is_not_verified_transition():
    p = protocol(1)
    session = run_fixture(
        p, revise=lambda ctx, s, ch, checks: replace(s, proposed_action="answer politely")
    )
    r = report(p, session)
    assert r["first_observed_action_change_round"] == 0
    assert r["first_verified_decision_change_round"] is None
    assert r["metrics"]["transition_detection"]["unresolved"] == ["transition-0"]


def test_late_correct_evidence_and_justified_recovery():
    p = protocol(2)
    extras = tuple(
        DebateOpportunity(name, 1, name, Severity.CONSEQUENTIAL, "correction")
        for name in ("correct_recovery", "justified_position_change", "premise_fault_localization")
    )
    p = replace(p, opportunities=p.opportunities + extras)
    session = run_fixture(
        p,
        stop=None,
        defend=lambda ctx: snapshot(action="wait"),
        verify=lambda ctx, s, ch: (
            result(s, ch, "supported" if ctx.round_index == 0 else "contradicted"),
        ),
        revise=lambda ctx, s, ch, checks: snapshot(
            action="answer" if ctx.round_index else "wait",
            premise="Corrected record" if ctx.round_index else "The record is valid",
        ),
    )
    verdicts = tuple(
        MetricVerdict(
            o.opportunity_id,
            o.round_index == 1,
            True if o.round_index == 1 else None,
            "oracle labels",
            snapshot().evidence_refs,
        )
        for o in p.opportunities
    )
    a = assessment(p, session, verdicts)
    r = report(p, session, assessment=a, authenticate_assessment=lambda candidate: candidate == a)
    assert r["first_verified_decision_change_round"] == 1
    assert r["metrics"]["correct_recovery"]["rate"] == 1
    assert r["transitions"][-1]["modified_claim_ids"] == ["p"]
    assert len(r["severe_events"]) == 3


def test_alternative_valid_conclusions_not_failures():
    p = protocol(1)
    session = run_fixture(
        p, revise=lambda ctx, s, ch, checks: replace(s, proposed_action="another valid action")
    )
    a = assessment(
        p,
        session,
        (
            MetricVerdict(
                "transition-0", False, None, "semantically equivalent", snapshot().evidence_refs
            ),
        ),
    )
    r = report(p, session, assessment=a, authenticate_assessment=lambda _: True)
    assert r["first_verified_decision_change_round"] is None
    assert r["metrics"]["transition_detection"]["ineligible"] == ["transition-0"]


def test_reused_claim_id_invalidates_prior_checks():
    p = protocol(2)
    session = run_fixture(
        p,
        stop=None,
        defend=lambda ctx: snapshot(premise="original" if ctx.round_index == 0 else "changed"),
        verify=lambda ctx, s, ch: (result(s, ch),) if ctx.round_index == 0 else (),
    )
    r = report(p, session)
    assert r["check_rows"][1]["status"] == "missing"
    assert r["transitions"][1]["modified_claim_ids"] == ["p"]
    assert len(r["verified_claims_by_snapshot"]) == 1


def test_dropped_and_censored_slots_preserve_denominators():
    p = protocol(2)
    session = run_fixture(
        p,
        challenge=lambda ctx, s: replace(challenge(s), requests=(), target_claim_ids=()),
        verify=lambda *args: (),
    )
    r = report(p, session)
    assert r["evidence_coverage"]["denominator"] == 2
    assert r["evidence_coverage"]["missing"] == ["check-0"]
    assert r["evidence_coverage"]["censored"] == ["check-1"]
    assert r["false_challenges"]["rate"] is None


@pytest.mark.parametrize("kind", ["absent", "untrusted", "human", "disputed", "mutated"])
def test_empty_and_unauthenticated_assessments_stay_unresolved(kind):
    p = protocol(1)
    session = run_fixture(p)
    a = assessment(
        p, session, (MetricVerdict("transition-0", True, True, "claim", snapshot().evidence_refs),)
    )
    if kind == "human":
        a = replace(a, human_required=True)
    if kind == "disputed":
        a = replace(a, status="disputed")

    def auth(candidate):
        if kind == "mutated":
            object.__setattr__(candidate, "reason", "changed")
        return kind != "untrusted"

    r = report(p, session, assessment=None if kind == "absent" else a, authenticate_assessment=auth)
    assert r["metrics"]["transition_detection"]["unresolved"] == ["transition-0"]


def test_duplicate_opportunity_cannot_relabel_severity():
    p = protocol(1)
    with pytest.raises(ValueError):
        replace(
            p,
            opportunities=p.opportunities
            + (
                replace(
                    p.opportunities[0], opportunity_id="again", severity=Severity.CONSEQUENTIAL
                ),
            ),
        )


def test_stale_session_and_foreign_labels_rejected():
    p = protocol(1)
    session = run_fixture(p)
    a = assessment(
        p, session, (MetricVerdict("foreign", True, True, "claim", snapshot().evidence_refs),)
    )
    with pytest.raises(ValueError):
        report(p, session, assessment=a)
    with pytest.raises(ValueError):
        report(replace(p, rubric_id="changed"), session)


def test_capture_and_authentication_remain_distinct():
    p = protocol(1)
    session = run_fixture(p)
    r = analyze_debate(p, session, enabled=True)
    assert r["unresolved_disputes"]["rate"] == 1
    assert r["evidence_coverage"]["rate"] == 0
    with pytest.raises(ValueError, match="enabled"):
        analyze_debate(p, session)
