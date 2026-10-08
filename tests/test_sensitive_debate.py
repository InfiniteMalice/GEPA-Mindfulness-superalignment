"""Bounded orchestration and authentication failure paths over real public records."""

from dataclasses import replace

import pytest
from test_debate_records import challenge, protocol, result, snapshot

from evaluation.sensitive_debate import run_sensitive_debate


def run_fixture(p=None, **overrides):
    """Run deterministic producers with independently configurable host acceptance."""
    options = dict(
        defend=lambda ctx: snapshot(),
        challenge=lambda ctx, before: challenge(before, ctx.round_index),
        verify=lambda ctx, before, ch: (result(before, ch),),
        revise=lambda ctx, before, ch, checks: before,
        authenticate=lambda envelope: True,
        stop=lambda session: True,
        enabled=True,
    )
    options.update(overrides)
    return run_sensitive_debate(p or protocol(), **options)


def test_disabled_calls_no_callbacks():
    with pytest.raises(ValueError, match="enabled"):
        run_fixture(enabled=False, defend=lambda ctx: pytest.fail("called while disabled"))


def test_ordered_public_rounds_and_host_stop():
    events = []

    def defend(ctx):
        events.append("defend")
        assert "expected_actions" not in str(ctx.to_dict())
        return snapshot()

    def check(ctx, before):
        events.append("challenge")
        return challenge(before, ctx.round_index)

    def verify(ctx, before, ch):
        events.append("verify")
        return (result(before, ch),)

    def authenticate(envelope):
        events.append("authenticate")
        return True

    def revise(ctx, before, ch, checks):
        events.append("revise")
        assert checks[0].verdict == "supported"
        return before

    session = run_fixture(
        defend=defend, challenge=check, verify=verify, authenticate=authenticate, revise=revise
    )
    assert events == ["defend", "challenge", "verify", "authenticate", "revise"]
    assert session.stop_reason == "host_stopped"
    assert len(session.rounds) == 1


def test_eight_round_cap_is_censored():
    session = run_fixture(protocol(8), stop=None)
    assert len(session.rounds) == 8
    assert session.stop_reason == "budget_exhausted"


@pytest.mark.parametrize("kind", ["actor", "digest", "request", "verifier", "revision"])
def test_role_spoofing_and_stale_bindings_rejected(kind):
    overrides = {}
    if kind == "actor":
        overrides["defend"] = lambda ctx: snapshot(actor="verifier")
    elif kind == "digest":
        overrides["challenge"] = lambda ctx, s: replace(challenge(s), snapshot_digest="0" * 64)
    elif kind == "request":
        overrides["challenge"] = lambda ctx, s: challenge(s, 1)
    else:

        def verify(ctx, s, ch):
            r = result(s, ch)
            changes = (
                {"verifier_id": "defender"}
                if kind == "verifier"
                else {"revision_claim_id": "absent"}
            )
            return (replace(r, result=replace(r.result, **changes)),)

        overrides["verify"] = verify
    with pytest.raises(ValueError):
        run_fixture(**overrides)


@pytest.mark.parametrize("acceptance", [None, False, 1, "yes", RuntimeError("secret")])
def test_rejected_receipts_are_unresolved_in_revision(acceptance):
    def authenticate(envelope):
        if isinstance(acceptance, Exception):
            raise acceptance
        return acceptance

    def revise(ctx, before, ch, checks):
        assert checks[0].verdict == "unresolved"
        return before

    session = run_fixture(authenticate=authenticate, revise=revise)
    assert session.rounds[0].results[0].result.verdict == "supported"


def test_callback_mutation_cannot_reuse_acceptance():
    def authenticate(envelope):
        object.__setattr__(envelope.check.result, "verdict", "contradicted")
        return True

    def revise(ctx, before, ch, checks):
        assert checks[0].verdict == "unresolved"
        return before

    session = run_fixture(authenticate=authenticate, revise=revise)
    assert session.rounds[0].results[0].result.verdict == "supported"


def test_pending_human_stops_before_revision():
    session = run_fixture(
        verify=lambda ctx, s, ch: (result(s, ch, human=True),),
        revise=lambda *args: pytest.fail("revision before human review"),
    )
    assert session.stop_reason == "human_pending"
    assert session.rounds[0].after is None


def test_exception_preserves_partial_transcript():
    def verify(*args):
        raise RuntimeError("private service token")

    session = run_fixture(verify=verify)
    assert session.rounds[0].challenge is not None
    assert session.stop_reason == "callback_error"
    assert "private service token" not in str(session.to_dict())


def test_empty_challenge_keeps_missing_slots():
    session = run_fixture(
        challenge=lambda ctx, s: replace(challenge(s), requests=(), target_claim_ids=()),
        verify=lambda *args: (),
    )
    assert session.rounds[0].results == ()
    assert session.stop_reason == "host_stopped"


def test_undecomposable_escalation():
    s = replace(
        snapshot(), graph=None, conclusion_claim_id=None, decomposition_status="undecomposable"
    )
    session = run_fixture(defend=lambda ctx: s)
    assert session.stop_reason == "undecomposable"
    assert session.rounds[0].challenge is None


def test_missing_and_stop_exception_keep_phase_evidence():
    assert run_fixture(challenge=lambda *args: None).stop_reason == "missing"

    def stop(session):
        raise RuntimeError("private")

    session = run_fixture(stop=stop)
    assert session.stop_reason == "callback_error"
    assert session.rounds[0].after is not None
