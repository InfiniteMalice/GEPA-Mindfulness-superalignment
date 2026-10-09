"""Simulated stakeholder callbacks cannot rewrite their binding or acquire authority."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from test_perspective_protocol import candidate, public_plan

# Local
from semantic_intent_robustness.perspective_generation import capture_perspectives
from semantic_intent_robustness.perspective_protocol import perspective_digest


def test_disabled_calls_nothing():
    """Opt-in is checked before invoking host code."""
    calls = []
    with pytest.raises(ValueError, match="enabled"):
        capture_perspectives(public_plan(), generate=lambda p: calls.append(p))
    assert calls == []


def test_callback_gets_public_detached_plan():
    """One callback sees a copy; even a malicious mutation cannot rebind the result."""
    plan = public_plan()
    digest = perspective_digest(plan)
    calls = []

    def generate(detached):
        calls.append(detached)
        assert detached == plan and detached is not plan
        object.__setattr__(detached.source, "source_text", "changed")
        return (candidate(),)

    cap = capture_perspectives(plan, generate=generate, enabled=True)
    assert len(calls) == 1
    assert cap.plan_digest == digest == perspective_digest(plan)
    assert cap.candidates[0].public_response == "Disclose risks"


def test_missing_and_empty_generation_differ():
    """No attempt is censored; an empty attempted return is observed."""
    plan = public_plan()
    assert capture_perspectives(plan, enabled=True).status == "censored"
    assert capture_perspectives(plan, candidates=(), enabled=True).status == "observed"
    with pytest.raises(ValueError):
        capture_perspectives(plan, candidates=(), generate=lambda p: (), enabled=True)


def test_omitted_stakeholder_slots_remain_planned():
    """Partial responses do not shrink the source roster."""
    plan = public_plan()
    cap = capture_perspectives(plan, candidates=(candidate(),), enabled=True)
    assert len(plan.slots) == 2 and len(cap.candidates) == 1
    assert all(c.status == "unverified" for c in plan.source.facts)
    assert "verified" not in cap.candidates[0].to_dict()


@pytest.mark.parametrize(
    "returned", [[], None, (candidate(), candidate()), (replace(candidate(), slot_id="foreign"),)]
)
def test_foreign_and_duplicate_candidates_rejected(returned):
    """Malformed output is a contract error, not silently counted as an observation."""
    with pytest.raises(ValueError):
        capture_perspectives(public_plan(), generate=lambda p: returned, enabled=True)


def test_error_type_retained_without_secret_text():
    """Host exception details cannot leak into public diagnostics."""

    def fail(plan):
        raise RuntimeError("SECRET TOKEN")

    cap = capture_perspectives(public_plan(), generate=fail, enabled=True)
    assert cap.status == "callback_error" and cap.candidates == ()
    assert "RuntimeError" in cap.reason and "SECRET" not in str(cap.to_dict())
