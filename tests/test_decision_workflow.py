"""Bounded decisions dispatch only through the existing one-use authority owner."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# isort: split
# Local
from test_decision_routing import backend, context, readout
from test_epistemic_routing import ENABLED, request
from test_runtime_authority import _clock, _grant, _issue

from gepa_mindfulness.factuality_observability.decision_workflow import (
    DecisionWorkflowConfig,
    run_decision_workflow,
)
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from gepa_mindfulness.verification import RuntimeCapability
from mindful_trace_gepa import ActionRecord


def setup_workflow():
    req = request()
    calls, decisions = [], []
    clock = _clock()

    def authorize(packet):
        action = ActionRecord(
            "routed-action", packet["route"], True, packet["action_scope"], "next-prediction"
        )
        decision, registry = _issue(
            RuntimeCapability.READ,
            (_grant(action=action, capabilities=(RuntimeCapability.READ,)),),
            action=action,
            clock=clock,
        )
        decisions.append((action, decision, registry))
        return action, decision, registry

    args = dict(
        public=context(req),
        backend=backend(),
        confidence_threshold=0.8,
        policy=ENABLED,
        config=DecisionWorkflowConfig(enabled=True, workflow_id="route:v1"),
        authenticate=lambda packet: True,
        authorize=authorize,
        clock=clock,
        handlers={
            action: lambda record, a=action: calls.append(a) or "host-result" for action in Action
        },
    )
    return req, args, calls, decisions


def test_disabled_does_not_read_inputs_or_invoke_roles():
    assert run_decision_workflow(None) is None


def test_confident_decision_dispatches_exactly_one_authorized_handler():
    req, args, calls, _ = setup_workflow()
    result = run_decision_workflow(req, **args)
    assert result.invoked and result.result == "host-result"
    assert calls == [Action.ACCEPT]
    report = result.to_dict()
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["authority_granted"] is False
    assert report["route"] == Action.ACCEPT.value
    assert len(report["decisions"]) == 1


def test_uncertain_decision_gets_one_generative_review_with_public_feedback():
    req, args, calls, _ = setup_workflow()
    args["backend"] = backend(lambda packet: readout(packet, probability=0.6))
    feedback = []

    def review(packet):
        feedback.append(packet.context.artifact)
        return readout(packet, action=Action.CLARIFY)

    args["review_backend"] = replace(backend(review, kind="generative"), name="reviewer")
    args["review_confidence_threshold"] = 0.8
    result = run_decision_workflow(req, **args)
    assert calls == [Action.CLARIFY]
    assert len(result.to_dict()["decisions"]) == 2
    assert len(feedback) == 1 and "confidence_escalation" in feedback[0]


def test_reviewer_cannot_override_required_external_verification():
    req, args, calls, _ = setup_workflow()
    req = replace(req, context=replace(req.context, verification_required=True))
    args["public"] = context(req)
    args["backend"] = backend(lambda packet: readout(packet, probability=0.6))
    args["review_backend"] = replace(backend(kind="generative"), name="reviewer")
    args["review_confidence_threshold"] = 0.8
    result = run_decision_workflow(req, **args)
    assert result.to_dict()["route"] == Action.ROUTE_EXTERNAL.value
    assert calls == [Action.ROUTE_EXTERNAL]


def test_uncertainty_without_reviewer_reaches_human_handler():
    req, args, calls, _ = setup_workflow()
    args["backend"] = backend(lambda packet: readout(packet, probability=0.6))
    run_decision_workflow(req, **args)
    assert calls == [Action.ESCALATE]


def test_source_authentication_precedes_any_model_callback():
    req, args, calls, _ = setup_workflow()
    args["authenticate"] = lambda packet: False
    args["backend"] = backend(lambda _: pytest.fail("unauthenticated model call"))
    with pytest.raises(ValueError):
        run_decision_workflow(req, **args)
    assert not calls


def test_missing_selected_handler_never_requests_a_grant():
    req, args, calls, _ = setup_workflow()
    args["handlers"] = {Action.CLARIFY: lambda _: None}
    args["authorize"] = lambda _: pytest.fail("missing handler reached authorization")
    result = run_decision_workflow(req, **args)
    assert not result.invoked
    assert result.to_dict()["status"] == "handler_unavailable"
    assert not calls


def test_misbound_grant_and_replay_cannot_invoke_any_handler(monkeypatch):
    timings = iter((0.0, 1.0, 2.0, 4.0, 5.0, 8.0))
    monkeypatch.setattr(
        "gepa_mindfulness.factuality_observability.epistemic_routing.perf_counter",
        lambda: next(timings),
    )
    req, args, calls, captured = setup_workflow()
    first = run_decision_workflow(req, **args)
    assert first.invoked
    args["authorize"] = lambda _: captured[0]
    with pytest.raises(ValueError, match="binding"):
        run_decision_workflow(req, **args)
    assert calls == [Action.ACCEPT]
    args["backend"] = backend(lambda packet: readout(packet, action=Action.CLARIFY))
    with pytest.raises(ValueError, match="binding"):
        run_decision_workflow(req, **args)
    assert calls == [Action.ACCEPT]


def test_mutating_host_authorization_packet_fails_before_consumption():
    req, args, calls, _ = setup_workflow()
    authorize = args["authorize"]

    def mutate(packet):
        issued = authorize(packet)
        packet["route"] = Action.CLARIFY.value
        return issued

    args["authorize"] = mutate
    with pytest.raises(ValueError):
        run_decision_workflow(req, **args)
    assert not calls


def test_config_closed_roundtrip():
    config = DecisionWorkflowConfig()
    assert type(config).from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        type(config).from_dict(config.to_dict() | {"unsafe": True})


def test_already_consumed_fresh_scope_grant_is_rejected():
    from gepa_mindfulness.verification import consume_authorization

    req, args, calls, _ = setup_workflow()
    authorize = args["authorize"]

    def consumed(packet):
        action, decision, registry = authorize(packet)
        consume_authorization(decision, action, grant_registry=registry, clock=args["clock"])
        return action, decision, registry

    args["authorize"] = consumed
    with pytest.raises(PermissionError, match="consumed"):
        run_decision_workflow(req, **args)
    assert not calls


def test_failed_generative_review_escalates_without_repeating():
    req, args, calls, _ = setup_workflow()
    args["backend"] = backend(lambda packet: readout(packet, probability=0.6))
    reviews = []

    def fail(packet):
        reviews.append(packet)
        raise RuntimeError("PRIVATE_UNTRUSTED_TEXT")

    args["review_backend"] = replace(backend(fail, kind="generative"), name="reviewer")
    args["review_confidence_threshold"] = 0.8
    result = run_decision_workflow(req, **args)
    assert calls == [Action.ESCALATE]
    assert len(reviews) == 1
    assert "PRIVATE_UNTRUSTED_TEXT" not in str(result.to_dict())


def test_large_feedback_escalates_without_calling_reviewer():
    req, args, calls, _ = setup_workflow()
    args["public"] = replace(context(req), artifact="x" * 60000)
    args["backend"] = backend(lambda packet: readout(packet, probability=0.6))
    args["review_backend"] = replace(
        backend(lambda _: pytest.fail("oversized reviewer input"), kind="generative"),
        name="reviewer",
    )
    args["review_confidence_threshold"] = 0.8
    result = run_decision_workflow(req, **args)
    assert calls == [Action.ESCALATE]
    assert result.to_dict()["review_skipped"] == "feedback_capacity"
