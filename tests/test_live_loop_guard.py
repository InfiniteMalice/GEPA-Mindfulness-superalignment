"""Actual suppression/delegation with the existing one-use runtime authority owner."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_deloop import fixture
from test_runtime_authority import _clock, _grant, _issue

from evaluation.causal_records import canonical_json
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.actionopd import DecisionKind, SemanticAction
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.deloop import LoopStep
from gepa_mindfulness.training.live_loop_guard import (
    LoopGuardConfig,
    execute_guarded_action,
    inspect_repeat_guard,
    loop_guard_action_scope,
)
from gepa_mindfulness.verification import (
    AuthorizationDecision,
    RuntimeCapability,
    consume_authorization,
)
from gepa_mindfulness.verification.interfaces import (
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)
from mindful_trace_gepa import ActionRecord, PredictionCommit


def setup_guard(
    *, context="ordinary", state="state-v1", observations=None, actions=None, history_contexts=None
):
    events, steps = fixture(actions=actions, observations=observations, contexts=history_contexts)
    events, steps = events[:10], steps[:2]
    current_ref = EvidenceReference("current-state", EvidenceSourceKind.EXTERNAL_RECORD)
    action = steps[1].action
    trajectory = replace(
        steps[1].request.trajectory,
        trajectory_id="pending-action",
        prompt="current public state",
        prompt_token_ids=(1, 9),
        response=canonical_json(action.to_dict()),
        evidence_references=(current_ref,),
    )
    pending = LoopStep(
        RewardRequest(trajectory, (current_ref,)), action, state, context, (current_ref,)
    )
    prediction = PredictionCommit(
        "prediction:pending", {"deloop": pending.commitment()}, 0.5, (current_ref.reference_id,)
    )
    config = LoopGuardConfig(
        enabled=True,
        policy_id="live-loop:v1",
        task_id="task",
        run_id="run",
        harness_version="host-v1",
        reviewer_id="independent",
        reviewer_version="v1",
        reviewer_contract_id="live-loop-review:v1",
    )
    record = ActionRecord(
        "pending-action",
        action.kind.value,
        True,
        loop_guard_action_scope(pending, prediction, config),
        prediction.prediction_commit_id,
    )
    clock = _clock("2026-10-27T12:00:00+00:00")
    decision, registry = _issue(
        RuntimeCapability.READ,
        (_grant(action=record, capabilities=(RuntimeCapability.READ,)),),
        action=record,
        clock=clock,
    )
    calls = []
    arguments = dict(
        config=config,
        events=events,
        steps=steps,
        pending=pending,
        prediction=prediction,
        authenticate_context=lambda _packet: True,
        verify_repeat=confirmed_repeat,
        grant_registry=registry,
        clock=clock,
        execute=lambda action: calls.append(action) or "output",
    )
    return record, decision, arguments, calls


def confirmed_repeat(packet):
    """Independently authored fixture findings bind both actual repeated observations."""
    refs = tuple(EvidenceReference.from_dict(item) for item in packet["source_references"])
    return RelationalVerificationResult(
        packet["repeat_action_id"],
        False,
        True,
        "none",
        True,
        True,
        False,
        True,
        refs,
        tuple(
            VerificationEvidenceBinding(name, refs)
            for name in (
                "dependencies_satisfied",
                "contradiction_status",
                "provenance_intact",
                "authorization_scope_valid",
                "repeated_failed_route",
            )
        ),
    )


def test_default_off_does_not_inspect_history():
    assert inspect_repeat_guard(object(), object(), object(), object(), object()) is None


def test_confirmed_repeat_is_suppressed_without_consuming_authority():
    action, decision, args, calls = setup_guard()
    result = execute_guarded_action(action, decision, **args)
    assert result.invoked is False
    assert result.gate["route"] == "blocked_repeat"
    assert result.gate["authority_granted"] is False
    assert result.gate["training_eligibility"] == "DEVELOPMENT"
    assert "task_completed" not in result.gate
    assert calls == []
    # No replacement action is performed; the original decision remains unused.
    assert (
        consume_authorization(
            decision, action, grant_registry=args["grant_registry"], clock=args["clock"]
        ).authorized
        is True
    )


@pytest.mark.parametrize(
    "context", ("human_recheck", "independent_verification", "productive_repeat")
)
def test_authenticated_necessary_rechecks_reach_the_existing_executor_once(context):
    action, decision, args, calls = setup_guard(context=context)
    args["verify_repeat"] = lambda _packet: pytest.fail("exempt repeat was reviewed as a loop")
    result = execute_guarded_action(action, decision, **args)
    assert result.invoked is True
    assert result.result == "output"
    assert calls == [action]
    with pytest.raises(PermissionError, match="already been consumed"):
        execute_guarded_action(action, decision, **args)
    assert calls == [action]


@pytest.mark.parametrize("change", ("state", "observation"))
def test_changed_state_or_informative_observation_does_not_block(change):
    action, decision, args, calls = setup_guard(
        state="state-v2" if change == "state" else "state-v1",
        observations=("same", "new", "new") if change == "observation" else None,
    )
    args["verify_repeat"] = lambda _packet: pytest.fail("useful retry was reviewed as a loop")
    assert execute_guarded_action(action, decision, **args).invoked is True
    assert calls == [action]


def test_answered_clarification_is_an_existing_diagnostic_subtype():
    question = SemanticAction(DecisionKind.CLARIFICATION, '{"question":"which file?"}')
    action, decision, args, calls = setup_guard(
        context="answered_clarification", actions=(question, question, question)
    )
    result = execute_guarded_action(action, decision, **args)
    assert result.gate["diagnostic_subtype"] == "repeated_clarification_after_answer"
    assert result.invoked is False
    assert calls == []


@pytest.mark.parametrize("failure", ("deny", "truthy", "mutate", "exception"))
def test_context_review_uncertainty_cannot_execute_even_a_claimed_exemption(failure):
    action, decision, args, calls = setup_guard(context="human_recheck")

    def authenticate(packet):
        if failure == "mutate":
            packet["authority_granted"] = True
            return True
        if failure == "exception":
            raise RuntimeError("private callback detail")
        return 1 if failure == "truthy" else False

    args["authenticate_context"] = authenticate
    result = execute_guarded_action(action, decision, **args)
    assert result.invoked is False
    assert result.gate["route"] == "escalate"
    assert "private callback detail" not in canonical_json(result.gate)
    assert calls == []


def test_a_continue_gate_never_makes_a_reconstructed_authorization_live():
    action, decision, args, calls = setup_guard(state="state-v2")
    reconstructed = AuthorizationDecision.from_dict(decision.to_dict())
    with pytest.raises(PermissionError, match="was not issued"):
        execute_guarded_action(action, reconstructed, **args)
    assert calls == []


def test_guard_disabled_still_requires_existing_authority():
    action, decision, args, calls = setup_guard()
    args.update(
        config=LoopGuardConfig(),
        events=object(),
        steps=object(),
        pending=object(),
        prediction=object(),
    )
    assert execute_guarded_action(action, decision, **args).invoked is True
    assert calls == [action]


def test_a_previous_authenticated_recheck_is_not_an_unproductive_loop():
    action, decision, args, calls = setup_guard(
        history_contexts=("ordinary", "human_recheck", "ordinary")
    )
    args["verify_repeat"] = lambda _packet: pytest.fail(
        "necessary source recheck reviewed as failure"
    )
    assert execute_guarded_action(action, decision, **args).invoked is True
    assert calls == [action]


def test_first_action_with_empty_history_still_uses_runtime_authority():
    action, decision, args, calls = setup_guard()
    args.update(events=(), steps=())
    args["verify_repeat"] = lambda _packet: pytest.fail("empty history reviewed as loop")
    assert execute_guarded_action(action, decision, **args).invoked is True
    assert calls == [action]


def test_intervening_new_action_invalidates_a_stale_loop_prediction():
    action, decision, args, calls = setup_guard()
    args["events"], args["steps"] = fixture()
    args["verify_repeat"] = lambda _packet: pytest.fail("new evidence ignored")
    assert execute_guarded_action(action, decision, **args).invoked is True
    assert calls == [action]


@pytest.mark.parametrize("mode", ("unknown", "mutate", "throw", "foreign", "missing_binding"))
def test_unresolved_or_unbound_repeat_reviews_escalate_without_execution(mode):
    action, decision, args, calls = setup_guard()

    def verify(packet):
        checked = confirmed_repeat(packet)
        if mode == "unknown":
            return replace(checked, contradiction_status="unknown")
        if mode == "mutate":
            packet["authority_granted"] = True
            return checked
        if mode == "throw":
            raise RuntimeError("private reviewer detail")
        if mode == "foreign":
            return replace(checked, action_id="foreign-action")
        kept = checked.evidence_refs[-1:]
        bindings = tuple(
            (
                VerificationEvidenceBinding(binding.field_name, kept)
                if binding.field_name == "repeated_failed_route"
                else binding
            )
            for binding in checked.evidence_bindings
        )
        return replace(checked, evidence_bindings=bindings)

    args["verify_repeat"] = verify
    result = execute_guarded_action(action, decision, **args)
    assert result.invoked is False
    assert result.gate["route"] == "escalate"
    assert "private reviewer detail" not in canonical_json(result.gate)
    assert calls == []


def test_independently_supported_productive_repeat_can_continue():
    action, decision, args, calls = setup_guard()

    def productive(packet):
        checked = confirmed_repeat(packet)
        extra = tuple(
            VerificationEvidenceBinding(name, checked.evidence_refs)
            for name in ("task_fit", "claimed_outcome_supported")
        )
        return replace(
            checked,
            task_fit=True,
            claimed_outcome_supported=True,
            repeated_failed_route=False,
            evidence_bindings=checked.evidence_bindings + extra,
        )

    args["verify_repeat"] = productive
    result = execute_guarded_action(action, decision, **args)
    assert result.invoked is True
    assert result.gate["reason"] == "independently_supported_repeat"
    assert calls == [action]


@pytest.mark.parametrize("missing", ("binding", "original", "repeat"))
def test_productive_repeat_still_requires_both_source_observations(missing):
    action, decision, args, calls = setup_guard()

    def incomplete(packet):
        refs = {
            item["reference_id"]: EvidenceReference.from_dict(item)
            for item in packet["source_references"]
        }
        current = refs["current-state"]
        required = packet["required_observation_refs"]
        kept = () if missing == "binding" else (refs[required[1 if missing == "original" else 0]],)
        bindings = tuple(
            VerificationEvidenceBinding(name, (current,))
            for name in (
                "task_fit",
                "dependencies_satisfied",
                "contradiction_status",
                "provenance_intact",
                "authorization_scope_valid",
                "claimed_outcome_supported",
            )
        )
        if kept:
            bindings += (VerificationEvidenceBinding("repeated_failed_route", kept),)
        return RelationalVerificationResult(
            packet["repeat_action_id"],
            True,
            True,
            "none",
            True,
            True,
            True,
            False,
            (current, *kept),
            bindings,
        )

    args["verify_repeat"] = incomplete
    result = execute_guarded_action(action, decision, **args)
    assert result.invoked is False
    assert result.gate["route"] == "escalate"
    assert calls == []


def test_exact_limit_recheck_reserves_final_output_before_context_callback():
    action, _decision, args, calls = setup_guard(context="human_recheck")
    pending = args["pending"]
    original_prediction = args["prediction"]
    baseline = inspect_repeat_guard(
        (),
        (),
        pending,
        original_prediction,
        action,
        config=args["config"],
        authenticate_context=lambda _packet: False,
    )
    empty_padding = replace(
        original_prediction,
        predicted_outcome={
            **original_prediction.to_dict()["predicted_outcome"],
            "padding": "",
        },
    )
    overhead = len(canonical_json(empty_padding.to_dict()).encode()) - len(
        canonical_json(original_prediction.to_dict()).encode()
    )
    padding = 4 * 1024 * 1024 - len(canonical_json(baseline).encode()) - overhead
    prediction = replace(
        empty_padding,
        predicted_outcome={
            **empty_padding.to_dict()["predicted_outcome"],
            "padding": "x" * padding,
        },
    )
    action = replace(
        action, authorization_scope=loop_guard_action_scope(pending, prediction, args["config"])
    )
    decision, registry = _issue(
        RuntimeCapability.READ,
        (_grant(action=action, capabilities=(RuntimeCapability.READ,)),),
        action=action,
        clock=args["clock"],
    )
    reviews = []
    args.update(
        events=(),
        steps=(),
        prediction=prediction,
        grant_registry=registry,
        authenticate_context=lambda _packet: reviews.append(True) or True,
    )
    with pytest.raises(ValueError, match="4 MiB"):
        execute_guarded_action(action, decision, **args)
    assert reviews == []
    assert calls == []


@pytest.mark.parametrize("field", ("action", "prediction", "scope"))
def test_pending_bindings_fail_before_context_review_or_execution(field):
    action, decision, args, calls = setup_guard()
    args["authenticate_context"] = lambda _packet: pytest.fail("bad binding reached reviewer")
    if field == "action":
        action = replace(action, action_id="forged")
    elif field == "prediction":
        args["prediction"] = replace(args["prediction"], predicted_outcome="unbound")
    else:
        args["config"] = replace(args["config"], run_id="foreign-run")
    with pytest.raises(ValueError):
        execute_guarded_action(action, decision, **args)
    assert calls == []


def test_a_coherently_rewritten_action_cannot_reuse_the_original_grant():
    action, decision, args, calls = setup_guard()
    substitute = SemanticAction(DecisionKind.TOOL_USE, '{"tool":"delete","args":{"path":"x"}}')
    original = args["pending"]
    source = replace(original.request.trajectory, response=canonical_json(substitute.to_dict()))
    pending = replace(
        original,
        action=substitute,
        request=RewardRequest(source, original.request.observable_references),
    )
    args.update(
        pending=pending,
        prediction=replace(
            args["prediction"],
            predicted_outcome={"deloop": pending.commitment()},
        ),
    )
    with pytest.raises(ValueError, match="scope"):
        execute_guarded_action(action, decision, **args)
    assert calls == []


def test_live_guard_reports_remain_optimizer_ineligible():
    from gepa_mindfulness.training.eligibility import require_training_eligible

    action, decision, args, _ = setup_guard()
    report = execute_guarded_action(action, decision, **args).gate
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible({"training_eligibility": "TRAIN", "source_record": report})


def test_complete_review_capacity_is_reserved_before_any_callback():
    action, decision, args, calls = setup_guard()
    repeated = args["pending"].action
    events, steps = fixture(actions=(repeated,) * 8, extra_ref_count=200)
    args.update(events=events, steps=steps)
    args["authenticate_context"] = lambda _packet: pytest.fail("oversized review reached callback")
    args["verify_repeat"] = lambda _packet: pytest.fail("oversized review reached verifier")
    with pytest.raises(ValueError):
        execute_guarded_action(action, decision, **args)
    assert calls == []


def test_configuration_roundtrip_is_closed_and_cannot_grant_authority():
    config = LoopGuardConfig()
    assert LoopGuardConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        LoopGuardConfig.from_dict({**config.to_dict(), "authority_granted": True})
