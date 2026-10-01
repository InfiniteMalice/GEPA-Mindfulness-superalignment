"""System-One proposals cannot bypass evidence, chronology or authority boundaries."""

from dataclasses import replace

import pytest

from gepa_mindfulness.factuality_observability.epistemic_routing import (
    RoutingBackend,
    RoutingPolicy,
    RoutingRequest,
    route_epistemic,
)
from gepa_mindfulness.factuality_observability.routing import RoutingContext, choose_routing_action
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from gepa_mindfulness.verification.epistemic_state import MismatchStatus
from gepa_mindfulness.verification.interfaces import VerificationEvidenceBinding
from mindful_trace_gepa.action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
    make_action_event,
    make_outcome_observation_event,
    make_prediction_commit_event,
    make_verification_result_event,
)
from mindful_trace_gepa.logging_schema import StructuredEventType
from tests.test_epistemic_reconciliation import (
    EVIDENCE,
    metadata,
    reconciliation,
    relational_verifier,
    sequence,
)

ENABLED = RoutingPolicy(enabled=True)


def request(
    *,
    typed=True,
    mismatch=MismatchStatus.NONE,
    aggregate_mismatch=MismatchStatus.NONE,
    **uncertainty,
):
    record = reconciliation()
    posterior = replace(
        record.update.posterior_state,
        **(
            dict(world_uncertainty=0.1, model_uncertainty=0.1, monitor_uncertainty=0.1)
            | uncertainty
        ),
    )
    record = replace(
        record,
        update=replace(record.update, posterior_state=posterior, model_mismatch=aggregate_mismatch),
        bindings=tuple(
            replace(b, innovation=replace(b.innovation, mismatch_status=mismatch))
            for b in record.bindings
        ),
    )
    events = sequence(record)
    if typed:
        fields = (
            "task_fit",
            "dependencies_satisfied",
            "contradiction_status",
            "provenance_intact",
            "authorization_scope_valid",
            "claimed_outcome_supported",
        )
        events[4] = relational_verifier(
            task_fit=True,
            dependencies_satisfied=True,
            contradiction_status="none",
            authorization_scope_valid=True,
            evidence_bindings=tuple(VerificationEvidenceBinding(f, (EVIDENCE,)) for f in fields),
        )
    events.append(
        make_prediction_commit_event(
            PredictionCommit("next-prediction", {"continued": True}, 0.9, ("sensor-log",)),
            **metadata("next-p", 6),
        )
    )
    events.append(
        make_action_event(
            ActionRecord("next", "continue", True, "read-only", "next-prediction"),
            StructuredEventType.ACTION_PROPOSED,
            **metadata("decision", 7, ("next-p",)),
        )
    )
    return RoutingRequest(
        events=tuple(
            replace(e, conversation_id="conversation", checkpoint_step=i)
            for i, e in enumerate(events)
        ),
        decision_event_id="decision",
        context=RoutingContext(1, 0.95, 0.1, 0.1, 3, True, False, True, 0.0, False),
    )


def backend(action=Action.ACCEPT):
    return RoutingBackend("test-candidate", "v1", lambda _: action)


def test_opt_in_and_legacy_compatibility():
    req = request()
    assert route_epistemic(req) is None
    assert choose_routing_action(req.context).recommended_action == Action.ACCEPT
    result = route_epistemic(req, policy=ENABLED)
    assert result.action == Action.ACCEPT
    assert result.to_dict()["confers_authority"] is False
    assert result.reconciliation_event_id == "r"
    assert result.decision_event_id == "decision"
    assert len(result.input_digest) == 64


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"world_uncertainty": 0.8}, Action.RETRIEVE_MORE),
        ({"monitor_uncertainty": 0.8}, Action.ROUTE_EXTERNAL),
        ({"model_uncertainty": 0.8}, Action.ROUTE_EXTERNAL),
        ({"monitor_uncertainty": None}, Action.ROUTE_EXTERNAL),
        ({"model_uncertainty": None}, Action.ROUTE_EXTERNAL),
        ({"world_uncertainty": None}, Action.RETRIEVE_MORE),
        ({"mismatch": MismatchStatus.MODEL_MISMATCH}, Action.DECOMPOSE_AND_VERIFY),
        ({"mismatch": MismatchStatus.UNASSESSED}, Action.DECOMPOSE_AND_VERIFY),
        ({"typed": False}, Action.ROUTE_EXTERNAL),
    ],
)
def test_epistemic_gates_override_accept(values, expected):
    result = route_epistemic(request(**values), policy=ENABLED, backend=backend())
    assert result.proposed_action == Action.ACCEPT
    assert result.action == expected
    assert result.overridden


@pytest.mark.parametrize("aggregate", tuple(MismatchStatus))
@pytest.mark.parametrize("binding", tuple(MismatchStatus))
def test_aggregate_and_binding_mismatches_both_constrain_continuation(aggregate, binding):
    """Only explicit NONE at both levels can permit the legacy ACCEPT proposal."""
    req = request(aggregate_mismatch=aggregate, mismatch=binding)
    result = route_epistemic(req, policy=ENABLED, backend=backend())
    mismatch = aggregate != MismatchStatus.NONE or binding != MismatchStatus.NONE
    assert result.proposed_action == Action.ACCEPT
    assert result.features.mismatch is mismatch
    assert result.action == (Action.DECOMPOSE_AND_VERIFY if mismatch else Action.ACCEPT)
    assert result.overridden is mismatch


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"verification_budget": 0}, Action.ABSTAIN),
        ({"verification_budget": 0, "abstention_viable": False}, Action.ESCALATE),
        ({"representation_sensitive": True}, Action.ROUTE_EXTERNAL),
        ({"verification_required": True}, Action.ROUTE_EXTERNAL),
        ({"domain_risk": 0.9}, Action.ROUTE_EXTERNAL),
        ({"claim_complexity": 0.8, "operational_confidence": 0.8}, Action.DECOMPOSE_AND_VERIFY),
    ],
)
def test_legacy_and_high_impact_gates(values, expected):
    req = request()
    req = replace(req, context=replace(req.context, **values))
    assert route_epistemic(req, policy=ENABLED, backend=backend()).action == expected


def test_irreversible_proposal_requires_external_review():
    req = request()
    decision = make_action_event(
        ActionRecord("next", "publish", False, "publish", "next-prediction"),
        StructuredEventType.ACTION_PROPOSED,
        **metadata("decision", 7, ("next-p",)),
    )
    decision = replace(decision, conversation_id="conversation", checkpoint_step=7)
    req = replace(req, events=(*req.events[:-1], decision))
    assert route_epistemic(req, policy=ENABLED, backend=backend()).action == Action.ROUTE_EXTERNAL


def test_missing_stale_and_changed_context_fail_closed():
    req = request()
    missing = replace(req, events=req.events[-2:])
    changed = replace(req, context_changed=True)
    stale = replace(
        req, events=(*req.events[:-1], replace(req.events[-1], timestamp="2026-09-30T13:00:00Z"))
    )
    for item in (missing, changed, stale):
        result = route_epistemic(item, policy=ENABLED, backend=backend())
        assert result.action == Action.ROUTE_EXTERNAL


def test_foreign_state_does_not_enable_accept():
    req = request()
    for changes in (
        {"conversation_id": "elsewhere"},
        {"model_version": "new", "run_id": "new-run"},
    ):
        foreign = replace(
            req, events=(*req.events[:-2], *(replace(e, **changes) for e in req.events[-2:]))
        )
        assert route_epistemic(foreign, policy=ENABLED, backend=backend()).action != Action.ACCEPT

    drift = replace(req, events=(*req.events[:-1], replace(req.events[-1], model_version="new")))
    with pytest.raises(ValueError, match="drift"):
        route_epistemic(drift, policy=ENABLED)


def test_future_reconciliation_cannot_enable_continuation():
    req = request()
    future = replace(req, events=(*req.events[:5], *req.events[-2:], req.events[5]))
    with pytest.raises(ValueError):
        route_epistemic(future, policy=ENABLED)


@pytest.mark.parametrize(
    "field,value",
    [
        ("operational_confidence", float("nan")),
        ("domain_risk", True),
        ("verification_budget", True),
        ("verification_required", 1),
        ("base_case_label", 18),
        ("claim_complexity", -0.1),
    ],
)
def test_invalid_context_rejected(field, value):
    req = request()
    with pytest.raises(ValueError):
        route_epistemic(
            replace(req, context=replace(req.context, **{field: value})), policy=ENABLED
        )


def test_backend_failure_invalid_output_and_conservative_stop():
    def failed(_):
        raise RuntimeError("untrusted contents must not reach diagnostics")

    result = route_epistemic(
        request(), policy=ENABLED, backend=RoutingBackend("broken", "v1", failed)
    )
    assert result.action == Action.ESCALATE
    assert result.backend_error == "RuntimeError"
    assert "untrusted contents" not in str(result.to_dict())
    for action in ("accept", {"action": "accept"}, None):
        result = route_epistemic(request(), policy=ENABLED, backend=backend(action))
        assert result.action == Action.ESCALATE
        assert result.backend_error == "invalid_proposal"
    assert route_epistemic(request(), policy=ENABLED, backend=backend(Action.ESCALATE)).action == (
        Action.ESCALATE
    )


def test_callback_cannot_rewrite_guard_inputs():
    req = request(world_uncertainty=0.9)

    def mutate(features):
        req.context.verification_required = False
        object.__setattr__(features, "required_action", Action.ACCEPT)
        return Action.ACCEPT

    result = route_epistemic(req, policy=ENABLED, backend=RoutingBackend("mutator", "v1", mutate))
    assert result.action == Action.RETRIEVE_MORE
    data = result.to_dict()
    data["action"] = "accept"
    assert result.action == Action.RETRIEVE_MORE


def test_reconciliation_expiry_and_foreign_ancestor_fail_closed():
    req = request()
    for index, changes in (
        (5, {"valid_until": "2026-09-30T12:00:06Z"}),
        (5, {"valid_from": "2026-09-30T12:00:08Z"}),
        (4, {"conversation_id": "foreign"}),
    ):
        events = list(req.events)
        events[index] = replace(events[index], **changes)
        result = route_epistemic(
            replace(req, events=tuple(events)), policy=ENABLED, backend=backend()
        )
        assert result.action == Action.ROUTE_EXTERNAL


def test_nonchronological_prefix_rejected():
    req = request()
    events = list(req.events)
    events[5] = replace(events[5], timestamp="2026-09-30T12:00:06.5Z")
    with pytest.raises(ValueError, match="chronolog"):
        route_epistemic(replace(req, events=tuple(events)), policy=ENABLED)


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": 1},
        {"uncertainty_threshold": float("nan")},
        {"high_risk_threshold": True},
        {"max_state_age_seconds": -1},
    ],
)
def test_policy_validation(values):
    with pytest.raises(ValueError):
        RoutingPolicy(**values)


@pytest.mark.parametrize("position", [5, 7])
def test_uncited_negative_verifier_blocks_even_without_envelope_action_id(position):
    req = request()
    second = 4 if position == 5 else 6
    negative = make_verification_result_event(
        VerificationResult("negative", "v1", "observation", False, ("negative-log",)),
        **metadata("negative", second, ("o",)),
    )
    assert negative.action_id is None
    events = list(req.events)
    events.insert(
        position, replace(negative, conversation_id="conversation", checkpoint_step=second)
    )
    result = route_epistemic(replace(req, events=tuple(events)), policy=ENABLED, backend=backend())
    assert result.action == Action.ROUTE_EXTERNAL
    assert result.features.reason == "unreconciled_evidence"
    assert not result.features.verified_evidence


def test_later_outcome_requires_new_reconciliation():
    req = request()
    observation = make_outcome_observation_event(
        OutcomeObservation("new-observation", "action", {"temperature": [99]}, ("new-sensor",)),
        **metadata("new-observation", 6, ("x",)),
    )
    observation = replace(observation, conversation_id="conversation", checkpoint_step=6)
    req = replace(req, events=(*req.events[:-1], observation, req.events[-1]))
    result = route_epistemic(req, policy=ENABLED, backend=backend())
    assert result.action == Action.ROUTE_EXTERNAL
    assert result.features.reason == "unreconciled_evidence"
