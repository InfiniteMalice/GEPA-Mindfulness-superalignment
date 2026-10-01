"""Matched routing comparisons expose backend mistakes before guard correction."""

from dataclasses import replace

import pytest

from evaluation.system_one_benchmark import RoutingCase, benchmark_routing
from gepa_mindfulness.factuality_observability.epistemic_routing import (
    LEGACY_BACKEND,
    UNCERTAINTY_BACKEND,
    RoutingBackend,
)
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from gepa_mindfulness.verification.epistemic_state import MismatchStatus
from tests.test_epistemic_routing import ENABLED, request


def benchmark_cases():
    cases = [RoutingCase("verified-low", request(), (Action.ACCEPT,))]
    for name, values, action in (
        ("world", {"world_uncertainty": 0.8}, Action.RETRIEVE_MORE),
        ("model", {"model_uncertainty": 0.8}, Action.ROUTE_EXTERNAL),
        ("monitor", {"monitor_uncertainty": 0.8}, Action.ROUTE_EXTERNAL),
        ("world-unknown", {"world_uncertainty": None}, Action.RETRIEVE_MORE),
        ("model-unknown", {"model_uncertainty": None}, Action.ROUTE_EXTERNAL),
        ("monitor-unknown", {"monitor_uncertainty": None}, Action.ROUTE_EXTERNAL),
        ("mismatch", {"mismatch": MismatchStatus.MODEL_MISMATCH}, Action.DECOMPOSE_AND_VERIFY),
        ("unassessed", {"mismatch": MismatchStatus.UNASSESSED}, Action.DECOMPOSE_AND_VERIFY),
        ("untyped-verifier", {"typed": False}, Action.ROUTE_EXTERNAL),
    ):
        cases.append(RoutingCase(name, request(**values), (action,)))
    for name, values, action in (
        ("budget", {"verification_budget": 0}, Action.ABSTAIN),
        (
            "budget-escalate",
            {"verification_budget": 0, "abstention_viable": False},
            Action.ESCALATE,
        ),
        ("representation", {"representation_sensitive": True}, Action.ROUTE_EXTERNAL),
        ("mandatory-verification", {"verification_required": True}, Action.ROUTE_EXTERNAL),
        ("high-risk", {"domain_risk": 0.9}, Action.ROUTE_EXTERNAL),
    ):
        req = request()
        cases.append(
            RoutingCase(name, replace(req, context=replace(req.context, **values)), (action,))
        )
    return tuple(cases)


def test_matched_baselines_report_raw_and_guarded_results():
    report = benchmark_routing(
        benchmark_cases(), (LEGACY_BACKEND, UNCERTAINTY_BACKEND), policy=ENABLED
    )
    legacy, rules = report["backends"]
    assert legacy["cases"] == rules["cases"] == 15
    assert legacy["raw_correct"] == 5
    assert legacy["disallowed_proposals"] == 10
    assert legacy["guarded_correct"] == rules["guarded_correct"] == 15
    assert rules["raw_correct"] == 15
    assert rules["disallowed_proposals"] == 0
    assert legacy["mean_latency_ms"] >= 0
    assert report["automatic_promotion"] is False
    for a, b in zip(legacy["results"], rules["results"]):
        assert a["assessment"]["features"]["input_digest"] == (
            b["assessment"]["features"]["input_digest"]
        )


def test_backend_error_is_counted_even_if_escalation_matches_label():
    def broken(_):
        raise RuntimeError("failure")

    report = benchmark_routing(
        (RoutingCase("stop", request(), (Action.ESCALATE,)),),
        (RoutingBackend("broken", "v1", broken),),
        policy=ENABLED,
    )
    result = report["backends"][0]
    assert result["backend_failures"] == 1
    assert result["raw_correct"] == 0
    assert result["guarded_correct"] == 1


def test_comparison_validates_all_cases_before_invoking_any_backend():
    calls = []
    req = request()
    invalid = replace(req, context=replace(req.context, domain_risk=float("nan")))
    cases = (
        RoutingCase("good", req, (Action.ACCEPT,)),
        RoutingCase("bad", invalid, (Action.ACCEPT,)),
    )
    with pytest.raises(ValueError):
        benchmark_routing(
            cases, (RoutingBackend("spy", "v1", lambda f: calls.append(f)),), policy=ENABLED
        )
    assert not calls


def test_callback_cannot_change_another_backends_inputs():
    cases = benchmark_cases()

    def mutate(_):
        cases[0].request.context.verification_required = True
        return Action.ACCEPT

    report = benchmark_routing(
        cases, (RoutingBackend("mutator", "v1", mutate), LEGACY_BACKEND), policy=ENABLED
    )
    first, second = report["backends"]
    assert first["results"][0]["assessment"]["features"] == (
        second["results"][0]["assessment"]["features"]
    )


@pytest.mark.parametrize(
    "cases,backends",
    [
        ((), (LEGACY_BACKEND,)),
        (benchmark_cases(), ()),
        ((benchmark_cases()[0],) * 2, (LEGACY_BACKEND,)),
        (benchmark_cases(), (LEGACY_BACKEND, LEGACY_BACKEND)),
    ],
)
def test_invalid_comparison_inventory(cases, backends):
    with pytest.raises(ValueError):
        benchmark_routing(cases, backends, policy=ENABLED)
