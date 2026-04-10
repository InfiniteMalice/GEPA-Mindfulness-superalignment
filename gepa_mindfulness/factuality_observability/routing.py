"""Verification-routing decisions with budget and risk awareness."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import RecommendedAction


@dataclass(slots=True)
class RoutingContext:
    """Inputs for routing action selection."""

    base_case_label: int
    operational_confidence: float
    claim_complexity: float
    domain_risk: float
    verification_budget: int
    has_provenance: bool
    trace_worthy: bool
    abstention_viable: bool
    guessing_pressure: float


@dataclass(slots=True)
class RoutingDecision:
    """Routing outcome and path trace."""

    recommended_action: RecommendedAction
    routing_path: list[str]
    routing_target: str


def choose_routing_action(context: RoutingContext) -> RoutingDecision:
    """Pick next action without relying on a single universal judge."""

    path = [f"case={context.base_case_label}"]

    if context.verification_budget <= 0:
        if context.abstention_viable:
            path.append("budget_exhausted->abstain")
            return RoutingDecision(RecommendedAction.ABSTAIN, path, "abstention_handler")
        path.append("budget_exhausted->human")
        return RoutingDecision(RecommendedAction.ESCALATE, path, "human_review")

    if context.domain_risk >= 0.8 and context.operational_confidence < 0.8:
        path.append("high_risk_low_confidence->external_checker")
        return RoutingDecision(
            RecommendedAction.ROUTE_EXTERNAL,
            path,
            "specialized_verifier",
        )

    if context.claim_complexity >= 0.6 and context.operational_confidence < 0.85:
        path.append("complex_claim->decompose_verify")
        return RoutingDecision(
            RecommendedAction.DECOMPOSE_AND_VERIFY,
            path,
            "atomic_fact_pipeline",
        )

    if not context.has_provenance and context.operational_confidence < 0.75:
        path.append("missing_provenance->retrieve_more")
        return RoutingDecision(RecommendedAction.RETRIEVE_MORE, path, "retrieval_stack")

    if (
        context.guessing_pressure > 0.7
        and context.abstention_viable
        and context.operational_confidence < 0.6
    ):
        path.append("guessing_pressure->abstain")
        return RoutingDecision(RecommendedAction.ABSTAIN, path, "abstention_handler")

    if context.trace_worthy:
        path.append("trace_worthy->downgrade_capture_trace")
        return RoutingDecision(
            RecommendedAction.DOWNGRADE_CONFIDENCE,
            path,
            "trace_queue",
        )

    path.append("accept")
    return RoutingDecision(RecommendedAction.ACCEPT, path, "final_output")
