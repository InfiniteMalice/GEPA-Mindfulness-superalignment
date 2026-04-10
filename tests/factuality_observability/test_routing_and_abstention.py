from gepa_mindfulness.factuality_observability.routing import RoutingContext, choose_routing_action
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction


def test_routing_prefers_abstention_when_budget_exhausted() -> None:
    decision = choose_routing_action(
        RoutingContext(
            base_case_label=12,
            operational_confidence=0.2,
            claim_complexity=0.8,
            domain_risk=0.9,
            verification_budget=0,
            has_provenance=False,
            trace_worthy=True,
            abstention_viable=True,
            guessing_pressure=0.9,
        )
    )
    assert decision.recommended_action is RecommendedAction.ABSTAIN
