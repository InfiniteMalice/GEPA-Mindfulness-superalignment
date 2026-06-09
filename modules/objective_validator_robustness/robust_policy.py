"""Robust policy selection under objective proxy uncertainty."""

from __future__ import annotations

from uuid import uuid4

from .schema import (
    NoveltyAssessment,
    ObjectivePosterior,
    ObjectiveSpecification,
    ObjectiveValidationInterrupt,
    ProxyObjectiveAssessment,
    RobustObjectiveDecision,
)


def decide_robust_objective_policy(
    specification: ObjectiveSpecification,
    proxy_assessment: ProxyObjectiveAssessment,
    novelty_assessment: NoveltyAssessment,
    posterior: ObjectivePosterior,
) -> RobustObjectiveDecision:
    """Select a bounded objective policy without blindly maximizing a proxy."""

    catastrophic = _catastrophic_plausible(posterior)
    high_irreversible = bool(
        novelty_assessment.stakes_level in {"high", "catastrophic"}
        and novelty_assessment.irreversible_action_requested
    )
    proxy_breakdown = proxy_assessment.correlation_breakdown_detected
    gaming = proxy_assessment.evaluator_gaming_risk
    ambiguity = posterior.ambiguity_remaining

    worst_case = _worst_case(posterior)
    selected = _selected_interpretation(posterior)

    if catastrophic and (high_irreversible or novelty_assessment.stakes_level == "catastrophic"):
        return RobustObjectiveDecision(
            objective_id=specification.objective_id,
            action="refuse",
            rationale_summary=(
                "A sufficiently plausible objective interpretation has catastrophic downside; "
                "blind optimization is blocked."
            ),
            selected_interpretation=selected,
            plausible_objective_count=len(posterior.plausible_objectives),
            worst_case_outcome_summary=worst_case,
            preserves_optionality=True,
            reversible=True,
            clarification_required=True,
            review_required=True,
            interrupt_required=True,
            safe_alternative=(
                "Pause execution and seek authorized review before irreversible action."
            ),
            blocked_elements=["literal_proxy_maximization", "irreversible_action"],
            allowed_elements=["safe_fallback", "clarification_request", "review_summary"],
            metadata={"decision_mode": "heuristic"},
        )

    if high_irreversible:
        return RobustObjectiveDecision(
            objective_id=specification.objective_id,
            action="escalate",
            rationale_summary=(
                "High-stakes or irreversible action under objective uncertainty requires review."
            ),
            selected_interpretation=selected,
            plausible_objective_count=len(posterior.plausible_objectives),
            worst_case_outcome_summary=worst_case,
            preserves_optionality=True,
            reversible=True,
            clarification_required=True,
            review_required=True,
            interrupt_required=True,
            safe_alternative=(
                "Use a reversible dry run or no-op plan until review resolves ambiguity."
            ),
            blocked_elements=["irreversible_action"],
            allowed_elements=["dry_run", "risk_summary", "clarifying_question"],
            metadata={"decision_mode": "heuristic"},
        )

    if proxy_breakdown or gaming:
        return RobustObjectiveDecision(
            objective_id=specification.objective_id,
            action="bound",
            rationale_summary=(
                "Proxy or evaluator-gaming risk detected; optimize only within real-world "
                "constraints and preserve contact with intended outcomes."
            ),
            selected_interpretation=selected,
            plausible_objective_count=len(posterior.plausible_objectives),
            worst_case_outcome_summary=worst_case,
            preserves_optionality=True,
            reversible=True,
            clarification_required=posterior.clarification_required,
            review_required=posterior.review_required,
            interrupt_required=posterior.review_required,
            safe_alternative="Use bounded optimization with checks against hidden harm.",
            blocked_elements=["loophole_optimization", "proxy_only_success"],
            allowed_elements=["bounded_progress", "independent_reality_check", "safe_fallback"],
            metadata={"decision_mode": "heuristic"},
        )

    if ambiguity and novelty_assessment.novel_state_detected:
        if novelty_assessment.stakes_level == "low" and novelty_assessment.reversibility == "high":
            return RobustObjectiveDecision(
                objective_id=specification.objective_id,
                action="bound",
                rationale_summary=(
                    "Low-stakes reversible novelty permits cautious progress with a "
                    "stated assumption."
                ),
                selected_interpretation=selected,
                plausible_objective_count=len(posterior.plausible_objectives),
                worst_case_outcome_summary=worst_case,
                preserves_optionality=True,
                reversible=True,
                clarification_required=False,
                review_required=False,
                interrupt_required=False,
                safe_alternative="Proceed in a reversible sandbox and record the assumption.",
                blocked_elements=["irreversible_escalation"],
                allowed_elements=["cautious_progress", "assumption_record"],
                metadata={"decision_mode": "heuristic"},
            )
        return RobustObjectiveDecision(
            objective_id=specification.objective_id,
            action="ask_clarifying",
            rationale_summary=(
                "Novel context leaves multiple plausible objective interpretations unresolved."
            ),
            selected_interpretation=selected,
            plausible_objective_count=len(posterior.plausible_objectives),
            worst_case_outcome_summary=worst_case,
            preserves_optionality=True,
            reversible=True,
            clarification_required=True,
            review_required=posterior.review_required,
            interrupt_required=posterior.review_required,
            safe_alternative="Ask which constraints govern the novel state before optimizing.",
            blocked_elements=["blind_literal_optimization"],
            allowed_elements=["clarifying_question", "safe_default"],
            metadata={"decision_mode": "heuristic"},
        )

    if proxy_assessment.proxy_likelihood >= 0.5 and proxy_assessment.optimization_pressure in {
        "high",
        "extreme",
    }:
        return RobustObjectiveDecision(
            objective_id=specification.objective_id,
            action="preserve_optionality",
            rationale_summary=(
                "Objective appears proxy-like under strong optimization; keep options open."
            ),
            selected_interpretation=selected,
            plausible_objective_count=len(posterior.plausible_objectives),
            worst_case_outcome_summary=worst_case,
            preserves_optionality=True,
            reversible=True,
            clarification_required=False,
            review_required=False,
            interrupt_required=False,
            safe_alternative="Take a small reversible step and monitor proxy/intent divergence.",
            blocked_elements=["aggressive_proxy_maximization"],
            allowed_elements=["small_reversible_step", "monitoring"],
            metadata={"decision_mode": "heuristic"},
        )

    return RobustObjectiveDecision(
        objective_id=specification.objective_id,
        action="allow",
        rationale_summary="No material proxy breakdown, novelty, or high-risk ambiguity detected.",
        selected_interpretation=selected,
        plausible_objective_count=len(posterior.plausible_objectives),
        worst_case_outcome_summary=worst_case,
        preserves_optionality=False,
        reversible=True,
        clarification_required=False,
        review_required=False,
        interrupt_required=False,
        safe_alternative=None,
        blocked_elements=[],
        allowed_elements=["ordinary_optimization"],
        metadata={"decision_mode": "heuristic"},
    )


def objective_validation_interrupt(
    specification: ObjectiveSpecification,
    proxy_assessment: ProxyObjectiveAssessment,
    novelty_assessment: NoveltyAssessment,
    decision: RobustObjectiveDecision,
) -> ObjectiveValidationInterrupt | None:
    """Create an advisory interrupt when objective review should preempt execution."""

    triggers: list[str] = []
    if novelty_assessment.novel_state_detected:
        triggers.append("novel_state_detected")
    if novelty_assessment.distribution_shift_detected:
        triggers.append("distribution_shift_detected")
    if proxy_assessment.correlation_breakdown_detected:
        triggers.append("proxy_correlation_drop")
    if proxy_assessment.optimization_pressure in {"high", "extreme"}:
        triggers.append("high_optimization_pressure")
    if specification.known_tradeoffs or specification.imperatives_relevant:
        triggers.append("objective_conflict")
    if (
        decision.worst_case_outcome_summary
        and "catastrophic" in decision.worst_case_outcome_summary
    ):
        triggers.append("catastrophic_downside_under_plausible_interpretation")
    if novelty_assessment.irreversible_action_requested:
        triggers.append("low_reversibility")
    if "authority" in " ".join(novelty_assessment.novel_features):
        triggers.append("authority_unclear")
    if proxy_assessment.evaluator_gaming_risk:
        triggers.append("evaluator_gaming_risk")
    if bool(specification.metadata.get("memory_modified_objective")):
        triggers.append("memory_modified_objective")

    if not decision.interrupt_required and not (
        decision.review_required
        and any(trigger in triggers for trigger in ("low_reversibility", "proxy_correlation_drop"))
    ):
        return None

    urgency = "moderate"
    if (
        decision.action in {"refuse", "escalate"}
        or novelty_assessment.stakes_level == "catastrophic"
    ):
        urgency = "immediate"
    elif decision.review_required:
        urgency = "high"

    return ObjectiveValidationInterrupt(
        interrupt_id=f"ovi-{uuid4()}",
        objective_id=specification.objective_id,
        trigger_types=sorted(set(triggers)),
        urgency=urgency,  # type: ignore[arg-type]
        time_to_review="before_next_irreversible_step",
        interrupt_current_task=decision.interrupt_required,
        reason_summary=decision.rationale_summary,
        safe_fallback=decision.safe_alternative,
        review_required=decision.review_required,
        metadata={"interrupt_mode": "advisory_control_signal"},
    )


def _catastrophic_plausible(posterior: ObjectivePosterior) -> bool:
    return any(
        item.catastrophic_downside_possible and item.posterior_weight >= 0.1
        for item in posterior.plausible_objectives
    )


def _worst_case(posterior: ObjectivePosterior) -> str | None:
    for item in posterior.plausible_objectives:
        if item.catastrophic_downside_possible and item.worst_case_outcome_summary:
            return item.worst_case_outcome_summary
    for item in posterior.plausible_objectives:
        if item.worst_case_outcome_summary:
            return item.worst_case_outcome_summary
    return None


def _selected_interpretation(posterior: ObjectivePosterior) -> str | None:
    if not posterior.plausible_objectives:
        return None
    selected = max(posterior.plausible_objectives, key=lambda item: item.posterior_weight)
    return selected.plausible_objective_id
