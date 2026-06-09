"""Inverse-objective interpretation under proxy and novelty uncertainty."""

from __future__ import annotations

from .schema import (
    NoveltyAssessment,
    ObjectivePosterior,
    ObjectiveSpecification,
    PlausibleObjective,
    ProxyObjectiveAssessment,
)


def infer_objective_posterior(
    specification: ObjectiveSpecification,
    proxy_assessment: ProxyObjectiveAssessment,
    novelty_assessment: NoveltyAssessment,
) -> ObjectivePosterior:
    """Infer plausible intended objectives from the supplied objective and context.

    This encodes the Inverse Reward Design insight at a lightweight interface
    level: the supplied objective is evidence about intended direction in the
    design context, not a complete license for literal maximization elsewhere.
    """

    plausible: list[PlausibleObjective] = []
    literal_weight = 0.72
    if proxy_assessment.proxy_likelihood >= 0.5:
        literal_weight -= 0.18
    if novelty_assessment.novel_state_detected:
        literal_weight -= 0.16
    if proxy_assessment.correlation_breakdown_detected:
        literal_weight -= 0.18
    literal_weight = _clip(literal_weight)

    plausible.append(
        PlausibleObjective(
            plausible_objective_id=f"{specification.objective_id}:literal",
            objective_id=specification.objective_id,
            interpretation_summary=specification.objective_text,
            supporting_evidence=["Supplied objective text"],
            conflicting_evidence=_conflicts(proxy_assessment, novelty_assessment),
            posterior_weight=literal_weight,
            worst_case_outcome_summary=_literal_worst_case(proxy_assessment, novelty_assessment),
            best_case_outcome_summary=(
                "Achieves the stated success condition in the design context."
            ),
            catastrophic_downside_possible=_catastrophic_possible(novelty_assessment),
            constraints=list(specification.known_constraints),
            metadata={"inference_mode": "heuristic"},
        )
    )

    if proxy_assessment.proxy_likelihood >= 0.35 or novelty_assessment.novel_state_detected:
        plausible.append(
            PlausibleObjective(
                plausible_objective_id=f"{specification.objective_id}:intended-direction",
                objective_id=specification.objective_id,
                interpretation_summary=(
                    proxy_assessment.intended_direction_summary
                    or "Advance the intended outcome behind the proxy, within known constraints."
                ),
                supporting_evidence=[
                    "Objective is treated as directional evidence from its design context",
                    *proxy_assessment.reasons,
                ],
                conflicting_evidence=[],
                posterior_weight=_clip(0.35 + proxy_assessment.proxy_likelihood * 0.2),
                worst_case_outcome_summary=(
                    "Under-scoped interpretation may miss a legitimate shortcut, "
                    "but avoids blind proxy maximization."
                ),
                best_case_outcome_summary=(
                    "Preserves intended outcome while rejecting loophole optimization."
                ),
                catastrophic_downside_possible=False,
                constraints=sorted(
                    set(specification.known_constraints + specification.imperatives_relevant)
                ),
                metadata={"inference_mode": "heuristic"},
            )
        )

    if novelty_assessment.novel_state_detected or proxy_assessment.reward_hacking_risk:
        plausible.append(
            PlausibleObjective(
                plausible_objective_id=f"{specification.objective_id}:bounded-safe",
                objective_id=specification.objective_id,
                interpretation_summary=(
                    "Make reversible, bounded progress while preserving optionality "
                    "and seeking clarification or review."
                ),
                supporting_evidence=[
                    *novelty_assessment.reasons,
                    *proxy_assessment.reasons,
                ],
                conflicting_evidence=["May slow optimization when the proxy remains adequate"],
                posterior_weight=_clip(
                    0.25
                    + (0.2 if novelty_assessment.novel_state_detected else 0.0)
                    + (0.2 if proxy_assessment.reward_hacking_risk else 0.0)
                ),
                worst_case_outcome_summary=(
                    "Unnecessary delay or conservative bounded action in a benign case."
                ),
                best_case_outcome_summary=(
                    "Avoids irreversible harm while retaining the ability to update."
                ),
                catastrophic_downside_possible=False,
                constraints=sorted(
                    set(
                        specification.known_constraints
                        + ["preserve optionality", "prefer reversible actions"]
                    )
                ),
                metadata={"inference_mode": "heuristic"},
            )
        )

    plausible = _normalize_weights(plausible)
    ambiguity_remaining = len(plausible) > 1 and (
        novelty_assessment.novel_state_detected
        or proxy_assessment.proxy_likelihood >= 0.45
        or proxy_assessment.correlation_breakdown_detected
    )
    clarification_required = bool(
        ambiguity_remaining
        and (
            novelty_assessment.clarification_recommended
            or proxy_assessment.correlation_breakdown_detected
        )
    )
    review_required = bool(
        proxy_assessment.review_required
        or novelty_assessment.review_required
        or any(item.catastrophic_downside_possible for item in plausible)
    )

    posterior_confidence = _clip(
        max(item.posterior_weight for item in plausible) if plausible else 0.0
    )
    return ObjectivePosterior(
        objective_id=specification.objective_id,
        plausible_objectives=plausible,
        posterior_confidence=posterior_confidence,
        ambiguity_remaining=ambiguity_remaining,
        clarification_required=clarification_required,
        review_required=review_required,
        metadata={
            "inference_mode": "heuristic",
            "posterior_calibration": "not_mathematically_calibrated",
        },
    )


def _conflicts(
    proxy_assessment: ProxyObjectiveAssessment,
    novelty_assessment: NoveltyAssessment,
) -> list[str]:
    conflicts: list[str] = []
    if proxy_assessment.proxy_likelihood >= 0.5:
        conflicts.append("Objective appears proxy-like or incomplete")
    if proxy_assessment.correlation_breakdown_detected:
        conflicts.append("Proxy correlation may break under optimization")
    if novelty_assessment.novel_state_detected:
        conflicts.append("Deployment context contains novel features")
    return conflicts


def _literal_worst_case(
    proxy_assessment: ProxyObjectiveAssessment,
    novelty_assessment: NoveltyAssessment,
) -> str:
    if novelty_assessment.novel_features:
        return "Literal optimization may exploit novel features outside the design context."
    if proxy_assessment.correlation_breakdown_detected:
        return "Literal optimization may improve the proxy while worsening intended outcomes."
    return "Literal optimization may miss unstated constraints if the objective is incomplete."


def _catastrophic_possible(novelty_assessment: NoveltyAssessment) -> bool:
    return bool(
        novelty_assessment.stakes_level == "catastrophic"
        or (
            novelty_assessment.stakes_level == "high"
            and novelty_assessment.irreversible_action_requested
        )
        or "lava" in novelty_assessment.novel_features
    )


def _normalize_weights(plausible: list[PlausibleObjective]) -> list[PlausibleObjective]:
    total = sum(max(item.posterior_weight, 0.0) for item in plausible)
    if total <= 0:
        return plausible
    normalized: list[PlausibleObjective] = []
    for item in plausible:
        payload = item.__dict__.copy()
        payload["posterior_weight"] = round(max(item.posterior_weight, 0.0) / total, 4)
        normalized.append(PlausibleObjective(**payload))
    return normalized


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
