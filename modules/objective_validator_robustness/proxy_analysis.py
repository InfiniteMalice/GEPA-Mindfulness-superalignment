"""Proxy-objective assessment for objective robustness.

The checks here are deterministic scaffolding. They identify inspectable
signals that a supplied objective may be a proxy whose usefulness depended on
the design context. They do not claim to measure true causal correlation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .schema import ObjectiveSpecification, OptimizationPressure, ProxyObjectiveAssessment

PROXY_CUES = (
    "metric",
    "score",
    "reward",
    "loss",
    "benchmark",
    "validator",
    "rubric",
    "unit test",
    "schema",
    "success condition",
    "minimize",
    "maximize",
    "optimize",
    "proxy",
)

ASSUMPTION_CUES = (
    "known context",
    "training environment",
    "base policy",
    "dirt",
    "grass",
    "reference policy",
    "original policy",
)

GAMING_CUES = (
    "loophole",
    "game",
    "gaming",
    "hack",
    "reward hacking",
    "checker",
    "test harness",
    "hidden harm",
    "metric can be improved",
    "without improving",
)

BREAKDOWN_CUES = (
    "correlation drop",
    "correlation breakdown",
    "proxy improves",
    "proxy score improves",
    "harm increases",
    "real-world harm increases",
    "outside base",
    "outside the base",
    "lava",
    "shortcut",
    "optimized policy",
)


def assess_proxy_objective(
    specification: ObjectiveSpecification,
    *,
    base_policy_evidence: object | None = None,
    optimized_policy_evidence: object | None = None,
) -> ProxyObjectiveAssessment:
    """Assess whether an objective is likely to be an incomplete proxy."""

    text = _join_text(
        specification.objective_text,
        specification.explicit_success_condition,
        specification.implicit_success_condition,
        specification.designer_context_summary,
        specification.training_environment_summary,
        specification.deployment_environment_summary,
        specification.known_constraints,
        specification.known_tradeoffs,
        specification.metadata,
    )
    base_text = _coerce_text(base_policy_evidence)
    optimized_text = _coerce_text(optimized_policy_evidence)
    all_text = " ".join([text, base_text, optimized_text]).lower()

    reasons: list[str] = []
    proxy_features = sorted({cue for cue in PROXY_CUES if cue in all_text})
    if proxy_features:
        reasons.append("Objective contains metric/reward/validator cues that may be proxy-like")

    if any(cue in all_text for cue in ASSUMPTION_CUES):
        reasons.append("Design context includes assumptions under which the proxy was useful")

    optimization_pressure = _optimization_pressure(specification, all_text)
    if optimization_pressure in {"high", "extreme"}:
        reasons.append("Optimization pressure may move behavior outside the base-policy region")

    evaluator_gaming_risk = any(cue in all_text for cue in GAMING_CUES)
    if evaluator_gaming_risk:
        reasons.append("Evaluator or metric can plausibly be gamed without improving intent")

    correlation_breakdown_detected = any(cue in all_text for cue in BREAKDOWN_CUES)
    if correlation_breakdown_detected:
        reasons.append("Proxy improvement may no longer track intended real-world outcome")

    conflict_with_values = bool(
        specification.imperatives_relevant and specification.known_tradeoffs
    )
    if conflict_with_values:
        reasons.append("Objective has known tradeoffs against broader imperatives")

    proxy_likelihood = 0.15
    proxy_likelihood += 0.12 * min(len(proxy_features), 4)
    if specification.explicit_success_condition or specification.implicit_success_condition:
        proxy_likelihood += 0.15
    if specification.known_tradeoffs:
        proxy_likelihood += 0.12
    if evaluator_gaming_risk:
        proxy_likelihood += 0.15
    if correlation_breakdown_detected:
        proxy_likelihood += 0.2
    proxy_likelihood = _clip(proxy_likelihood)

    confidence_objective_is_complete = _clip(1.0 - proxy_likelihood)
    reward_hacking_risk = bool(
        evaluator_gaming_risk
        or correlation_breakdown_detected
        or (proxy_likelihood >= 0.55 and optimization_pressure in {"high", "extreme"})
    )
    review_required = bool(
        correlation_breakdown_detected
        or evaluator_gaming_risk
        or optimization_pressure == "extreme"
        or conflict_with_values
    )

    return ProxyObjectiveAssessment(
        objective_id=specification.objective_id,
        proxy_features=proxy_features,
        intended_direction_summary=_intended_direction(specification),
        confidence_objective_is_complete=confidence_objective_is_complete,
        proxy_likelihood=proxy_likelihood,
        proxy_correlation_confidence="heuristic",
        optimization_pressure=optimization_pressure,
        base_policy_reference=_reference_from_evidence(base_policy_evidence),
        current_policy_reference=_reference_from_evidence(optimized_policy_evidence),
        correlation_breakdown_detected=correlation_breakdown_detected,
        evaluator_gaming_risk=evaluator_gaming_risk,
        reward_hacking_risk=reward_hacking_risk,
        reasons=reasons,
        review_required=review_required,
        metadata={
            "analysis_mode": "heuristic",
            "correlation_claim": "unavailable_without_empirical_policy_evidence",
        },
    )


def proxy_overlay_tier(assessment: ProxyObjectiveAssessment) -> str:
    """Map proxy analysis to additive P0..P5 overlay tiers."""

    if assessment.proxy_correlation_confidence == "unavailable" and not assessment.proxy_features:
        return "P0"
    if assessment.proxy_likelihood < 0.35 and not assessment.review_required:
        return "P1"
    if assessment.proxy_likelihood >= 0.35 and not assessment.correlation_breakdown_detected:
        if (
            assessment.optimization_pressure in {"high", "extreme"}
            or assessment.reward_hacking_risk
        ):
            return "P3"
        return "P2"
    if assessment.correlation_breakdown_detected and not assessment.review_required:
        return "P4"
    if assessment.correlation_breakdown_detected and assessment.review_required:
        if (
            assessment.optimization_pressure in {"high", "extreme"}
            or assessment.reward_hacking_risk
        ):
            return "P5"
        return "P4"
    return "P3" if assessment.review_required else "P2"


def _optimization_pressure(
    specification: ObjectiveSpecification,
    all_text: str,
) -> OptimizationPressure:
    raw = specification.metadata.get("optimization_pressure")
    if isinstance(raw, str) and raw.lower() in {"low", "moderate", "high", "extreme"}:
        return raw.lower()  # type: ignore[return-value]
    if any(
        cue in all_text
        for cue in ("extreme", "maximize at all costs", "aggressively optimize")
    ):
        return "extreme"
    if any(cue in all_text for cue in ("high pressure", "optimize aggressively", "shortcut")):
        return "high"
    if any(cue in all_text for cue in ("optimize", "maximize", "minimize", "benchmark")):
        return "moderate"
    return "low"


def _intended_direction(specification: ObjectiveSpecification) -> str:
    if specification.implicit_success_condition:
        return specification.implicit_success_condition
    if specification.explicit_success_condition:
        return specification.explicit_success_condition
    return specification.objective_text


def _reference_from_evidence(evidence: object | None) -> str | None:
    if isinstance(evidence, Mapping):
        for key in ("reference", "policy_reference", "id", "policy_id"):
            value = evidence.get(key)
            if value:
                return str(value)
    return None


def _join_text(*parts: object) -> str:
    return " ".join(_coerce_text(part) for part in parts if part is not None).lower()


def _coerce_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_coerce_text(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_coerce_text(item) for item in value)
    return str(value)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
