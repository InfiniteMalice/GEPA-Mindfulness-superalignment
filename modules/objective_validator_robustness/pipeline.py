"""Orchestration for objective, validator, proxy, and novelty robustness."""

from __future__ import annotations

from typing import Any

from .decomposition import decompose_objective
from .detection import detect_validator_capture, validator_overlay_tier
from .inverse_objective import infer_objective_posterior
from .novelty import assess_objective_novelty
from .policy import decide_validator_policy
from .proxy_analysis import assess_proxy_objective, proxy_overlay_tier
from .robust_policy import decide_robust_objective_policy, objective_validation_interrupt
from .schema import (
    ObjectiveSpecification,
    ProxyBreakdownReport,
)

OBJECTIVE_EVENT_TYPES = (
    "objective_specification",
    "validator_capture_assessment",
    "proxy_objective_assessment",
    "novelty_assessment",
    "objective_posterior_update",
    "robust_objective_decision",
    "proxy_breakdown_report",
    "objective_validation_interrupt",
)


def evaluate_objective_robustness(
    specification: ObjectiveSpecification,
    *,
    observed_environment: object | None = None,
    base_policy_evidence: object | None = None,
    optimized_policy_evidence: object | None = None,
) -> ProxyBreakdownReport:
    """Evaluate objective robustness with additive proxy and novelty overlays."""

    structure = decompose_objective(
        specification.objective_text,
        metadata={
            "domain": specification.metadata.get("domain", ""),
            "requested_capability": specification.metadata.get("requested_capability", ""),
        },
    )
    validator_signal = detect_validator_capture(structure)
    validator_decision = decide_validator_policy(structure, validator_signal)
    validator_tier = validator_overlay_tier(validator_signal)

    proxy_assessment = assess_proxy_objective(
        specification,
        base_policy_evidence=base_policy_evidence,
        optimized_policy_evidence=optimized_policy_evidence,
    )
    novelty_assessment = assess_objective_novelty(
        specification,
        observed_environment=observed_environment,
    )
    posterior = infer_objective_posterior(specification, proxy_assessment, novelty_assessment)
    robust_decision = decide_robust_objective_policy(
        specification,
        proxy_assessment,
        novelty_assessment,
        posterior,
    )
    interrupt = objective_validation_interrupt(
        specification,
        proxy_assessment,
        novelty_assessment,
        robust_decision,
    )

    proxy_tier = proxy_overlay_tier(proxy_assessment)
    reasons = [
        *validator_signal.reasons,
        *proxy_assessment.reasons,
        *novelty_assessment.reasons,
        robust_decision.rationale_summary,
    ]
    if specification.metadata.get("memory_modified_objective"):
        reasons.append(
            "Untrusted memory attempted to modify objective; memory boundary review required"
        )
    if specification.imperatives_relevant and specification.known_tradeoffs:
        reasons.append("Value decomposition review required for local imperative domination risk")

    report = ProxyBreakdownReport(
        objective_id=specification.objective_id,
        proxy_breakdown_detected=proxy_assessment.correlation_breakdown_detected,
        novelty_detected=novelty_assessment.novel_state_detected,
        distribution_shift_detected=novelty_assessment.distribution_shift_detected,
        optimization_pressure=proxy_assessment.optimization_pressure,
        correlation_warning=(
            "Proxy may no longer correlate with intended outcome under current optimization."
            if proxy_assessment.correlation_breakdown_detected
            else None
        ),
        objective_posterior_reference=f"{specification.objective_id}:posterior",
        robust_decision_reference=f"{specification.objective_id}:robust-decision",
        interrupt_required=bool(interrupt),
        review_required=bool(
            proxy_assessment.review_required
            or novelty_assessment.review_required
            or posterior.review_required
            or robust_decision.review_required
            or validator_decision.action in {"refuse", "escalate"}
        ),
        reasons=reasons,
        metadata={
            "validator_overlay_tier": validator_tier,
            "proxy_overlay_tier": proxy_tier,
            "validator_capture_assessment": validator_signal.to_dict(),
            "validator_policy_decision": validator_decision.to_dict(),
            "proxy_objective_assessment": proxy_assessment.to_dict(),
            "novelty_assessment": novelty_assessment.to_dict(),
            "objective_posterior": posterior.to_dict(),
            "robust_objective_decision": robust_decision.to_dict(),
            "objective_validation_interrupt": interrupt.to_dict() if interrupt else None,
            "trace_events": objective_trace_events(
                specification,
                validator_signal.to_dict(),
                proxy_assessment.to_dict(),
                novelty_assessment.to_dict(),
                posterior.to_dict(),
                robust_decision.to_dict(),
                None,
                interrupt.to_dict() if interrupt else None,
            ),
            "semantic_assessment_reference": specification.metadata.get(
                "semantic_assessment_reference"
            ),
            "memory_boundary_reference": specification.metadata.get("memory_boundary_reference"),
            "value_decomposition_reference": specification.metadata.get(
                "value_decomposition_reference"
            ),
            "deception_fingerprint_reference": specification.metadata.get(
                "deception_fingerprint_reference"
            ),
            "attribution_graph_reference": specification.metadata.get(
                "attribution_graph_reference"
            ),
        },
    )
    trace_events = report.metadata["trace_events"]
    if isinstance(trace_events, list):
        for event in trace_events:
            if event.get("event_type") == "proxy_breakdown_report":
                event["payload"] = report.to_dict()
                break
    return report


def objective_trace_events(
    specification: ObjectiveSpecification,
    validator_capture_assessment: dict[str, Any],
    proxy_objective_assessment: dict[str, Any],
    novelty_assessment: dict[str, Any],
    objective_posterior: dict[str, Any],
    robust_objective_decision: dict[str, Any],
    proxy_breakdown_report: dict[str, Any] | None,
    objective_validation_interrupt: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return JSON-friendly trace event payloads for structured log envelopes."""

    rows = [
        ("objective_specification", specification.to_dict()),
        ("validator_capture_assessment", validator_capture_assessment),
        ("proxy_objective_assessment", proxy_objective_assessment),
        ("novelty_assessment", novelty_assessment),
        ("objective_posterior_update", objective_posterior),
        ("robust_objective_decision", robust_objective_decision),
        ("proxy_breakdown_report", proxy_breakdown_report or {}),
    ]
    if objective_validation_interrupt is not None:
        rows.append(("objective_validation_interrupt", objective_validation_interrupt))
    return [
        {
            "event_type": event_type,
            "objective_id": specification.objective_id,
            "payload": payload,
        }
        for event_type, payload in rows
    ]
