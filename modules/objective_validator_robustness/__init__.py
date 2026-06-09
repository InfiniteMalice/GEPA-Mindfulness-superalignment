"""Objective / Validator robustness package for Validator Capture detection."""

# Standard library
from __future__ import annotations

from .config import (
    ObjectiveValidatorRobustnessConfig,
    load_objective_validator_robustness_config,
)
from .decomposition import decompose_objective
from .detection import detect_validator_capture, validator_overlay_tier
from .evals import (
    evaluate_examples,
    evaluate_proxy_examples,
    load_examples,
    summarize_proxy_robustness_results,
    summarize_results,
)
from .inverse_objective import infer_objective_posterior
from .novelty import assess_objective_novelty
from .pipeline import evaluate_objective_robustness, objective_trace_events
from .policy import decide_validator_policy
from .proxy_analysis import assess_proxy_objective, proxy_overlay_tier
from .robust_policy import decide_robust_objective_policy, objective_validation_interrupt
from .schema import (
    NoveltyAssessment,
    ObjectivePosterior,
    ObjectiveProxyMetrics,
    ObjectiveSpecification,
    ObjectiveStructure,
    ObjectiveValidationInterrupt,
    PlausibleObjective,
    ProxyBreakdownReport,
    ProxyObjectiveAssessment,
    RobustObjectiveDecision,
    ValidatorCaptureSignal,
    ValidatorPolicyDecision,
    ValidatorRobustnessScore,
)
from .scoring import score_validator_robustness

__all__ = [
    "ObjectiveStructure",
    "ObjectiveValidatorRobustnessConfig",
    "ObjectiveSpecification",
    "ProxyObjectiveAssessment",
    "NoveltyAssessment",
    "PlausibleObjective",
    "ObjectivePosterior",
    "RobustObjectiveDecision",
    "ProxyBreakdownReport",
    "ObjectiveValidationInterrupt",
    "ObjectiveProxyMetrics",
    "ValidatorCaptureSignal",
    "ValidatorPolicyDecision",
    "ValidatorRobustnessScore",
    "decompose_objective",
    "detect_validator_capture",
    "validator_overlay_tier",
    "assess_proxy_objective",
    "proxy_overlay_tier",
    "assess_objective_novelty",
    "infer_objective_posterior",
    "decide_robust_objective_policy",
    "objective_validation_interrupt",
    "evaluate_objective_robustness",
    "objective_trace_events",
    "decide_validator_policy",
    "score_validator_robustness",
    "load_examples",
    "evaluate_examples",
    "evaluate_proxy_examples",
    "summarize_results",
    "summarize_proxy_robustness_results",
    "load_objective_validator_robustness_config",
]
