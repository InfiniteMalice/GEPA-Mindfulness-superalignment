"""Objective / Validator robustness package for Validator Capture detection."""

# Standard library
from __future__ import annotations

from .decomposition import decompose_objective
from .detection import detect_validator_capture, validator_overlay_tier
from .evals import evaluate_examples, load_examples, summarize_results
from .policy import decide_validator_policy
from .schema import (
    ObjectiveStructure,
    ValidatorCaptureSignal,
    ValidatorPolicyDecision,
    ValidatorRobustnessScore,
)
from .scoring import score_validator_robustness

__all__ = [
    "ObjectiveStructure",
    "ValidatorCaptureSignal",
    "ValidatorPolicyDecision",
    "ValidatorRobustnessScore",
    "decompose_objective",
    "detect_validator_capture",
    "validator_overlay_tier",
    "decide_validator_policy",
    "score_validator_robustness",
    "load_examples",
    "evaluate_examples",
    "summarize_results",
]
