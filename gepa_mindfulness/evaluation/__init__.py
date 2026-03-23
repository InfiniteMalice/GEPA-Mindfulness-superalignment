"""Evaluation utilities for GEPA Mindfulness."""

from gepa_mindfulness.evaluation.baseline_evaluator import (
    BaselineEvaluator,
    EvaluationExample,
    EvaluationResult,
    load_evaluation_dataset,
)
from gepa_mindfulness.evaluation.context_classifier import ContextClassifier

try:
    from semantic_intent_robustness import SemanticRobustnessEvaluator
except ImportError:  # pragma: no cover
    SemanticRobustnessEvaluator = None

__all__ = [
    "BaselineEvaluator",
    "ContextClassifier",
    "EvaluationExample",
    "EvaluationResult",
    "load_evaluation_dataset",
    "SemanticRobustnessEvaluator",
]
