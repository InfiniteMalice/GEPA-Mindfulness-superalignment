"""Evaluation utilities for GEPA Mindfulness."""

from gepa_mindfulness.evaluation.context_classifier import ContextClassifier

__all__ = ["ContextClassifier"]

try:  # pragma: no cover - executed when optional deps available
    from gepa_mindfulness.evaluation import baseline_evaluator as _baseline
except ModuleNotFoundError as error:  # pragma: no cover - skip when torch missing
    if error.name != "torch":
        raise
else:
    BaselineEvaluator = _baseline.BaselineEvaluator
    EvaluationExample = _baseline.EvaluationExample
    EvaluationResult = _baseline.EvaluationResult
    load_evaluation_dataset = _baseline.load_evaluation_dataset
    __all__.extend(
        [
            "BaselineEvaluator",
            "EvaluationExample",
            "EvaluationResult",
            "load_evaluation_dataset",
        ]
    )
