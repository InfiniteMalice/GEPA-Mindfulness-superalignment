"""Evaluation utilities for GEPA Mindfulness."""

from gepa_mindfulness.evaluation.context_classifier import ContextClassifier

__all__ = ["ContextClassifier"]

try:  # pragma: no cover - executed when optional deps available
    from gepa_mindfulness.evaluation.baseline_evaluator import (
        BaselineEvaluator,  # noqa: F401
        EvaluationExample,  # noqa: F401
        EvaluationResult,  # noqa: F401
        load_evaluation_dataset,  # noqa: F401
    )
except ModuleNotFoundError as error:  # pragma: no cover - skip when deps missing
    if error.name not in {"torch", "transformers"}:
        raise
else:
    __all__.extend(
        [
            "BaselineEvaluator",
            "EvaluationExample",
            "EvaluationResult",
            "load_evaluation_dataset",
        ]
    )
