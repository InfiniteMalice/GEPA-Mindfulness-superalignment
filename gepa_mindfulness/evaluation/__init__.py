"""Evaluation utilities for GEPA Mindfulness."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gepa_mindfulness.evaluation.baseline_evaluator import (
    BaselineEvaluator,
    EvaluationExample,
    EvaluationResult,
    load_evaluation_dataset,
)
from gepa_mindfulness.evaluation.context_classifier import ContextClassifier
from mindful_trace_gepa.import_utils import optional_repo_module

if TYPE_CHECKING:
    from semantic_intent_robustness.evaluators import (
        SemanticRobustnessEvaluator as _SemanticRobustnessEvaluator,
    )

MODULES_PATH = Path(__file__).resolve().parents[2] / "modules"


try:
    _semantic_evaluator_module = optional_repo_module(
        "semantic_intent_robustness.evaluators",
        MODULES_PATH,
        allow_missing_submodules=False,
        allow_import_error_name_match=True,
    )
except ModuleNotFoundError as exc:
    if (exc.name or "").startswith("semantic_intent_robustness"):
        _semantic_evaluator_module = None
    else:
        raise
SemanticRobustnessEvaluator: type["_SemanticRobustnessEvaluator"] | None = (
    getattr(_semantic_evaluator_module, "SemanticRobustnessEvaluator", None)
    if _semantic_evaluator_module
    else None
)

__all__ = [
    "BaselineEvaluator",
    "ContextClassifier",
    "EvaluationExample",
    "EvaluationResult",
    "load_evaluation_dataset",
    "SemanticRobustnessEvaluator",
]
