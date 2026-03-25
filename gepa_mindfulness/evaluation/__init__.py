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
    from semantic_intent_robustness import (
        SemanticRobustnessEvaluator as _SemanticRobustnessEvaluator,
    )

MODULES_PATH = Path(__file__).resolve().parents[2] / "modules"


_semantic_pkg = optional_repo_module(
    "semantic_intent_robustness",
    MODULES_PATH,
    allow_missing_submodules=True,
    allow_import_error_name_match=True,
)
SemanticRobustnessEvaluator: type["_SemanticRobustnessEvaluator"] | None = (
    getattr(_semantic_pkg, "SemanticRobustnessEvaluator", None) if _semantic_pkg else None
)

if TYPE_CHECKING:
    from semantic_intent_robustness import (
        SemanticRobustnessEvaluator as _SemanticRobustnessEvaluator,
    )

MODULES_PATH = Path(__file__).resolve().parents[2] / "modules"


def _optional_repo_module(name: str) -> Any | None:
    inserted_path = str(MODULES_PATH)
    added_path = False
    if MODULES_PATH.is_dir() and inserted_path not in sys.path:
        sys.path.insert(0, inserted_path)
        added_path = True
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name or (exc.name and exc.name.startswith(f"{name}.")):
            return None
        raise
    except ImportError as exc:
        if name in str(exc):
            return None
        raise
    finally:
        if added_path and inserted_path in sys.path:
            sys.path.remove(inserted_path)


_semantic_pkg = _optional_repo_module("semantic_intent_robustness")
SemanticRobustnessEvaluator: type["_SemanticRobustnessEvaluator"] | None = (
    getattr(_semantic_pkg, "SemanticRobustnessEvaluator", None) if _semantic_pkg else None
)

__all__ = [
    "BaselineEvaluator",
    "ContextClassifier",
    "EvaluationExample",
    "EvaluationResult",
    "load_evaluation_dataset",
    "SemanticRobustnessEvaluator",
]
