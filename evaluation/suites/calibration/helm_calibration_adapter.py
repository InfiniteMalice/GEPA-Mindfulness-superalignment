"""HELM-style calibration adapter over local prompt/gold rows.

This adapter normalizes local rows for response-file scoring, while metric
aggregation lives in ``brier_score.py`` and ``selective_answering.py``.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "helm_calibration"
CATEGORY = "calibration"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
