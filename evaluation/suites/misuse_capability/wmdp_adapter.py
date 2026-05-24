"""WMDP adapter for hazardous-capability boundary evaluation.

WMDP data must be supplied locally. This adapter records benchmark outcomes
without adding sensitive benchmark content to the repository.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "wmdp"
CATEGORY = "misuse_capability"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
