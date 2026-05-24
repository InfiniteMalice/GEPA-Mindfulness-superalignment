"""Synthetic PII leakage probe adapter.

Fixtures must use fake data only. Real user secrets should never be committed or
used in CI.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "pii_leakage"
CATEGORY = "privacy"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
