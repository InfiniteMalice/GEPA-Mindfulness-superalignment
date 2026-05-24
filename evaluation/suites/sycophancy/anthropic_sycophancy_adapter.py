"""Anthropic-style sycophancy adapter.

Rows can encode user-belief pressure and expected truth-preserving behavior via
``expected_outcome`` and ``trace_flags.user_belief_over_truth`` metadata.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "anthropic_sycophancy"
CATEGORY = "sycophancy"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
