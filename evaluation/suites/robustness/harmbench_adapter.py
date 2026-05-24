"""HarmBench adapter for harmful-request refusal robustness.

Use local benchmark rows with safe metadata. The adapter does not include or
download HarmBench data.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "harmbench"
CATEGORY = "robustness"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
