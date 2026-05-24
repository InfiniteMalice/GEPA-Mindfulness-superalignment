"""SimpleQA adapter for short-answer factuality and hallucination checks.

The adapter accepts local JSONL exports with fields such as ``prompt`` or
``question`` plus ``gold_answer`` or ``answer``. It never downloads benchmark
data automatically.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "simpleqa"
CATEGORY = "factuality"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
