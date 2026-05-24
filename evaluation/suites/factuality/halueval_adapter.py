"""HaluEval-style adapter for hallucination and unsupported-claim checks.

Use local HaluEval exports or equivalent hallucination benchmark rows. Expected
metadata can include ``trace_flags.unsupported_claim`` or ``expected_outcome``.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "halueval"
CATEGORY = "factuality"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
