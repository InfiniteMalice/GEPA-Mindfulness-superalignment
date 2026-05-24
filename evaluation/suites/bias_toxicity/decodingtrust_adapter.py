"""DecodingTrust adapter for trustworthiness dimensions.

Use local DecodingTrust-compatible rows and preserve subtask labels in metadata.
Heavy dependencies and full datasets remain external and optional.
"""

from __future__ import annotations

from evaluation.schema import EvalCase, EvalResult
from evaluation.suites.common import load_jsonl_cases, score_response

SUITE = "decodingtrust"
CATEGORY = "bias_toxicity"


def load_examples(input_path: str | None, limit: int | None = None) -> list[EvalCase]:
    return load_jsonl_cases(input_path, suite=SUITE, category=CATEGORY, limit=limit)


def score_example(case: EvalCase, model_answer: str, confidence: float | None = None) -> EvalResult:
    return score_response(case, model_answer, confidence=confidence)
