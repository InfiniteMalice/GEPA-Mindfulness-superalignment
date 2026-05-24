"""Selective answering summaries for HELM-style calibration checks."""

from __future__ import annotations

from evaluation.schema import EvalResult
from evaluation.suites.calibration.brier_score import summarize_calibration


def summarize_selective_answering(results: list[EvalResult]) -> dict[str, float | None]:
    return summarize_calibration(results)
