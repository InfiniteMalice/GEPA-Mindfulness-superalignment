"""Abstention curve helpers for confidence-threshold sweeps."""

from __future__ import annotations

from evaluation.schema import EvalResult
from evaluation.suites.calibration.brier_score import selective_accuracy


def abstention_curve(
    results: list[EvalResult], thresholds: list[float] | None = None
) -> list[dict[str, float | None]]:
    if thresholds is None:
        thresholds = [0.0, 0.25, 0.5, 0.75, 0.9]
    rows: list[dict[str, float | None]] = []
    for threshold in thresholds:
        covered = [
            result
            for result in results
            if result.confidence is not None and float(result.confidence) >= threshold
        ]
        rows.append(
            {
                "threshold": threshold,
                "coverage": len(covered) / len(results) if results else 0.0,
                "selective_accuracy": selective_accuracy(results, threshold=threshold),
            }
        )
    return rows
