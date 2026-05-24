import pytest

from evaluation.schema import EvalResult
from evaluation.suites.calibration.abstention_curve import abstention_curve
from evaluation.suites.calibration.brier_score import (
    abstention_appropriateness,
    accuracy,
    brier_score,
    expected_calibration_error,
    selective_accuracy,
    summarize_calibration,
)


def _results() -> list[EvalResult]:
    return [
        EvalResult("a", "simpleqa", "factuality", "p", "x", "x", "correct", confidence=0.9),
        EvalResult("b", "simpleqa", "factuality", "p", "y", "x", "incorrect", confidence=0.8),
        EvalResult(
            "c",
            "truthfulqa",
            "factuality",
            "p",
            "I do not know",
            "x",
            "abstained",
            confidence=0.6,
            trace_flags={"abstention_appropriate": True},
        ),
    ]


def test_calibration_metrics_are_stable() -> None:
    results = _results()

    assert accuracy(results) == pytest.approx(1 / 3)
    assert brier_score(results) == pytest.approx((0.1**2 + 0.8**2 + 0.6**2) / 3)
    assert expected_calibration_error(results, bins=2) == pytest.approx(0.4333333333)
    assert selective_accuracy(results, threshold=0.85) == 1.0
    assert abstention_appropriateness(results) == 1.0
    assert summarize_calibration(results)["incorrect_answer_rate"] == pytest.approx(1 / 3)


def test_calibration_helpers_validate_edge_cases() -> None:
    results = _results()

    assert abstention_curve(results, thresholds=[]) == []
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(results, bins=0)
