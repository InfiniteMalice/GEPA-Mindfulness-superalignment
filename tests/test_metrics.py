import math
from decimal import Decimal
from fractions import Fraction

import pytest

from gepa_mindfulness import (
    AggregateResult,
    PracticeSession,
    aggregate_gepa_metrics,
    aggregate_gepa_score,
)


def test_aggregate_gepa_score_basic_weighting():
    sessions = [
        PracticeSession(duration_minutes=30, grounding=0.6, equanimity=0.5, purpose=0.8, awareness=0.7),
        PracticeSession(duration_minutes=15, grounding=0.9, equanimity=0.8, purpose=0.9, awareness=0.85),
    ]

    # expected value computed manually
    expected = (
        ((0.6 + 0.5 + 0.8 + 0.7) / 4.0) * 30
        + ((0.9 + 0.8 + 0.9 + 0.85) / 4.0) * 15
    ) / 45

    assert math.isclose(aggregate_gepa_score(sessions), expected)


def test_aggregate_gepa_metrics_reports_per_axis():
    sessions = [
        PracticeSession(duration_minutes=10, grounding=0.2, equanimity=0.4, purpose=0.6, awareness=0.8),
        PracticeSession(duration_minutes=20, grounding=0.5, equanimity=0.6, purpose=0.7, awareness=0.4),
    ]

    result = aggregate_gepa_metrics(sessions)

    expected_grounding = (0.2 * 10 + 0.5 * 20) / 30
    expected_equanimity = (0.4 * 10 + 0.6 * 20) / 30
    expected_purpose = (0.6 * 10 + 0.7 * 20) / 30
    expected_awareness = (0.8 * 10 + 0.4 * 20) / 30
    expected_gepa = (expected_grounding + expected_equanimity + expected_purpose + expected_awareness) / 4

    assert isinstance(result, AggregateResult)
    assert math.isclose(result.total_duration, 30)
    assert math.isclose(result.grounding, expected_grounding)
    assert math.isclose(result.equanimity, expected_equanimity)
    assert math.isclose(result.purpose, expected_purpose)
    assert math.isclose(result.awareness, expected_awareness)
    assert math.isclose(aggregate_gepa_score(sessions), result.gepa)
    assert math.isclose(result.gepa, expected_gepa)


def test_zero_duration_sessions_are_ignored():
    sessions = [
        PracticeSession(duration_minutes=0, grounding=0.4, equanimity=0.5, purpose=0.6, awareness=0.7),
        PracticeSession(duration_minutes=0, grounding=0.9, equanimity=0.9, purpose=0.9, awareness=0.9),
    ]

    result = aggregate_gepa_metrics(sessions)

    assert result == AggregateResult(
        total_duration=0.0,
        grounding=0.0,
        equanimity=0.0,
        purpose=0.0,
        awareness=0.0,
    )
    assert aggregate_gepa_score(sessions) == 0.0


def test_validation_rejects_out_of_range_scores():
    session = PracticeSession(duration_minutes=10, grounding=1.5, equanimity=0.5, purpose=0.5, awareness=0.5)

    with pytest.raises(ValueError):
        aggregate_gepa_score([session])


def test_validation_rejects_negative_duration():
    session = PracticeSession(duration_minutes=-1, grounding=0.5, equanimity=0.5, purpose=0.5, awareness=0.5)

    with pytest.raises(ValueError):
        aggregate_gepa_score([session])


def test_validation_rejects_non_finite_duration():
    session = PracticeSession(
        duration_minutes=float("nan"),
        grounding=0.5,
        equanimity=0.5,
        purpose=0.5,
        awareness=0.5,
    )

    with pytest.raises(ValueError):
        aggregate_gepa_score([session])

    session = PracticeSession(
        duration_minutes=float("inf"),
        grounding=0.5,
        equanimity=0.5,
        purpose=0.5,
        awareness=0.5,
    )

    with pytest.raises(ValueError):
        aggregate_gepa_score([session])


def test_validation_rejects_non_numeric_inputs():
    with pytest.raises(TypeError, match="duration_minutes must be a real number"):
        aggregate_gepa_score(
            [
                PracticeSession(
                    duration_minutes="ten",
                    grounding=0.5,
                    equanimity=0.5,
                    purpose=0.5,
                    awareness=0.5,
                )
            ]
        )

    with pytest.raises(TypeError, match="grounding must be a real number"):
        aggregate_gepa_score(
            [
                PracticeSession(
                    duration_minutes=10,
                    grounding=None,
                    equanimity=0.5,
                    purpose=0.5,
                    awareness=0.5,
                )
            ]
        )

    with pytest.raises(TypeError, match="awareness must be a real number"):
        aggregate_gepa_score(
            [
                PracticeSession(
                    duration_minutes=10,
                    grounding=0.5,
                    equanimity=0.5,
                    purpose=0.5,
                    awareness=True,
                )
            ]
        )


def test_aggregate_gepa_metrics_supports_decimal_and_fraction_inputs():
    sessions = [
        PracticeSession(
            duration_minutes=Decimal("15"),
            grounding=Fraction(1, 2),
            equanimity=Fraction(2, 3),
            purpose=Decimal("0.8"),
            awareness=Fraction(3, 5),
        ),
        PracticeSession(
            duration_minutes=Fraction(45, 2),
            grounding=Decimal("0.9"),
            equanimity=Fraction(4, 5),
            purpose=Fraction(7, 10),
            awareness=Decimal("0.6"),
        ),
    ]

    result = aggregate_gepa_metrics(sessions)

    first_weight = float(Decimal("15"))
    second_weight = float(Fraction(45, 2))
    total_duration = first_weight + second_weight
    expected_grounding = (
        float(Fraction(1, 2)) * first_weight + float(Decimal("0.9")) * second_weight
    ) / total_duration
    expected_equanimity = (
        float(Fraction(2, 3)) * first_weight + float(Fraction(4, 5)) * second_weight
    ) / total_duration
    expected_purpose = (
        float(Decimal("0.8")) * first_weight + float(Fraction(7, 10)) * second_weight
    ) / total_duration
    expected_awareness = (
        float(Fraction(3, 5)) * first_weight + float(Decimal("0.6")) * second_weight
    ) / total_duration

    assert math.isclose(result.total_duration, total_duration)
    assert math.isclose(result.grounding, expected_grounding)
    assert math.isclose(result.equanimity, expected_equanimity)
    assert math.isclose(result.purpose, expected_purpose)
    assert math.isclose(result.awareness, expected_awareness)
    assert math.isclose(aggregate_gepa_score(sessions), result.gepa)
