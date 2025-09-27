"""Scoring helpers for GEPA mindfulness sessions.

The module provides a :class:`PracticeSession` data class that represents a

single training session alongside :func:`aggregate_gepa_metrics` and
``aggregate_gepa_score`` helpers that combine several sessions into weighted
averages.

A previous bug caused aggregation to raise ``ZeroDivisionError`` when all
sessions had a duration of zero minutes.  This can easily happen in practice
when users record preparatory notes without starting the actual timer.  The
aggregators guard against this situation by returning zeroed metrics whenever
there is no time information to average over.
=======


"""

from __future__ import annotations

from dataclasses import dataclass

from decimal import Decimal
from math import isfinite
from numbers import Real, Rational
from typing import Iterable


def _ensure_real_number(label: str, value: float) -> None:
    """Raise ``TypeError`` when *value* is not a real number."""

    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise TypeError(f"{label} must be a real number")


def _coerce_finite_float(label: str, value: Real | Decimal) -> float:
    """Return ``value`` as ``float`` while ensuring it is finite."""

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{label} must be finite")
        return float(value)

    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc

    if not isfinite(numeric):
        raise ValueError(f"{label} must be finite")

    return numeric


def _to_decimal(value: Real | Decimal) -> Decimal:
    """Convert ``value`` to :class:`~decimal.Decimal` without precision loss."""

    if isinstance(value, Decimal):
        return value

    if isinstance(value, Rational):
        return Decimal(value.numerator) / Decimal(value.denominator)

    return Decimal(str(value))


def _decimal_to_float(label: str, value: Decimal) -> float:
    """Convert *value* to ``float`` and ensure it remains finite and non-zero."""

    result = float(value)
    if result == 0.0 and value != 0:
        raise ValueError(f"{label} is too small to represent as a finite float")
    if not isfinite(result):
        raise ValueError(f"{label} is too large to represent as a finite float")
    return result

=======

@dataclass(frozen=True)
class PracticeSession:
    """Container describing a mindfulness practice session.

    Attributes
    ----------
    duration_minutes:
        Length of the practice session.  Must be non-negative.
    grounding:
        Score in ``[0, 1]`` capturing how grounded the practitioner felt.
    equanimity:
        Score in ``[0, 1]`` capturing the level of equanimity.
    purpose:
        Score in ``[0, 1]`` capturing clarity of purpose or intention.
    awareness:
        Score in ``[0, 1]`` capturing mindfulness awareness.
    """

    duration_minutes: float
    grounding: float
    equanimity: float
    purpose: float
    awareness: float

    def validate(self) -> None:
        """Ensure the session data lives within the supported domain."""


        _ensure_real_number("duration_minutes", self.duration_minutes)
        _coerce_finite_float("duration_minutes", self.duration_minutes)
        if self.duration_minutes < 0:
            raise ValueError("duration_minutes must be non-negative")

=======
        for label, value in (
            ("grounding", self.grounding),
            ("equanimity", self.equanimity),
            ("purpose", self.purpose),
            ("awareness", self.awareness),
        ):

            _ensure_real_number(label, value)
            numeric_value = _coerce_finite_float(label, value)
            if not 0.0 <= numeric_value <= 1.0:
                raise ValueError(f"{label} must be within [0.0, 1.0]")


@dataclass(frozen=True)
class AggregateResult:
    """Weighted aggregate of GEPA metrics.

    The per-axis values represent duration-weighted averages.  ``gepa`` is the
    overall mean of those axes and ``total_duration`` is the sum of the
    contributing durations.
    """

    total_duration: float
    grounding: float
    equanimity: float
    purpose: float
    awareness: float

    @property
    def gepa(self) -> float:
        """Return the grand mean across the four GEPA axes."""

        return (self.grounding + self.equanimity + self.purpose + self.awareness) / 4.0


def aggregate_gepa_metrics(sessions: Iterable[PracticeSession]) -> AggregateResult:
=======

            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be within [0.0, 1.0]")

    """Compute a weighted aggregate GEPA score for several sessions.

    The function averages the per-session GEPA scores weighted by their duration.
    The score per session is a simple arithmetic mean over the four GEPA axes.

    Parameters
    ----------
    sessions:
        Iterable of :class:`PracticeSession` objects.

    Returns
    -------

    AggregateResult
        Duration-weighted averages across the four GEPA axes and the combined
        GEPA score.  All values are zero when there is no positive-duration data
        which avoids the previously existing division-by-zero bug.
    """

    total_duration = Decimal("0")
    grounding_total = Decimal("0")
    equanimity_total = Decimal("0")
    purpose_total = Decimal("0")
    awareness_total = Decimal("0")
=======



    for session in sessions:
        session.validate()
        
        weight = _to_decimal(session.duration_minutes)

        if weight == 0:
=======
            # Zero-duration sessions provide qualitative signal without affecting
            # the quantitative average.  They are ignored but still validated.
            continue


        total_duration += weight
        grounding_total += _to_decimal(session.grounding) * weight
        equanimity_total += _to_decimal(session.equanimity) * weight
        purpose_total += _to_decimal(session.purpose) * weight
        awareness_total += _to_decimal(session.awareness) * weight

    if total_duration == 0:
        return AggregateResult(
            total_duration=0.0,
            grounding=0.0,
            equanimity=0.0,
            purpose=0.0,
            awareness=0.0,
        )

    grounding_avg = grounding_total / total_duration
    equanimity_avg = equanimity_total / total_duration
    purpose_avg = purpose_total / total_duration
    awareness_avg = awareness_total / total_duration

    return AggregateResult(
        total_duration=_decimal_to_float("total duration", total_duration),
        grounding=_decimal_to_float("grounding", grounding_avg),
        equanimity=_decimal_to_float("equanimity", equanimity_avg),
        purpose=_decimal_to_float("purpose", purpose_avg),
        awareness=_decimal_to_float("awareness", awareness_avg),
    )


def aggregate_gepa_score(sessions: Iterable[PracticeSession]) -> float:
    """Return only the overall GEPA score for convenience."""

    return aggregate_gepa_metrics(sessions).gepa
=======
