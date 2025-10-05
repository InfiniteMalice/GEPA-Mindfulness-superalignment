"""Aggregators for GEPA mindfulness practice sessions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import isfinite
from numbers import Real
from typing import Iterable, Iterator, Tuple, Union

_DECIMAL_ZERO = Decimal("0")
_DECIMAL_ONE = Decimal("1")

NumberLike = Union[Real, Decimal, Fraction]


def _ensure_numeric(label: str, value: object) -> None:
    """Ensure *value* behaves like a real number.

    ``bool`` instances and ``str``/``bytes`` values are rejected explicitly so
    they do not silently coerce to integers or floats.
    """

    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number")

    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a real number")

    if isinstance(value, (Real, Decimal, Fraction)):
        return

    try:
        float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{label} must be a real number") from exc


def _to_decimal(label: str, value: NumberLike | object) -> Decimal:
    """Coerce *value* into :class:`~decimal.Decimal` with sanity checks."""

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{label} must be finite")
        if value != 0 and value.adjusted() > 308:
            # Values outside the IEEE-754 float exponent range overflow when
            # converted to ``float``.  Treat them as non-finite for the API.
            raise ValueError(f"{label} must be finite")
        return value

    if isinstance(value, Fraction):
        if value.denominator == 0:
            raise ValueError(f"{label} must be finite")
        return Decimal(value.numerator) / Decimal(value.denominator)

    if isinstance(value, Real):
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{label} must be finite")
        return Decimal(str(numeric))

    # Fallback for float-like objects that are not ``Real`` (e.g. numpy
    # scalar-like wrappers used in the tests).
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{label} must be a real number") from exc

    if not isfinite(numeric):
        raise ValueError(f"{label} must be finite")

    return Decimal(str(numeric))


def _decimal_to_float(label: str, value: Decimal) -> float:
    """Convert *value* to float while guarding against under/overflow."""

    try:
        result = float(value)
    except (OverflowError, InvalidOperation) as exc:
        raise ValueError(f"{label} is too large") from exc

    if not isfinite(result):
        raise ValueError(f"{label} is too large")

    if result == 0.0 and value != 0:
        if label == "duration_minutes":
            raise ValueError("duration_minutes is too small; total duration is too small")
        raise ValueError(f"{label} is too small")

    return result


def _coerce_duration(value: object) -> Decimal:
    _ensure_numeric("duration_minutes", value)
    duration = _to_decimal("duration_minutes", value)
    if duration < _DECIMAL_ZERO:
        raise ValueError("duration_minutes must be non-negative")
    return duration


def _coerce_score(label: str, value: object) -> Decimal:
    _ensure_numeric(label, value)

    if isinstance(value, Fraction):
        if value.denominator == 0:
            raise ValueError(f"{label} must be finite")
        if value < 0 or value > 1:
            raise ValueError(f"{label} must be within [0.0, 1.0]")

    score = _to_decimal(label, value)
    if score < _DECIMAL_ZERO or score > _DECIMAL_ONE:
        raise ValueError(f"{label} must be within [0.0, 1.0]")
    return score


@dataclass(frozen=True)
class PracticeSession:
    """Container describing a mindfulness practice session."""

    duration_minutes: NumberLike | object
    grounding: NumberLike | object
    equanimity: NumberLike | object
    purpose: NumberLike | object
    awareness: NumberLike | object

    def validate(self) -> None:
        """Ensure the session contains sane numeric values."""

        duration = _coerce_duration(self.duration_minutes)
        # Underflow/overflow detection for duration happens here so callers get
        # a precise error message even before aggregation.
        _decimal_to_float("duration_minutes", duration)

        for label, value in self._iter_scores():
            score = _coerce_score(label, value)
            _decimal_to_float(label, score)

    def _iter_scores(self) -> Iterator[Tuple[str, object]]:
        yield "grounding", self.grounding
        yield "equanimity", self.equanimity
        yield "purpose", self.purpose
        yield "awareness", self.awareness


@dataclass(frozen=True)
class AggregateResult:
    """Weighted aggregate of GEPA metrics."""

    total_duration: float
    grounding: float
    equanimity: float
    purpose: float
    awareness: float

    @property
    def gepa(self) -> float:
        return (self.grounding + self.equanimity + self.purpose + self.awareness) / 4.0


def aggregate_gepa_metrics(sessions: Iterable[PracticeSession]) -> AggregateResult:
    """Compute a duration-weighted aggregate GEPA score."""

    total_duration = Decimal("0")
    grounding_total = Decimal("0")
    equanimity_total = Decimal("0")
    purpose_total = Decimal("0")
    awareness_total = Decimal("0")

    for session in sessions:
        session.validate()
        duration = _coerce_duration(session.duration_minutes)
        if duration == 0:
            continue

        total_duration += duration
        grounding_total += _coerce_score("grounding", session.grounding) * duration
        equanimity_total += _coerce_score("equanimity", session.equanimity) * duration
        purpose_total += _coerce_score("purpose", session.purpose) * duration
        awareness_total += _coerce_score("awareness", session.awareness) * duration

    if total_duration == 0:
        return AggregateResult(0.0, 0.0, 0.0, 0.0, 0.0)

    total_duration_float = _decimal_to_float("total duration", total_duration)
    grounding_avg = grounding_total / total_duration
    equanimity_avg = equanimity_total / total_duration
    purpose_avg = purpose_total / total_duration
    awareness_avg = awareness_total / total_duration

    return AggregateResult(
        total_duration=total_duration_float,
        grounding=_decimal_to_float("grounding", grounding_avg),
        equanimity=_decimal_to_float("equanimity", equanimity_avg),
        purpose=_decimal_to_float("purpose", purpose_avg),
        awareness=_decimal_to_float("awareness", awareness_avg),
    )


def aggregate_gepa_score(sessions: Iterable[PracticeSession]) -> float:
    """Return only the overall GEPA score."""

    return aggregate_gepa_metrics(sessions).gepa
