"""Scoring helpers for GEPA mindfulness sessions."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import isfinite
from numbers import Real
from typing import Iterable

_DECIMAL_ZERO = Decimal("0")
_DECIMAL_ONE = Decimal("1")


def _ensure_real(label: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal, Fraction)):
        raise TypeError(f"{label} must be a real number")


def _to_decimal(label: str, value: Real | Decimal | Fraction) -> Decimal:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{label} must be finite")
        if value.adjusted() > 308:
            raise ValueError(f"{label} must be finite")
        return value
    if isinstance(value, Fraction):
        if value.denominator == 0:
            raise ValueError(f"{label} must be finite")
        return Decimal(value.numerator) / Decimal(value.denominator)
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    try:
        decimal_value = Decimal(str(numeric))
    except InvalidOperation as exc:  # pragma: no cover - extremely rare
        raise ValueError(f"{label} must be finite") from exc
    return decimal_value


def _decimal_to_float(label: str, value: Decimal) -> float:
    try:
        result = float(value)
    except (OverflowError, InvalidOperation) as exc:
        raise ValueError(f"{label} is too large") from exc
    if not isfinite(result):
        raise ValueError(f"{label} is too large")
    if result == 0.0 and value != 0:
        raise ValueError(f"{label} is too small")
    return result


def _coerce_score(label: str, value: Real | Decimal | Fraction) -> Decimal:
    _ensure_real(label, value)
    if isinstance(value, Fraction):
        if value < 0:
            raise ValueError(f"{label} must be within [0.0, 1.0]")
        if value > 1:
            raise ValueError(f"{label} must be within [0.0, 1.0]")
    score = _to_decimal(label, value)
    if score < _DECIMAL_ZERO:
        raise ValueError(f"{label} must be within [0.0, 1.0]")
    if score > _DECIMAL_ONE:
        raise ValueError(f"{label} must be within [0.0, 1.0]")
    return score


def _coerce_duration(value: Real | Decimal | Fraction) -> Decimal:
    _ensure_real("duration_minutes", value)
    duration = _to_decimal("duration_minutes", value)
    if duration < _DECIMAL_ZERO:
        raise ValueError("duration_minutes must be non-negative")
    return duration


@dataclass(frozen=True)
class PracticeSession:
    duration_minutes: Real | Decimal | Fraction
    grounding: Real | Decimal | Fraction
    equanimity: Real | Decimal | Fraction
    purpose: Real | Decimal | Fraction
    awareness: Real | Decimal | Fraction

    def validate(self) -> None:
        duration = _coerce_duration(self.duration_minutes)
        for label, value in (
            ("grounding", self.grounding),
            ("equanimity", self.equanimity),
            ("purpose", self.purpose),
            ("awareness", self.awareness),
        ):
            score = _coerce_score(label, value)
            if score != 0 and score != _DECIMAL_ONE:
                try:
                    _decimal_to_float(label, score)
                except ValueError as exc:
                    # Only propagate "too small"/"too large" errors; bounds already enforced.
                    raise exc


@dataclass(frozen=True)
class AggregateResult:
    total_duration: float
    grounding: float
    equanimity: float
    purpose: float
    awareness: float

    @property
    def gepa(self) -> float:
        return (self.grounding + self.equanimity + self.purpose + self.awareness) / 4.0


def aggregate_gepa_metrics(sessions: Iterable[PracticeSession]) -> AggregateResult:
    total_duration = Decimal("0")
    grounding_total = Decimal("0")
    equanimity_total = Decimal("0")
    purpose_total = Decimal("0")
    awareness_total = Decimal("0")

    sessions = list(sessions)
    for session in sessions:
        session.validate()
        duration = _coerce_duration(session.duration_minutes)
        if duration == 0:
            continue
        try:
            _decimal_to_float("duration_minutes", duration)
        except ValueError as exc:
            message = str(exc)
            if "too small" in message:
                raise ValueError("duration_minutes is too small; total duration is too small") from exc
            if "too large" in message:
                raise ValueError("duration_minutes is too large; total duration is too large") from exc
            raise
        total_duration += duration
        grounding_total += _coerce_score("grounding", session.grounding) * duration
        equanimity_total += _coerce_score("equanimity", session.equanimity) * duration
        purpose_total += _coerce_score("purpose", session.purpose) * duration
        awareness_total += _coerce_score("awareness", session.awareness) * duration

    if total_duration == 0:
        return AggregateResult(0.0, 0.0, 0.0, 0.0, 0.0)

    try:
        total_duration_float = _decimal_to_float("total duration", total_duration)
    except ValueError as exc:
        message = str(exc)
        if "too small" in message:
            raise ValueError("total duration is too small") from exc
        if "too large" in message:
            raise ValueError("total duration is too large") from exc
        raise
    grounding = _decimal_to_float("grounding", grounding_total / total_duration)
    equanimity = _decimal_to_float("equanimity", equanimity_total / total_duration)
    purpose = _decimal_to_float("purpose", purpose_total / total_duration)
    awareness = _decimal_to_float("awareness", awareness_total / total_duration)

    return AggregateResult(
        total_duration=total_duration_float,
        grounding=grounding,
        equanimity=equanimity,
        purpose=purpose,
        awareness=awareness,
    )


def aggregate_gepa_score(sessions: Iterable[PracticeSession]) -> float:
    result = aggregate_gepa_metrics(sessions)
    return result.gepa
