"""Scoring helpers for GEPA mindfulness sessions.

The module provides a :class:`PracticeSession` data class that represents a
single training session alongside :func:`aggregate_gepa_score` that aggregates
several sessions into a single scalar reward.

A bug previously caused :func:`aggregate_gepa_score` to raise ``ZeroDivisionError``
when all sessions had a duration of zero minutes.  This can easily happen in
practice when users record preparatory notes without starting the actual timer.
The fix guards against this situation by returning ``0.0`` for the aggregate
score whenever there is no time information to average over.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


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

        if self.duration_minutes < 0:
            raise ValueError("duration_minutes must be non-negative")

        if not isfinite(self.duration_minutes):
            raise ValueError("duration_minutes must be finite")

        for label, value in (
            ("grounding", self.grounding),
            ("equanimity", self.equanimity),
            ("purpose", self.purpose),
            ("awareness", self.awareness),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be within [0.0, 1.0]")


def aggregate_gepa_score(sessions: Iterable[PracticeSession]) -> float:
    """Compute a weighted aggregate GEPA score for several sessions.

    The function averages the per-session GEPA scores weighted by their duration.
    The score per session is a simple arithmetic mean over the four GEPA axes.

    Parameters
    ----------
    sessions:
        Iterable of :class:`PracticeSession` objects.

    Returns
    -------
    float
        Weighted average GEPA score.  Returns ``0.0`` if all sessions are zero
        length which avoids the previously existing division-by-zero bug.
    """

    total_duration = 0.0
    weighted_sum = 0.0

    for session in sessions:
        session.validate()

        if session.duration_minutes == 0:
            # Zero-duration sessions provide qualitative signal without affecting
            # the quantitative average.  They are ignored but still validated.
            continue

        gepa_value = (
            session.grounding
            + session.equanimity
            + session.purpose
            + session.awareness
        ) / 4.0

        total_duration += session.duration_minutes
        weighted_sum += gepa_value * session.duration_minutes

    if total_duration == 0:
        return 0.0

    return weighted_sum / total_duration
