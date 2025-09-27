"""Scoring helpers for GEPA mindfulness sessions.

The module provides a :class:`PracticeSession` data class that represents a

"""

from __future__ import annotations

from dataclasses import dataclass

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

        for label, value in (
            ("grounding", self.grounding),
            ("equanimity", self.equanimity),
            ("purpose", self.purpose),
            ("awareness", self.awareness),
        ):

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


    for session in sessions:
        session.validate()

            # Zero-duration sessions provide qualitative signal without affecting
            # the quantitative average.  They are ignored but still validated.
            continue


