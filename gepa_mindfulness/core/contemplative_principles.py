"""Definitions for GEPA contemplative principles."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping


class ContemplativePrinciple(str, Enum):
    """Supported contemplative principles for GEPA scoring."""

    MINDFULNESS = "mindfulness"
    EMPATHY = "empathy"
    PERSPECTIVE = "perspective"
    AGENCY = "agency"


@dataclass(frozen=True)
class GEPAPrincipleScore:
    """Score container for a single principle."""

    value: float
    rationale: str

    def normalized(self) -> float:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Principle scores must lie within [0, 1].")
        return self.value


@dataclass
class GEPAPrinciples:
    """Collection of contemplative principle scores."""

    scores: Dict[ContemplativePrinciple, GEPAPrincipleScore]

    @classmethod
    def from_iterable(
        cls, entries: Iterable[tuple[ContemplativePrinciple, GEPAPrincipleScore]]
    ) -> "GEPAPrinciples":
        return cls(scores=dict(entries))

    def as_mapping(self) -> Mapping[ContemplativePrinciple, GEPAPrincipleScore]:
        return self.scores

    def aggregate(self) -> float:
        if not self.scores:
            raise ValueError("At least one contemplative principle score required.")
        total = 0.0
        for score in self.scores.values():
            total += score.normalized()
        return total / len(self.scores)

    def rationale_summary(self) -> Dict[str, str]:
        return {principle.value: score.rationale for principle, score in self.scores.items()}
