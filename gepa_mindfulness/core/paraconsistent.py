"""Minimal paraconsistent logic primitives."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParaconsistentTruthValue:
    """Represents truth in a paraconsistent system with support/opposition."""

    support: float
    opposition: float

    @classmethod
    def from_support_opposition(cls, support: float, opposition: float) -> "ParaconsistentTruthValue":
        if not 0.0 <= support <= 1.0:
            raise ValueError("support must be within [0, 1]")
        if not 0.0 <= opposition <= 1.0:
            raise ValueError("opposition must be within [0, 1]")
        return cls(support=support, opposition=opposition)

    @property
    def truthiness(self) -> float:
        return self.support * (1 - self.opposition)

    @property
    def falsity(self) -> float:
        return self.opposition * (1 - self.support)

    @property
    def contradiction_level(self) -> float:
        return self.support * self.opposition

    def resolve(self, tolerance: float = 0.1) -> float:
        """Resolve to a scalar signal while respecting contradictions."""

        if self.contradiction_level > tolerance:
            # degrade confidence if contradiction high
            return (self.support - self.opposition) * (1 - self.contradiction_level)
        return self.support - self.opposition


def dialetheic_and(left: ParaconsistentTruthValue, right: ParaconsistentTruthValue) -> ParaconsistentTruthValue:
    """Combine two paraconsistent truth values using a cautious conjunction."""

    support = min(left.support, right.support)
    opposition = max(left.opposition, right.opposition)
    return ParaconsistentTruthValue.from_support_opposition(support, opposition)
