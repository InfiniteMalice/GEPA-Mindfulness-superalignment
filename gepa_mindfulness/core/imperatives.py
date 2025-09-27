"""Alignment imperative modeling with paraconsistent evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping

from .paraconsistent import ParaconsistentTruthValue, dialetheic_and


class AlignmentImperative(str, Enum):
    """Imperatives guiding the GEPA training process."""

    REDUCE_SUFFERING = "reduce_suffering"
    INCREASE_PROSPERITY = "increase_prosperity"
    INCREASE_KNOWLEDGE = "increase_knowledge"


@dataclass(frozen=True)
class ImperativeSignal:
    """Signal describing support and opposition for an imperative."""

    support: float
    opposition: float
    rationale: str

    def to_paraconsistent(self) -> ParaconsistentTruthValue:
        return ParaconsistentTruthValue.from_support_opposition(
            support=self.support, opposition=self.opposition
        )


@dataclass
class ImperativeEvaluator:
    """Aggregate signals for imperatives using paraconsistent logic."""

    signals: Dict[AlignmentImperative, ImperativeSignal]

    @classmethod
    def from_iterable(
        cls, entries: Iterable[tuple[AlignmentImperative, ImperativeSignal]]
    ) -> "ImperativeEvaluator":
        return cls(signals=dict(entries))

    def as_mapping(self) -> Mapping[AlignmentImperative, ImperativeSignal]:
        return self.signals

    def aggregate(self) -> ParaconsistentTruthValue:
        if not self.signals:
            raise ValueError("At least one imperative signal required.")
        truth = None
        for signal in self.signals.values():
            value = signal.to_paraconsistent()
            truth = value if truth is None else dialetheic_and(truth, value)
        return truth

    def contradiction_report(self) -> Dict[str, float]:
        return {
            imperative.value: signal.to_paraconsistent().contradiction_level
            for imperative, signal in self.signals.items()
        }

    def rationale_summary(self) -> Dict[str, str]:
        return {imperative.value: signal.rationale for imperative, signal in self.signals.items()}
