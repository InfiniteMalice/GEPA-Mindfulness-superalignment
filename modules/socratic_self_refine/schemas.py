"""Typed records for SSR-inspired reasoning-unit verification and repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReasoningUnit:
    unit_id: str
    parent_unit_id: str | None
    sequence_index: int
    sub_question: str
    sub_answer: str
    evidence_summary: str
    assumptions: tuple[str, ...]
    uncertainty_markers: tuple[str, ...]
    confidence: float
    verifier_status: str
    repair_status: str
    dependencies: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "uncertainty_markers", tuple(self.uncertainty_markers))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledResolveResult:
    unit_id: str
    attempt_id: str
    resolved_answer: str
    confidence: float
    agrees_with_original: bool
    verifier_status: str
    evidence_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningUnitAssessment:
    unit_id: str
    original_confidence: float
    resolve_attempts: tuple[ControlledResolveResult, ...]
    self_consistency_score: float
    dependency_risk: float
    repair_recommended: bool
    repair_reason: str
    review_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolve_attempts", tuple(self.resolve_attempts))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolve_attempts"] = [attempt.to_dict() for attempt in self.resolve_attempts]
        return payload


@dataclass(frozen=True)
class RepairEvent:
    repair_id: str
    unit_id: str
    before_summary: str
    after_summary: str
    repair_reason: str
    confidence_before: float
    confidence_after: float
    dependency_updates: tuple[str, ...]
    review_status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependency_updates", tuple(self.dependency_updates))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SSRRunReport:
    run_id: str
    reasoning_unit_count: int
    units_assessed: int
    units_repaired: int
    max_iterations: int
    stopped_reason: str
    initial_answer_reference: str
    refined_answer_reference: str
    repair_events: tuple[RepairEvent, ...]
    review_required: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repair_events", tuple(self.repair_events))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repair_events"] = [event.to_dict() for event in self.repair_events]
        return payload


__all__ = [
    "ControlledResolveResult",
    "ReasoningUnit",
    "ReasoningUnitAssessment",
    "RepairEvent",
    "SSRRunReport",
]
