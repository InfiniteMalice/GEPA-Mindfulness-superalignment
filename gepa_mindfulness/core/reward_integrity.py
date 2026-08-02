"""Observable, component-preserving reward-integrity calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

from .evidence import EvidenceReference

COMPONENT_NAMES = (
    "objective_fidelity",
    "feedback_integrity",
    "skill_transfer",
    "reality_contact",
    "exploit_disclosure",
    "long_horizon_agency",
    "benign_creativity",
    "repair_quality",
)


def _validated_component(name: str, value: object) -> float:
    """Return one finite, bounded observable component."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {name} in [-1.0, 1.0].")
    numeric_value = float(value)
    if not isfinite(numeric_value) or not -1.0 <= numeric_value <= 1.0:
        raise ValueError(f"Expected finite {name} in [-1.0, 1.0].")
    return numeric_value


def _validated_observable_references(references: object) -> tuple[EvidenceReference, ...]:
    """Copy an explicit boundary containing only typed observable source kinds."""
    if not isinstance(references, (list, tuple)) or not all(
        isinstance(reference, EvidenceReference) for reference in references
    ):
        raise ValueError("Expected observable references as typed EvidenceReference values.")
    if any(not reference.is_observable for reference in references):
        raise ValueError("Observable evidence requires an observable source kind.")
    return tuple(references)


def _validated_observable_evidence(
    observable_evidence: object,
    observable_references: tuple[EvidenceReference, ...],
) -> Mapping[str, tuple[EvidenceReference, ...]]:
    """Copy component evidence and require every citation to stay in the supplied boundary."""
    if not isinstance(observable_evidence, Mapping):
        raise ValueError("Expected observable_evidence as a mapping of component references.")
    evidence: dict[str, tuple[EvidenceReference, ...]] = {}
    for name, references in observable_evidence.items():
        if name not in COMPONENT_NAMES:
            raise ValueError(f"Evidence was provided for unknown component {name!r}.")
        cited_references = _validated_observable_references(references)
        if not set(cited_references).issubset(observable_references):
            raise ValueError(f"Evidence for {name!r} must use observable references.")
        evidence[name] = cited_references
    return MappingProxyType(evidence)


@dataclass(frozen=True)
class RewardObservation:
    """The observable component values and citations available to the overlay."""

    objective_fidelity: float = 0.0
    feedback_integrity: float = 0.0
    skill_transfer: float = 0.0
    reality_contact: float = 0.0
    exploit_disclosure: float = 0.0
    long_horizon_agency: float = 0.0
    benign_creativity: float = 0.0
    repair_quality: float = 0.0
    observable_evidence: Mapping[str, Sequence[EvidenceReference]] = field(default_factory=dict)
    observable_references: Sequence[EvidenceReference] = ()

    def __post_init__(self) -> None:
        """Bound values and bind every negative value to observable evidence."""
        components = self.components
        for name, value in components.items():
            object.__setattr__(self, name, _validated_component(name, value))

        observable_references = _validated_observable_references(self.observable_references)
        evidence = _validated_observable_evidence(
            self.observable_evidence,
            observable_references,
        )

        for name, value in self.components.items():
            if value < 0.0 and not evidence.get(name):
                raise ValueError(f"Negative {name} requires observable evidence.")
        object.__setattr__(self, "observable_evidence", MappingProxyType(evidence))
        object.__setattr__(self, "observable_references", observable_references)

    @property
    def components(self) -> Mapping[str, float]:
        """Return all eight components without reducing them to an aggregate."""
        return MappingProxyType({name: getattr(self, name) for name in COMPONENT_NAMES})


@dataclass(frozen=True)
class RewardIntegrityWeights:
    """Non-negative component weights for the reward-integrity aggregate."""

    objective_fidelity: float = 1.0
    feedback_integrity: float = 1.0
    skill_transfer: float = 1.0
    reality_contact: float = 1.0
    exploit_disclosure: float = 1.0
    long_horizon_agency: float = 1.0
    benign_creativity: float = 1.0
    repair_quality: float = 1.0

    def __getitem__(self, name: str) -> float:
        """Return the named component weight for aggregation."""
        if name not in COMPONENT_NAMES:
            raise KeyError(name)
        return float(getattr(self, name))

    @property
    def total(self) -> float:
        """Return the validated weight mass."""
        return sum(self[name] for name in COMPONENT_NAMES)

    def validate(self) -> None:
        """Require finite, non-negative weights with positive total mass."""
        for name in COMPONENT_NAMES:
            value = self[name]
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"Weight {name} must be finite and non-negative.")
        if not isfinite(self.total):
            raise ValueError("Reward-integrity weight total mass must be finite.")
        if self.total <= 0.0:
            raise ValueError("Reward-integrity weights must have positive mass.")


@dataclass(frozen=True)
class RewardIntegrityBreakdown:
    """All source components and their weighted aggregate for one observation."""

    objective_fidelity: float
    feedback_integrity: float
    skill_transfer: float
    reality_contact: float
    exploit_disclosure: float
    long_horizon_agency: float
    benign_creativity: float
    repair_quality: float
    aggregate: float
    observable_evidence: Mapping[str, Sequence[EvidenceReference]] = field(default_factory=dict)
    observable_references: Sequence[EvidenceReference] = ()
    weights: RewardIntegrityWeights = field(default_factory=RewardIntegrityWeights)

    def __post_init__(self) -> None:
        """Keep public component and aggregate records in their bounded range."""
        for name in COMPONENT_NAMES:
            object.__setattr__(self, name, _validated_component(name, getattr(self, name)))
        object.__setattr__(self, "aggregate", _validated_component("aggregate", self.aggregate))
        self.weights.validate()
        expected_aggregate = aggregate_components(self.components, self.weights)
        if self.aggregate != expected_aggregate:
            raise ValueError("aggregate must equal the weighted component aggregate.")
        observable_references = _validated_observable_references(self.observable_references)
        evidence = _validated_observable_evidence(
            self.observable_evidence,
            observable_references,
        )
        for name, value in self.components.items():
            if value < 0.0 and not evidence.get(name):
                raise ValueError(f"Negative {name} requires observable evidence.")
        object.__setattr__(self, "observable_evidence", evidence)
        object.__setattr__(self, "observable_references", observable_references)

    @property
    def components(self) -> Mapping[str, float]:
        """Return the full component record for observable logs and audits."""
        return MappingProxyType({name: getattr(self, name) for name in COMPONENT_NAMES})


def aggregate_components(
    components: Mapping[str, float],
    weights: RewardIntegrityWeights,
) -> float:
    """Compute the validated weighted arithmetic mean of all eight components."""
    if set(components) != set(COMPONENT_NAMES):
        raise ValueError("Expected exactly the eight reward-integrity components.")
    weights.validate()
    validated_components = {
        name: _validated_component(name, components[name]) for name in COMPONENT_NAMES
    }
    weighted = sum(validated_components[name] * weights[name] for name in COMPONENT_NAMES)
    return weighted / weights.total


class RewardIntegrityCalculator:
    """Calculate a separately observable reward-integrity breakdown."""

    def __init__(self, weights: RewardIntegrityWeights | None = None) -> None:
        self.weights = weights or RewardIntegrityWeights()
        self.weights.validate()

    def compute(self, observation: RewardObservation) -> RewardIntegrityBreakdown:
        """Preserve components and add their weighted aggregate."""
        components = observation.components
        return RewardIntegrityBreakdown(
            **components,
            aggregate=aggregate_components(components, self.weights),
            observable_evidence=observation.observable_evidence,
            observable_references=observation.observable_references,
            weights=self.weights,
        )


__all__ = [
    "COMPONENT_NAMES",
    "RewardIntegrityBreakdown",
    "RewardIntegrityCalculator",
    "RewardIntegrityWeights",
    "RewardObservation",
    "aggregate_components",
]
