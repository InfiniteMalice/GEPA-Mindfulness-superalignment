"""Observable, component-preserving reward-integrity calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

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
_PRIVATE_EVIDENCE_TERMS = (
    "activation",
    "chain of thought",
    "hidden state",
    "private scratchpad",
)


def _validated_component(name: str, value: object) -> float:
    """Return one finite, bounded observable component."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {name} in [-1.0, 1.0].")
    numeric_value = float(value)
    if not isfinite(numeric_value) or not -1.0 <= numeric_value <= 1.0:
        raise ValueError(f"Expected finite {name} in [-1.0, 1.0].")
    return numeric_value


def _is_private_evidence(reference: str) -> bool:
    """Identify references to model-private data, which are never observable evidence."""
    normalised = reference.lower().replace("_", " ")
    return any(term in normalised for term in _PRIVATE_EVIDENCE_TERMS)


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
    observable_evidence: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Bound values and bind every negative value to observable evidence."""
        components = self.components
        for name, value in components.items():
            object.__setattr__(self, name, _validated_component(name, value))

        if not isinstance(self.observable_evidence, Mapping):
            raise ValueError("Expected observable_evidence as a mapping of component references.")
        evidence: dict[str, tuple[str, ...]] = {}
        for name, references in self.observable_evidence.items():
            if name not in COMPONENT_NAMES:
                raise ValueError(f"Evidence was provided for unknown component {name!r}.")
            if not isinstance(references, (list, tuple)) or not all(
                isinstance(reference, str) and reference.strip() for reference in references
            ):
                raise ValueError(f"Expected observable evidence references for {name!r}.")
            if any(_is_private_evidence(reference) for reference in references):
                raise ValueError("Observable evidence cannot contain private model information.")
            evidence[name] = tuple(references)

        for name, value in self.components.items():
            if value < 0.0 and not evidence.get(name):
                raise ValueError(f"Negative {name} requires observable evidence.")
        object.__setattr__(self, "observable_evidence", MappingProxyType(evidence))

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

    def __post_init__(self) -> None:
        """Keep public component and aggregate records in their bounded range."""
        for name in COMPONENT_NAMES:
            object.__setattr__(self, name, _validated_component(name, getattr(self, name)))
        object.__setattr__(self, "aggregate", _validated_component("aggregate", self.aggregate))

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
    weighted = sum(components[name] * weights[name] for name in COMPONENT_NAMES)
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
        )


__all__ = [
    "COMPONENT_NAMES",
    "RewardIntegrityBreakdown",
    "RewardIntegrityCalculator",
    "RewardIntegrityWeights",
    "RewardObservation",
    "aggregate_components",
]
