"""Observable, component-preserving reward-integrity calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

from .evidence import EvidenceReference
from .reward_provenance import RewardProvenance, VerificationRoute

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


def _validated_component_provenance(
    value: object,
    components: Mapping[str, float],
    observable_references: tuple[EvidenceReference, ...],
) -> Mapping[str, RewardProvenance]:
    """Bind each nonzero component to one route valid inside the observable boundary."""
    if not isinstance(value, Mapping):
        raise ValueError("Expected reward_component_provenance as a component mapping.")
    provenance_by_component: dict[str, RewardProvenance] = {}
    for component, provenance in value.items():
        if component not in COMPONENT_NAMES:
            raise ValueError(f"Provenance was provided for unknown component {component!r}.")
        if not isinstance(provenance, RewardProvenance):
            raise ValueError(f"Provenance for {component!r} must be a RewardProvenance record.")
        if provenance.component_name != component:
            raise ValueError(f"Provenance for {component!r} has a different component_name.")
        if provenance.route is VerificationRoute.OBSERVABLE_EVIDENCE and not set(
            provenance.evidence_refs
        ).issubset(observable_references):
            raise ValueError(
                f"Provenance for {component!r} must use authorized observable references."
            )
        provenance_by_component[component] = provenance

    for component, component_value in components.items():
        if component_value != 0.0 and component not in provenance_by_component:
            raise ValueError(f"Nonzero {component} requires reward provenance.")
    return MappingProxyType(provenance_by_component)


@dataclass(frozen=True)
class SelfServingJustificationCheck:
    """Reviewer-supplied act diagnostics; no field authorizes actions or numeric reward.

    Optional booleans record a review finding: True passes, False fails, and None
    means unresolved. Override references are citations to inspect, not permission.
    """

    act_classification: str
    beneficiary: str
    self_benefit_present: bool
    constraint_recognized: bool
    independent_override_evidence: tuple[EvidenceReference, ...] = ()
    counterfactual_without_self_benefit: bool | None = None
    role_reversal_result: bool | None = None
    authority: bool | None = None
    necessity: bool | None = None
    proportionality: bool | None = None
    reversibility: bool | None = None

    def __post_init__(self) -> None:
        for name in ("act_classification", "beneficiary"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        for name in ("self_benefit_present", "constraint_recognized"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        for name in self._review_fields:
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or None.")
        object.__setattr__(
            self,
            "independent_override_evidence",
            _validated_observable_references(self.independent_override_evidence),
        )

    @property
    def _review_fields(self) -> tuple[str, ...]:
        return (
            "counterfactual_without_self_benefit",
            "role_reversal_result",
            "authority",
            "necessity",
            "proportionality",
            "reversibility",
        )

    @property
    def scrutiny_reasons(self) -> tuple[str, ...]:
        """Keep self-interest and recognized constraints visible even with cited evidence."""
        reasons: list[str] = []
        if self.self_benefit_present:
            reasons.append("self_benefit_present")
        if self.constraint_recognized:
            reasons.append("recognized_constraint_requires_independent_review")
            if not self.independent_override_evidence:
                reasons.append("missing_independent_override_evidence")
        for name in self._review_fields:
            value = getattr(self, name)
            if value is not True:
                reasons.append(f"{name}_{'unresolved' if value is None else 'failed'}")
        return tuple(reasons)

    @property
    def requires_scrutiny(self) -> bool:
        return bool(self.scrutiny_reasons)

    @property
    def conclusion(self) -> str:
        """Return a scrutiny decision, never an approval or a reward component."""
        return "INCREASE_SCRUTINY" if self.requires_scrutiny else "NO_ADDITIONAL_SCRUTINY"


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
    reward_component_provenance: Mapping[str, RewardProvenance] = field(default_factory=dict)
    self_serving_justification: SelfServingJustificationCheck | None = None

    def __post_init__(self) -> None:
        """Require observable evidence for negatives and provenance for every nonzero component."""
        if self.self_serving_justification is not None and not isinstance(
            self.self_serving_justification, SelfServingJustificationCheck
        ):
            raise ValueError("self_serving_justification must be a SelfServingJustificationCheck.")
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
        provenance = _validated_component_provenance(
            self.reward_component_provenance,
            self.components,
            observable_references,
        )
        object.__setattr__(self, "observable_evidence", MappingProxyType(evidence))
        object.__setattr__(self, "observable_references", observable_references)
        object.__setattr__(self, "reward_component_provenance", provenance)

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
    reward_component_provenance: Mapping[str, RewardProvenance] = field(default_factory=dict)
    self_serving_justification: SelfServingJustificationCheck | None = None

    def __post_init__(self) -> None:
        """Bound components, require negative evidence, and bind every nonzero provenance."""
        if self.self_serving_justification is not None and not isinstance(
            self.self_serving_justification, SelfServingJustificationCheck
        ):
            raise ValueError("self_serving_justification must be a SelfServingJustificationCheck.")
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
        provenance = _validated_component_provenance(
            self.reward_component_provenance,
            self.components,
            observable_references,
        )
        object.__setattr__(self, "observable_evidence", evidence)
        object.__setattr__(self, "observable_references", observable_references)
        object.__setattr__(self, "reward_component_provenance", provenance)

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
            reward_component_provenance=observation.reward_component_provenance,
            weights=self.weights,
            self_serving_justification=observation.self_serving_justification,
        )


__all__ = [
    "COMPONENT_NAMES",
    "RewardIntegrityBreakdown",
    "RewardIntegrityCalculator",
    "RewardIntegrityWeights",
    "RewardObservation",
    "SelfServingJustificationCheck",
    "aggregate_components",
]
