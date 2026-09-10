"""Optional composition of an existing reward provider and reward-integrity overlay."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_integrity import (
    COMPONENT_NAMES,
    RewardIntegrityBreakdown,
    RewardIntegrityCalculator,
    RewardObservation,
)

from .contracts import RewardProvider, RewardRequest


@dataclass(frozen=True)
class RewardResult:
    """An existing provider result plus an optional, fully observable overlay."""

    base_reward: float
    total: float
    base_result: object
    integrity_breakdown: RewardIntegrityBreakdown | None = None

    @property
    def gepa_reward(self) -> float:
        """Return the existing GEPA reward using the design-document terminology."""
        return self.base_reward


class RewardPipeline:
    """Score a typed request and optionally add an observable integrity overlay."""

    def __init__(
        self,
        provider: RewardProvider,
        *,
        integrity_calculator: RewardIntegrityCalculator | None = None,
        overlay_weight: float = 0.0,
    ) -> None:
        if isinstance(overlay_weight, bool) or not isinstance(overlay_weight, (int, float)):
            raise ValueError("overlay_weight must be a finite non-negative number.")
        if not isfinite(float(overlay_weight)) or overlay_weight < 0.0:
            raise ValueError("overlay_weight must be a finite non-negative number.")
        self.provider = provider
        self.integrity_calculator = integrity_calculator
        self.overlay_weight = float(overlay_weight)

    @property
    def overlay_enabled(self) -> bool:
        """Return whether scoring will evaluate and compose the overlay."""
        return self.integrity_calculator is not None and self.overlay_weight > 0.0

    def score(
        self,
        request: RewardRequest,
        *,
        observation: RewardObservation | None = None,
    ) -> RewardResult:
        """Return the existing reward unchanged unless overlay composition is enabled."""
        base_result = self.provider.score(request)
        base_reward = self._reward_value(base_result)
        calculator = self.integrity_calculator
        if calculator is None or self.overlay_weight == 0.0:
            return RewardResult(
                base_reward=base_reward,
                total=base_reward,
                base_result=base_result,
            )

        if observation is not None and not set(observation.observable_references).issubset(
            request.observable_references
        ):
            raise ValueError(
                "Evaluator observation evidence falls outside request.observable_references."
            )
        overlay_observation = observation or self._observation_from_request(request)
        breakdown = calculator.compute(overlay_observation)
        return RewardResult(
            base_reward=base_reward,
            total=base_reward + self.overlay_weight * breakdown.aggregate,
            base_result=base_result,
            integrity_breakdown=breakdown,
        )

    @staticmethod
    def _reward_value(result: object) -> float:
        """Read a finite scalar from numeric providers or legacy result objects."""
        value = result if isinstance(result, (int, float)) else getattr(result, "total", None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("Reward provider must return a finite number or an object with total.")
        return float(value)

    @staticmethod
    def _observation_from_request(request: RewardRequest) -> RewardObservation:
        """Build an overlay observation using only request-authorized typed evidence references."""
        trajectory = request.trajectory
        if set(trajectory.reward_components) != set(COMPONENT_NAMES):
            raise ValueError(
                "Enabled overlay requires a complete authored observation with exactly the eight "
                "reward-integrity components."
            )
        allowed_references = set(request.observable_references).intersection(
            trajectory.evidence_references
        )
        evidence: dict[str, tuple[EvidenceReference, ...]] = {}
        for name in COMPONENT_NAMES:
            references = tuple(trajectory.reward_component_evidence.get(name, ()))
            if not set(references).issubset(allowed_references):
                raise ValueError(
                    f"Evidence for {name!r} falls outside request.observable_references."
                )
            if references:
                evidence[name] = references
        components = {name: trajectory.reward_components[name] for name in COMPONENT_NAMES}
        provenance = {
            name: trajectory.reward_component_provenance[name]
            for name in COMPONENT_NAMES
            if name in trajectory.reward_component_provenance
        }
        return RewardObservation(
            observable_evidence=evidence,
            observable_references=request.observable_references,
            reward_component_provenance=provenance,
            **components,
        )


__all__ = ["RewardPipeline", "RewardResult"]
