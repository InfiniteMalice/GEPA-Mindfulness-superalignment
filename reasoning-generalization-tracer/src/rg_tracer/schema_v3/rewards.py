"""Reward helpers for Schema V3 additive overlays."""

from __future__ import annotations

from collections.abc import Sequence
from math import isclose, isfinite
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .case_v3 import RewardComponents


class ProcessComponentName(Protocol):
    """Structural interface for a named verified process component."""

    @property
    def value(self) -> str: ...


class ProcessComponentProvenance(Protocol):
    """Structural interface for the component identity bound into provenance."""

    @property
    def component_name(self) -> str: ...


class VerifiedProcessComponent(Protocol):
    """Structural interface consumed by Schema V3 reward lookup."""

    @property
    def component(self) -> ProcessComponentName: ...

    @property
    def score(self) -> float: ...

    @property
    def provenance(self) -> ProcessComponentProvenance: ...


class EpistemicProcessAssessment(Protocol):
    """Dependency-minimal interface for a verified process assessment."""

    @property
    def verified_components(self) -> Sequence[VerifiedProcessComponent]: ...

    def optimizer_score(self) -> float: ...


def verified_component_score(
    epistemic_process: EpistemicProcessAssessment | None,
    component_name: str,
) -> float:
    """Return an exact independently verified component score, or zero."""
    if epistemic_process is None:
        return 0.0
    components = _validated_components(epistemic_process)
    for verified_component in components:
        if verified_component.component.value == component_name:
            return _validated_score(verified_component.score)
    return 0.0


def verified_optimizer_score(
    epistemic_process: EpistemicProcessAssessment | None,
) -> float:
    """Return a bounded optimizer score consistent with verified components."""

    if epistemic_process is None:
        return 0.0
    components = _validated_components(epistemic_process)
    score = _validated_score(epistemic_process.optimizer_score())
    expected = (
        sum(_validated_score(item.score) for item in components) / len(components)
        if components
        else 0.0
    )
    if not isclose(score, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("optimizer score must equal the mean of verified component scores")
    return score


def _validated_components(
    epistemic_process: EpistemicProcessAssessment,
) -> tuple[VerifiedProcessComponent, ...]:
    components = tuple(epistemic_process.verified_components)
    names = tuple(item.component.value for item in components)
    if len(set(names)) != len(names):
        raise ValueError("verified process components must have unique names")
    for component in components:
        if type(component.component.value) is not str or not component.component.value:
            raise ValueError("verified process component name must be a nonblank string")
        if component.provenance.component_name != component.component.value:
            raise ValueError("process provenance component_name must match the component")
        _validated_score(component.score)
    return components


def _validated_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a finite number in [0.0, 1.0]")
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a finite number in [0.0, 1.0]")
    return score


def assert_thought_reward_non_negative(reward_components: RewardComponents) -> None:
    """Raise if a caller attempts to introduce a negative hidden-thought reward."""
    if reward_components.r_thought < 0.0:
        raise ValueError(
            "r_thought must be non-negative: 0 or H * optimizer_score(), never negative"
        )
