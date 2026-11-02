"""Versioned evaluator-only inputs for bounded action and latent-cost probes."""

# Standard library
from __future__ import annotations

import math
from dataclasses import dataclass

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.verification.debate_records import _digest
from gepa_mindfulness.verification.diagnostic_records import (
    DiagnosticRecord,
    public_refs,
    restore_refs,
    strings,
)

MAX_ROWS = 512
MAX_DIMENSION = 16


def text(value: object, name: str) -> None:
    """Bound inert record coordinates without interpreting their text."""
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 512:
        raise ValueError(f"bounded nonempty {name} required")


def number(value: object, *, limit: float = 1e6) -> float:
    """Reject coercible, nonfinite or unsupported-scale numerical inputs."""
    if type(value) not in (float, int):
        raise ValueError("finite nonboolean number required")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("number outside supported range") from error
    if not math.isfinite(result) or abs(result) > limit:
        raise ValueError("number outside supported range")
    return result


def vector(value: object) -> tuple[float, ...]:
    """Detach small real vectors without importing or invoking an array backend."""
    if type(value) not in (list, tuple) or not 1 <= len(value) <= MAX_DIMENSION:
        raise ValueError("1..16 vector coordinates required")
    return tuple(number(v) for v in value)


def count(value: object, name: str, maximum: int) -> None:
    """Require exact bounded integer coordinates."""
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"invalid {name}")


def refs(value: object) -> tuple[EvidenceReference, ...]:
    """Retain bounded observable provenance; a reference alone is not verification."""
    result = public_refs(value, required=True)
    if len(result) > 16:
        raise ValueError("too many evidence references")
    for ref in result:
        text(ref.reference_id, "reference_id")
    return result


@dataclass(frozen=True)
class DynamicsProbeConfig(DiagnosticRecord):
    """Default-off fixed-protocol settings, not a training or promotion configuration."""

    enabled: bool = False
    seed: int = 0
    schema_version = "intrinsic-dynamics-config-v1"

    def __post_init__(self) -> None:
        """Reject implicit activation and unsupported permutation seeds."""
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        count(self.seed, "seed", 2**32 - 1)


@dataclass(frozen=True)
class ProbeTransition(DiagnosticRecord):
    """Declared frozen latents with independently accessible transition provenance."""

    trajectory_id: str
    step_index: int
    root_world_digest: str
    transition_digest: str
    action_id: str
    action_vocabulary: tuple[str, ...]
    before: tuple[float, ...]
    after: tuple[float, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    predicted_after: tuple[float, ...] | None = None
    schema_version = "intrinsic-dynamics-transition-v1"
    restorers = {
        "action_vocabulary": lambda value: strings(value, "action_vocabulary", required=True),
        "evidence_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        """Preserve immutable numeric inputs while refusing undeclared action semantics."""
        text(self.trajectory_id, "trajectory_id")
        text(self.action_id, "action_id")
        count(self.step_index, "step_index", MAX_ROWS - 1)
        _digest(self.root_world_digest)
        _digest(self.transition_digest)
        vocabulary = strings(self.action_vocabulary, "action_vocabulary", required=True)
        if len(vocabulary) > MAX_DIMENSION or self.action_id not in vocabulary:
            raise ValueError("bounded action vocabulary must contain action_id")
        for action in vocabulary:
            text(action, "action vocabulary entry")
        object.__setattr__(self, "action_vocabulary", vocabulary)
        for name in ("before", "after", "predicted_after"):
            value = getattr(self, name)
            if value is not None or name != "predicted_after":
                object.__setattr__(self, name, vector(value))
        if len(self.before) != len(self.after) or (
            self.predicted_after is not None and len(self.predicted_after) != len(self.after)
        ):
            raise ValueError("latent dimensions must match")
        object.__setattr__(self, "evidence_refs", refs(self.evidence_refs))


@dataclass(frozen=True)
class RolloutCost(DiagnosticRecord):
    """One predeclared candidate's supplied costs; no certified outcome or selected action."""

    trajectory_id: str
    decision_id: str
    horizon: int
    candidate_id: str
    plan_digest: str
    predicted_cost: float | None
    realized_cost: float | None
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version = "intrinsic-dynamics-rollout-cost-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        """Missing costs stay missing; malformed or negative costs cannot become zeros."""
        for name in ("trajectory_id", "decision_id", "candidate_id"):
            text(getattr(self, name), name)
        count(self.horizon, "horizon", MAX_ROWS)
        if self.horizon == 0:
            raise ValueError("positive horizon required")
        _digest(self.plan_digest)
        for name in ("predicted_cost", "realized_cost"):
            value = getattr(self, name)
            if value is not None:
                value = number(value, limit=1e12)
                if value < 0:
                    raise ValueError("latent costs must be nonnegative")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "evidence_refs", refs(self.evidence_refs))
