"""Typed policy versions and deterministic actor-staleness decisions."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from enum import Enum

_CANONICAL_VERSION = re.compile(r"0|[1-9][0-9]*")
_MAX_DECISION_REASON_CHARACTERS = 512


@dataclass(frozen=True, order=True, slots=True)
class PolicyVersion:
    """A comparable, non-negative monotonic policy version."""

    value: int

    def __post_init__(self) -> None:
        """Reject coercible or negative values at the typed boundary."""
        if type(self.value) is not int or self.value < 0:
            raise ValueError("policy version must be a non-negative integer")

    def to_json(self) -> str:
        """Return the canonical string stored in trajectory policy-version fields."""
        return str(self.value)

    @classmethod
    def from_json(cls, value: object) -> PolicyVersion:
        """Restore a version only from its canonical JSON string representation."""
        if type(value) is not str or _CANONICAL_VERSION.fullmatch(value) is None:
            raise ValueError("policy version JSON value must be a canonical decimal string")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(
                "policy version JSON value must be a canonical decimal string"
            ) from exc
        return cls(parsed)

    def __str__(self) -> str:
        """Return the stable external representation."""
        return self.to_json()


class StalenessPolicy(str, Enum):
    """Action to take when actor lag exceeds the configured allowance."""

    REJECT = "reject"
    DOWN_WEIGHT = "down_weight"


@dataclass(frozen=True, slots=True)
class StalenessDecision:
    """Immutable evidence for one actor-versus-learner version decision."""

    accepted: bool
    learner_version: PolicyVersion
    actor_version: PolicyVersion
    lag: int
    weight: float
    policy: StalenessPolicy
    max_lag: int
    downweight_decay: float | None
    reason: str

    def __post_init__(self) -> None:
        """Reject contradictory or ambiguous audit evidence."""
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        _validate_version_argument(self.learner_version, "learner_version")
        _validate_version_argument(self.actor_version, "actor_version")
        if type(self.lag) is not int or self.lag < 0:
            raise ValueError("lag must be a non-negative integer")
        expected_lag = self.learner_version.value - self.actor_version.value
        if expected_lag < 0:
            raise ValueError("lag cannot represent an actor version ahead of the learner")
        if self.lag != expected_lag:
            raise ValueError("lag must equal learner_version minus actor_version")
        if type(self.weight) is not float:
            raise ValueError("weight must be a float")
        if not math.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be finite and in [0, 1]")
        _validate_max_lag(self.max_lag)
        decay = _validate_policy_configuration(self.policy, self.downweight_decay)
        if type(self.reason) is not str:
            raise ValueError("reason must be a string")
        if not self.reason.strip() or len(self.reason) > _MAX_DECISION_REASON_CHARACTERS:
            raise ValueError("reason must be nonblank and at most 512 characters")
        self._validate_outcome(decay)

    def _validate_outcome(self, decay: float | None) -> None:
        if self.accepted and self.weight <= 0.0:
            raise ValueError("accepted decisions must have a positive weight")
        if not self.accepted and self.weight != 0.0:
            raise ValueError("rejected decisions must have weight 0.0")
        if self.lag <= self.max_lag:
            if not self.accepted or self.weight != 1.0:
                raise ValueError("allowed-lag decisions must be accepted with weight 1.0")
            return
        if self.policy is StalenessPolicy.REJECT:
            if self.accepted or self.weight != 0.0:
                raise ValueError("reject policy must reject stale decisions with weight 0.0")
            return
        assert decay is not None
        expected_weight = _decayed_weight(decay, self.lag - self.max_lag)
        if not self.accepted or self.weight != expected_weight:
            raise ValueError(
                "down_weight policy must accept stale decisions with the configured weight"
            )


def evaluate_staleness(
    learner_version: PolicyVersion,
    actor_version: PolicyVersion,
    *,
    max_lag: int = 0,
    policy: StalenessPolicy = StalenessPolicy.REJECT,
    downweight_decay: float | None = None,
) -> StalenessDecision:
    """Evaluate an actor version without consulting time or mutable state."""
    _validate_version_argument(learner_version, "learner_version")
    _validate_version_argument(actor_version, "actor_version")
    _validate_max_lag(max_lag)
    decay = _validate_policy_configuration(policy, downweight_decay)

    lag = learner_version.value - actor_version.value
    if lag < 0:
        ahead = -lag
        raise ValueError(
            f"actor policy version {actor_version} is {ahead} version(s) ahead of learner "
            f"policy version {learner_version}"
        )
    if lag <= max_lag:
        return StalenessDecision(
            accepted=True,
            learner_version=learner_version,
            actor_version=actor_version,
            lag=lag,
            weight=1.0,
            policy=policy,
            max_lag=max_lag,
            downweight_decay=decay,
            reason=(
                f"accepted actor policy version {actor_version}: lag {lag} is within "
                f"max_lag {max_lag}"
            ),
        )
    if policy is StalenessPolicy.REJECT:
        return StalenessDecision(
            accepted=False,
            learner_version=learner_version,
            actor_version=actor_version,
            lag=lag,
            weight=0.0,
            policy=policy,
            max_lag=max_lag,
            downweight_decay=None,
            reason=(
                f"rejected actor policy version {actor_version}: lag {lag} exceeds "
                f"max_lag {max_lag} under reject policy"
            ),
        )

    assert decay is not None
    weight = _decayed_weight(decay, lag - max_lag)
    return StalenessDecision(
        accepted=True,
        learner_version=learner_version,
        actor_version=actor_version,
        lag=lag,
        weight=weight,
        policy=policy,
        max_lag=max_lag,
        downweight_decay=decay,
        reason=(
            f"accepted actor policy version {actor_version} with weight {weight!r}: lag {lag} "
            f"exceeds max_lag {max_lag} under {policy.value} policy with "
            f"downweight_decay {decay!r}"
        ),
    )


def _validate_version_argument(value: object, field_name: str) -> None:
    if type(value) is not PolicyVersion:
        raise ValueError(f"{field_name} must be a PolicyVersion")


def _validate_max_lag(max_lag: object) -> None:
    if type(max_lag) is not int or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")


def _validate_policy_configuration(
    policy: object,
    downweight_decay: object,
) -> float | None:
    if type(policy) is not StalenessPolicy:
        raise ValueError("policy must be a StalenessPolicy")
    if policy is StalenessPolicy.REJECT:
        if downweight_decay is not None:
            raise ValueError("downweight_decay must be omitted for the reject policy")
        return None
    if (
        isinstance(downweight_decay, bool)
        or not isinstance(downweight_decay, (int, float))
        or not math.isfinite(downweight_decay)
        or not 0.0 < downweight_decay < 1.0
    ):
        raise ValueError("downweight_decay must be finite and in (0, 1)")
    return float(downweight_decay)


def _decayed_weight(decay: float, excess_lag: int) -> float:
    minimum = sys.float_info.min
    underflow_boundary = math.log(minimum) / math.log(decay)
    if excess_lag >= underflow_boundary:
        return minimum
    return decay**excess_lag


__all__ = [
    "PolicyVersion",
    "StalenessDecision",
    "StalenessPolicy",
    "evaluate_staleness",
]
