"""Inert host contracts for the experimental, offline world-model ablation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from math import isfinite
from typing import TypeVar

from mindful_trace_gepa._json_values import require_serialization_safe_integer


class Arm(str, Enum):
    """Representation treatment; no arm receives additional latent world facts."""

    DIRECT = "direct"
    STRUCTURED = "structured"
    PEO = "peo"


def _text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonblank exact string")


def _count(value: int, name: str) -> None:
    require_serialization_safe_integer(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _probability(value: float | None, name: str) -> None:
    if value is not None and (
        type(value) not in (int, float) or not 0 <= value <= 1 or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite unit probability or None")


@dataclass(frozen=True, slots=True)
class Budget:
    """Common caps or remaining allowance; compute units are named by ModelContract."""

    model_calls: int
    tool_calls: int
    compute_units: int

    def __post_init__(self) -> None:
        for name in ("model_calls", "tool_calls", "compute_units"):
            _count(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ModelContract:
    """Host-declared shared identity and training budget; the harness cannot attest them."""

    model_version: str
    checkpoint_sha256: str
    adapter_version: str
    training_examples: int
    training_compute_units: int
    compute_unit: str

    def __post_init__(self) -> None:
        for name in ("model_version", "checkpoint_sha256", "adapter_version", "compute_unit"):
            _text(getattr(self, name), name)
        if len(self.checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be 64 lowercase hexadecimal characters")
        _count(self.training_examples, "training_examples")
        _count(self.training_compute_units, "training_compute_units")


@dataclass(frozen=True, slots=True)
class WorldCase:
    """Evaluator-only JSON world snapshot and the public actor/target task definition."""

    case_id: str
    world_json: str
    actor_id: str
    target_action: str
    cohort: str
    severity: str = "routine"

    def __post_init__(self) -> None:
        for name in ("case_id", "world_json", "actor_id", "target_action", "cohort", "severity"):
            _text(getattr(self, name), name)
        if self.severity not in ("routine", "consequential", "catastrophic"):
            raise ValueError("severity must be routine, consequential or catastrophic")


@dataclass(frozen=True, slots=True)
class DecisionInput:
    """Disposable public JSON, treatment and remaining budget passed to the host policy."""

    arm: Arm
    payload_json: str
    remaining: Budget


@dataclass(frozen=True, slots=True)
class Decision:
    """A prospective action (None means abstain) and host-metered compute receipt."""

    action_id: str | None
    compute_used: int
    predicted_success: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.action_id is not None:
            _text(self.action_id, "action_id")
        _count(self.compute_used, "compute_used")
        if self.compute_used == 0:
            raise ValueError("compute_used must be positive")
        _probability(self.predicted_success, "predicted_success")
        _probability(self.confidence, "confidence")


Policy = Callable[[DecisionInput], Decision]


@dataclass(frozen=True, slots=True)
class WorldBackend:
    """One shared adapter; factory(seed) returns a fresh, isolated policy per case/arm."""

    contract: ModelContract
    factory: Callable[[int], Policy]

    def __post_init__(self) -> None:
        _snapshot(self.contract, ModelContract)
        if not callable(self.factory):
            raise ValueError("factory must be callable")


_T = TypeVar("_T", Budget, ModelContract, WorldCase, Decision)


def _snapshot(value: _T, cls: type[_T]) -> _T:
    if type(value) is not cls:
        raise ValueError(f"expected exact {cls.__name__}")
    return cls(**{field.name: getattr(value, field.name) for field in fields(cls)})


def _primitive_fields(value: Budget | ModelContract) -> dict[str, str | int]:
    return {field.name: getattr(value, field.name) for field in fields(type(value))}
