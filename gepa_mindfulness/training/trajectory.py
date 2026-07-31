"""Backend-neutral, JSON-serializable reinforcement-learning trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence


def _optional_int_tuple(value: object | None) -> tuple[int, ...] | None:
    """Restore an optional JSON integer array as an immutable tuple."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected an array or null for token IDs.")
    return tuple(int(item) for item in value)


def _optional_float_tuple(value: object | None) -> tuple[float, ...] | None:
    """Restore an optional JSON numeric array as an immutable tuple."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected an array or null for trajectory values.")
    if not all(isinstance(item, (int, float)) for item in value):
        raise ValueError("Expected numeric trajectory values.")
    return tuple(float(item) for item in value)


def _mapping(value: object | None, field_name: str) -> Mapping[str, object]:
    """Restore a JSON object while rejecting non-object metadata fields."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object or null for {field_name}.")
    return dict(value)


def _string_tuple(value: object | None, field_name: str) -> tuple[str, ...]:
    """Restore a JSON string array as an immutable tuple."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected an array or null for {field_name}.")
    return tuple(str(item) for item in value)


def _component_evidence(value: object | None) -> Mapping[str, tuple[str, ...]]:
    """Restore component evidence references from their JSON object representation."""
    raw_evidence = _mapping(value, "reward_component_evidence")
    return {
        component: _string_tuple(references, "reward component evidence")
        for component, references in raw_evidence.items()
    }


@dataclass(frozen=True)
class Trajectory:
    """One completed rollout with optional data that the backend actually observed."""

    trajectory_id: str
    case_id: str | None
    prompt: str
    response: str
    prompt_token_ids: tuple[int, ...] | None = None
    response_token_ids: tuple[int, ...] | None = None
    old_log_probs: tuple[float, ...] | None = None
    reference_log_probs: tuple[float, ...] | None = None
    value_predictions: tuple[float, ...] | None = None
    reward_total: float | None = None
    reward_components: Mapping[str, float] = field(default_factory=dict)
    reward_component_evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    advantage: tuple[float, ...] | None = None
    returns: tuple[float, ...] | None = None
    sampling_parameters: Mapping[str, object] = field(default_factory=dict)
    backend_name: str = ""
    backend_version: str = ""
    model_identifier: str = ""
    adapter_identifier: str | None = None
    policy_version: str | None = None
    seed: int | None = None
    trace_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate reward signals and bind negative values to recorded evidence."""
        components: dict[str, float] = {}
        for component, value in self.reward_components.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Expected a numeric reward component for {component!r}.")
            numeric_value = float(value)
            if not isfinite(numeric_value) or not -1.0 <= numeric_value <= 1.0:
                raise ValueError(
                    f"Expected a finite reward component in [-1.0, 1.0] for {component!r}."
                )
            components[component] = numeric_value

        recorded_references = set(self.trace_references)
        component_evidence: dict[str, tuple[str, ...]] = {}
        for component, references in self.reward_component_evidence.items():
            if component not in components:
                raise ValueError(
                    f"Evidence was provided for unknown reward component {component!r}."
                )
            if not isinstance(references, (list, tuple)) or not all(
                isinstance(reference, str) for reference in references
            ):
                raise ValueError(f"Expected trace references for reward component {component!r}.")
            if not set(references).issubset(recorded_references):
                raise ValueError(f"Evidence for {component!r} must use recorded trace references.")
            component_evidence[component] = tuple(references)

        for component, value in components.items():
            if value < 0.0 and not component_evidence.get(component):
                raise ValueError(
                    f"Negative reward component {component!r} requires observable evidence."
                )

        object.__setattr__(self, "reward_components", MappingProxyType(components))
        object.__setattr__(self, "reward_component_evidence", MappingProxyType(component_evidence))
        object.__setattr__(
            self,
            "sampling_parameters",
            MappingProxyType(dict(self.sampling_parameters)),
        )

    @classmethod
    def minimal(cls, trajectory_id: str, prompt: str, response: str) -> "Trajectory":
        """Create a trajectory when a backend only returned text."""
        return cls(
            trajectory_id=trajectory_id,
            case_id=None,
            prompt=prompt,
            response=response,
        )

    def to_dict(self) -> dict[str, object]:
        """Convert this trajectory to the stable JSON representation."""
        return {
            "trajectory_id": self.trajectory_id,
            "case_id": self.case_id,
            "prompt": self.prompt,
            "response": self.response,
            "prompt_token_ids": self._optional_list(self.prompt_token_ids),
            "response_token_ids": self._optional_list(self.response_token_ids),
            "old_log_probs": self._optional_list(self.old_log_probs),
            "reference_log_probs": self._optional_list(self.reference_log_probs),
            "value_predictions": self._optional_list(self.value_predictions),
            "reward_total": self.reward_total,
            "reward_components": dict(self.reward_components),
            "reward_component_evidence": {
                component: list(references)
                for component, references in self.reward_component_evidence.items()
            },
            "advantage": self._optional_list(self.advantage),
            "return": self._optional_list(self.returns),
            "sampling_parameters": dict(self.sampling_parameters),
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "model_identifier": self.model_identifier,
            "adapter_identifier": self.adapter_identifier,
            "policy_version": self.policy_version,
            "seed": self.seed,
            "trace_references": list(self.trace_references),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Trajectory":
        """Restore a trajectory from its JSON representation without inventing values."""
        reward_components = _mapping(data.get("reward_components"), "reward_components")
        components: dict[str, float] = {}
        for key, value in reward_components.items():
            if not isinstance(value, (int, float)):
                raise ValueError("Expected numeric reward component values.")
            components[key] = float(value)
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            case_id=cls._optional_string(data.get("case_id")),
            prompt=str(data["prompt"]),
            response=str(data["response"]),
            prompt_token_ids=_optional_int_tuple(data.get("prompt_token_ids")),
            response_token_ids=_optional_int_tuple(data.get("response_token_ids")),
            old_log_probs=_optional_float_tuple(data.get("old_log_probs")),
            reference_log_probs=_optional_float_tuple(data.get("reference_log_probs")),
            value_predictions=_optional_float_tuple(data.get("value_predictions")),
            reward_total=cls._optional_float(data.get("reward_total")),
            reward_components=components,
            reward_component_evidence=_component_evidence(data.get("reward_component_evidence")),
            advantage=_optional_float_tuple(data.get("advantage")),
            returns=_optional_float_tuple(data.get("return")),
            sampling_parameters=_mapping(data.get("sampling_parameters"), "sampling_parameters"),
            backend_name=str(data.get("backend_name", "")),
            backend_version=str(data.get("backend_version", "")),
            model_identifier=str(data.get("model_identifier", "")),
            adapter_identifier=cls._optional_string(data.get("adapter_identifier")),
            policy_version=cls._optional_string(data.get("policy_version")),
            seed=cls._optional_int(data.get("seed")),
            trace_references=_string_tuple(data.get("trace_references"), "trace_references"),
        )

    @staticmethod
    def _optional_list(values: Sequence[int] | Sequence[float] | None) -> list[object] | None:
        """Convert optional immutable numeric sequences to JSON arrays."""
        if values is None:
            return None
        return list(values)

    @staticmethod
    def _optional_string(value: object | None) -> str | None:
        """Restore an optional string without converting null to text."""
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _optional_float(value: object | None) -> float | None:
        """Restore an optional JSON number."""
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            raise ValueError("Expected a number or null for reward_total.")
        return float(value)

    @staticmethod
    def _optional_int(value: object | None) -> int | None:
        """Restore an optional JSON integer."""
        if value is None:
            return None
        if not isinstance(value, int):
            raise ValueError("Expected an integer or null for seed.")
        return int(value)


@dataclass(frozen=True)
class TrajectoryBatch:
    """A batch of trajectories and optional masks aligned to response tokens."""

    trajectories: tuple[Trajectory, ...]
    response_token_masks: tuple[tuple[bool, ...], ...] | None = None


@dataclass(frozen=True)
class RolloutRequest:
    """A request to generate one or more responses for a prompt."""

    prompt: str
    case_id: str | None = None
    num_samples: int = 1
    sampling_parameters: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    policy_version: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class PolicyEvaluation:
    """Backend tensors from evaluating a trajectory batch under a policy."""

    log_probs: object
    reference_log_probs: object | None = None
    value_predictions: object | None = None
    entropy: object | None = None
