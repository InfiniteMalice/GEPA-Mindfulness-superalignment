"""Backend-neutral, JSON-serializable reinforcement-learning trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.evidence import EvidenceSourceKind as EvidenceSourceKind


def _required_string(value: object, field_name: str) -> str:
    """Return an actual string without converting another type."""
    if not isinstance(value, str):
        raise ValueError(f"Expected a string for {field_name}.")
    return value


def _optional_string(value: object | None, field_name: str) -> str | None:
    """Return an optional actual string without coercion."""
    if value is None:
        return None
    return _required_string(value, field_name)


def _optional_int_tuple(
    value: object | None,
    field_name: str,
) -> tuple[int, ...] | None:
    """Return an optional array of non-boolean integer token IDs."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected an array or null for {field_name}.")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"Expected non-boolean integer values for {field_name}.")
    return tuple(value)


def _optional_float_tuple(
    value: object | None,
    field_name: str,
) -> tuple[float, ...] | None:
    """Return an optional array of finite, non-boolean numeric values."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected an array or null for {field_name}.")
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and isfinite(item)
        for item in value
    ):
        raise ValueError(f"Expected finite non-boolean numeric values for {field_name}.")
    return tuple(float(item) for item in value)


def _optional_float(value: object | None, field_name: str) -> float | None:
    """Return one optional finite, non-boolean numeric value."""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"Expected a finite non-boolean number or null for {field_name}.")
    return float(value)


def _optional_int(value: object | None, field_name: str) -> int | None:
    """Return one optional non-boolean integer."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Expected a non-boolean integer or null for {field_name}.")
    return value


def _mapping(value: object | None, field_name: str) -> Mapping[str, object]:
    """Restore a JSON object while rejecting non-object metadata fields."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected an object or null for {field_name}.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Expected string object keys for {field_name}.")
    return dict(value)


def _evidence_reference_tuple(
    value: object | None,
    field_name: str,
    *,
    restore: bool = False,
) -> tuple[EvidenceReference, ...]:
    """Return an immutable array of typed evidence references."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected an array or null for {field_name}.")
    if restore:
        try:
            return tuple(EvidenceReference.from_dict(item) for item in value)
        except ValueError as error:
            raise ValueError(f"Invalid {field_name}: {error}") from error
    if not all(isinstance(item, EvidenceReference) for item in value):
        raise ValueError(f"Expected EvidenceReference values for {field_name}.")
    return tuple(value)


def _component_evidence(
    value: object | None,
    *,
    restore: bool = False,
) -> Mapping[str, tuple[EvidenceReference, ...]]:
    """Restore component evidence references from their JSON object representation."""
    raw_evidence = _mapping(value, "reward_component_evidence")
    evidence: dict[str, tuple[EvidenceReference, ...]] = {}
    for component, references in raw_evidence.items():
        if not isinstance(component, str):
            raise ValueError("Expected string reward_component_evidence keys.")
        evidence[component] = _evidence_reference_tuple(
            references,
            f"reward_component_evidence.{component}",
            restore=restore,
        )
    return evidence


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
    advantage: tuple[float, ...] | None = None
    returns: tuple[float, ...] | None = None
    sampling_parameters: Mapping[str, object] = field(default_factory=dict)
    backend_name: str = ""
    backend_version: str = ""
    model_identifier: str = ""
    adapter_identifier: str | None = None
    policy_version: str | None = None
    seed: int | None = None
    trace_references: tuple[EvidenceReference, ...] = ()
    reward_component_evidence: Mapping[str, tuple[EvidenceReference, ...]] = field(
        default_factory=dict,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        """Validate reward signals and bind negative values to recorded evidence."""
        for field_name in (
            "trajectory_id",
            "prompt",
            "response",
            "backend_name",
            "backend_version",
            "model_identifier",
        ):
            _required_string(getattr(self, field_name), field_name)
        for field_name in ("case_id", "adapter_identifier", "policy_version"):
            _optional_string(getattr(self, field_name), field_name)

        prompt_token_ids = _optional_int_tuple(self.prompt_token_ids, "prompt_token_ids")
        response_token_ids = _optional_int_tuple(self.response_token_ids, "response_token_ids")
        numeric_sequences = {
            field_name: _optional_float_tuple(getattr(self, field_name), field_name)
            for field_name in (
                "old_log_probs",
                "reference_log_probs",
                "value_predictions",
                "advantage",
                "returns",
            )
        }
        reward_total = _optional_float(self.reward_total, "reward_total")
        seed = _optional_int(self.seed, "seed")
        sampling_parameters = _mapping(self.sampling_parameters, "sampling_parameters")
        trace_references = _evidence_reference_tuple(
            self.trace_references,
            "trace_references",
        )

        if not isinstance(self.reward_components, Mapping):
            raise ValueError("Expected reward_components as a mapping.")
        components: dict[str, float] = {}
        for component, value in self.reward_components.items():
            if not isinstance(component, str):
                raise ValueError("Expected string reward component names.")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Expected a numeric reward component for {component!r}.")
            numeric_value = float(value)
            if not isfinite(numeric_value) or not -1.0 <= numeric_value <= 1.0:
                raise ValueError(
                    f"Expected a finite reward component in [-1.0, 1.0] for {component!r}."
                )
            components[component] = numeric_value

        recorded_references = set(trace_references)
        component_evidence = _component_evidence(self.reward_component_evidence)
        for component, references in component_evidence.items():
            if component not in components:
                raise ValueError(
                    f"Evidence was provided for unknown reward component {component!r}."
                )
            if not set(references).issubset(recorded_references):
                raise ValueError(f"Evidence for {component!r} must use recorded trace references.")

        for component, value in components.items():
            if value < 0.0 and not component_evidence.get(component):
                raise ValueError(
                    f"Negative reward component {component!r} requires observable evidence."
                )
            if value < 0.0 and any(
                not reference.is_observable for reference in component_evidence[component]
            ):
                raise ValueError(
                    f"Negative reward component {component!r} requires an observable source kind."
                )

        object.__setattr__(self, "prompt_token_ids", prompt_token_ids)
        object.__setattr__(self, "response_token_ids", response_token_ids)
        for field_name, values in numeric_sequences.items():
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "reward_total", reward_total)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "reward_components", MappingProxyType(components))
        object.__setattr__(self, "reward_component_evidence", MappingProxyType(component_evidence))
        object.__setattr__(self, "trace_references", trace_references)
        object.__setattr__(
            self,
            "sampling_parameters",
            MappingProxyType(sampling_parameters),
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
                component: [reference.to_dict() for reference in references]
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
            "trace_references": [reference.to_dict() for reference in self.trace_references],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Trajectory":
        """Restore a trajectory from its JSON representation without inventing values."""
        reward_components = _mapping(data.get("reward_components"), "reward_components")
        components: dict[str, float] = {}
        for key, value in reward_components.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Expected numeric reward component values.")
            components[key] = float(value)
        return cls(
            trajectory_id=_required_string(data.get("trajectory_id"), "trajectory_id"),
            case_id=_optional_string(data.get("case_id"), "case_id"),
            prompt=_required_string(data.get("prompt"), "prompt"),
            response=_required_string(data.get("response"), "response"),
            prompt_token_ids=_optional_int_tuple(data.get("prompt_token_ids"), "prompt_token_ids"),
            response_token_ids=_optional_int_tuple(
                data.get("response_token_ids"),
                "response_token_ids",
            ),
            old_log_probs=_optional_float_tuple(data.get("old_log_probs"), "old_log_probs"),
            reference_log_probs=_optional_float_tuple(
                data.get("reference_log_probs"),
                "reference_log_probs",
            ),
            value_predictions=_optional_float_tuple(
                data.get("value_predictions"),
                "value_predictions",
            ),
            reward_total=_optional_float(data.get("reward_total"), "reward_total"),
            reward_components=components,
            reward_component_evidence=_component_evidence(
                data.get("reward_component_evidence"),
                restore=True,
            ),
            advantage=_optional_float_tuple(data.get("advantage"), "advantage"),
            returns=_optional_float_tuple(data.get("return"), "return"),
            sampling_parameters=_mapping(data.get("sampling_parameters"), "sampling_parameters"),
            backend_name=_required_string(data.get("backend_name", ""), "backend_name"),
            backend_version=_required_string(data.get("backend_version", ""), "backend_version"),
            model_identifier=_required_string(
                data.get("model_identifier", ""),
                "model_identifier",
            ),
            adapter_identifier=_optional_string(
                data.get("adapter_identifier"),
                "adapter_identifier",
            ),
            policy_version=_optional_string(data.get("policy_version"), "policy_version"),
            seed=_optional_int(data.get("seed"), "seed"),
            trace_references=_evidence_reference_tuple(
                data.get("trace_references"),
                "trace_references",
                restore=True,
            ),
        )

    @staticmethod
    def _optional_list(values: Sequence[int] | Sequence[float] | None) -> list[object] | None:
        """Convert optional immutable numeric sequences to JSON arrays."""
        if values is None:
            return None
        return list(values)


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
