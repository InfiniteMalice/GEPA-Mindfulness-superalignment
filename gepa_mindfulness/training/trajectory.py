"""Backend-neutral, JSON-serializable reinforcement-learning trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.evidence import EvidenceSourceKind as EvidenceSourceKind
from gepa_mindfulness.core.reward_integrity import COMPONENT_NAMES
from gepa_mindfulness.core.reward_provenance import (
    PublicRationaleComparisonEvidence,
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)


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


def _optional_sha256(value: object | None, field_name: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Expected a canonical lowercase SHA-256 digest for {field_name}.")
    return value


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


def _string_tuple(value: object | None, field_name: str) -> tuple[str, ...]:
    """Return an immutable array of legacy diagnostic string references."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected an array or null for {field_name}.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected string values for {field_name}.")
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


def _trusted_evaluator_contract(value: object) -> TrustedEvaluatorContract:
    """Restore the exact identity fields of one trusted evaluator contract."""
    data = _mapping(value, "reward provenance evaluator")
    expected_fields = {"evaluator_id", "evaluator_version", "contract_id"}
    if set(data) != expected_fields:
        raise ValueError(
            "Trusted evaluator contract requires exactly evaluator_id, evaluator_version, "
            "and contract_id."
        )
    return TrustedEvaluatorContract(
        evaluator_id=_required_string(data["evaluator_id"], "evaluator_id"),
        evaluator_version=_required_string(data["evaluator_version"], "evaluator_version"),
        contract_id=_required_string(data["contract_id"], "contract_id"),
    )


def _public_rationale_comparison(value: object) -> PublicRationaleComparisonEvidence:
    """Restore every role in a structured public-rationale comparison."""
    data = _mapping(value, "public_rationale_comparison")
    expected_fields = {
        "public_rationale",
        "committed_prediction",
        "selected_action",
        "observed_outcome",
    }
    if set(data) != expected_fields:
        raise ValueError("Public rationale comparison requires exactly four evidence roles.")
    return PublicRationaleComparisonEvidence(
        public_rationale=EvidenceReference.from_dict(data["public_rationale"]),
        committed_prediction=EvidenceReference.from_dict(data["committed_prediction"]),
        selected_action=EvidenceReference.from_dict(data["selected_action"]),
        observed_outcome=EvidenceReference.from_dict(data["observed_outcome"]),
    )


def _reward_provenance_from_dict(value: object) -> RewardProvenance:
    """Restore one exclusive provenance route without filling absent route fields."""
    data = _mapping(value, "reward component provenance")
    common_fields = {"component_name", "verification_method", "route"}
    if not common_fields.issubset(data):
        raise ValueError(
            "Reward provenance requires component_name, verification_method, and route."
        )
    route_value = _required_string(data["route"], "route")
    try:
        route = VerificationRoute(route_value)
    except ValueError as error:
        raise ValueError(f"Unknown reward provenance route {route_value!r}.") from error

    component_name = _required_string(data["component_name"], "component_name")
    verification_method = _required_string(data["verification_method"], "verification_method")
    if route is VerificationRoute.OBSERVABLE_EVIDENCE:
        allowed_fields = common_fields | {"evidence_refs", "public_rationale_comparison"}
        if "evidence_refs" not in data or not set(data).issubset(allowed_fields):
            raise ValueError("Observable reward provenance has invalid or missing route fields.")
        comparison = (
            _public_rationale_comparison(data["public_rationale_comparison"])
            if "public_rationale_comparison" in data
            else None
        )
        return RewardProvenance(
            component_name=component_name,
            verification_method=verification_method,
            route=route,
            evidence_refs=_evidence_reference_tuple(
                data["evidence_refs"],
                "reward provenance evidence_refs",
                restore=True,
            ),
            public_rationale_comparison=comparison,
        )

    if set(data) != common_fields | {"evaluator"}:
        raise ValueError("Trusted-evaluator reward provenance requires exactly evaluator fields.")
    return RewardProvenance(
        component_name=component_name,
        verification_method=verification_method,
        route=route,
        evaluator=_trusted_evaluator_contract(data["evaluator"]),
    )


def _reward_provenance_to_dict(provenance: RewardProvenance) -> dict[str, object]:
    """Serialize only the fields carried by the selected provenance route."""
    data: dict[str, object] = {
        "component_name": provenance.component_name,
        "verification_method": provenance.verification_method,
        "route": provenance.route.value,
    }
    if provenance.route is VerificationRoute.OBSERVABLE_EVIDENCE:
        data["evidence_refs"] = [reference.to_dict() for reference in provenance.evidence_refs]
        comparison = provenance.public_rationale_comparison
        if comparison is not None:
            data["public_rationale_comparison"] = {
                "public_rationale": comparison.public_rationale.to_dict(),
                "committed_prediction": comparison.committed_prediction.to_dict(),
                "selected_action": comparison.selected_action.to_dict(),
                "observed_outcome": comparison.observed_outcome.to_dict(),
            }
        return data

    evaluator = provenance.evaluator
    if evaluator is None:
        raise ValueError("Trusted-evaluator reward provenance requires an evaluator contract.")
    data["evaluator"] = {
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_version": evaluator.evaluator_version,
        "contract_id": evaluator.contract_id,
    }
    return data


def _component_provenance(
    value: object | None,
    *,
    restore: bool = False,
) -> Mapping[str, RewardProvenance]:
    """Copy or restore the component-keyed reward provenance mapping."""
    raw_provenance = _mapping(value, "reward_component_provenance")
    provenance_by_component: dict[str, RewardProvenance] = {}
    for component, provenance in raw_provenance.items():
        if restore:
            try:
                parsed = _reward_provenance_from_dict(provenance)
            except ValueError as error:
                raise ValueError(
                    f"Invalid reward_component_provenance.{component}: {error}"
                ) from error
        elif isinstance(provenance, RewardProvenance):
            parsed = provenance
        else:
            raise ValueError(
                f"Expected RewardProvenance for reward_component_provenance.{component}."
            )
        provenance_by_component[component] = parsed
    return provenance_by_component


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
    adapter_sha256: str | None = None
    policy_version: str | None = None
    seed: int | None = None
    trace_references: tuple[str, ...] = ()
    evidence_references: tuple[EvidenceReference, ...] = field(default=(), kw_only=True)
    reward_component_evidence: Mapping[str, tuple[EvidenceReference, ...]] = field(
        default_factory=dict,
        kw_only=True,
    )
    reward_component_provenance: Mapping[str, RewardProvenance] = field(
        default_factory=dict,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        """Require evidence for negative rewards and provenance for nonzero integrity components."""
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
        adapter_sha256 = _optional_sha256(self.adapter_sha256, "adapter_sha256")

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
        trace_references = _string_tuple(self.trace_references, "trace_references")
        evidence_references = _evidence_reference_tuple(
            self.evidence_references,
            "evidence_references",
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

        recorded_references = set(evidence_references)
        component_evidence = _component_evidence(self.reward_component_evidence)
        for component, references in component_evidence.items():
            if component not in components:
                raise ValueError(
                    f"Evidence was provided for unknown reward component {component!r}."
                )
            if not set(references).issubset(recorded_references):
                raise ValueError(
                    f"Evidence for {component!r} must use recorded evidence references."
                )

        component_provenance = _component_provenance(self.reward_component_provenance)
        for component, provenance in component_provenance.items():
            if component not in components:
                raise ValueError(
                    f"Provenance was provided for unknown reward component {component!r}."
                )
            if provenance.component_name != component:
                raise ValueError(f"Provenance for {component!r} has a different component_name.")
            if provenance.route is VerificationRoute.OBSERVABLE_EVIDENCE and not set(
                provenance.evidence_refs
            ).issubset(recorded_references):
                raise ValueError(
                    f"Provenance for {component!r} must use recorded evidence references."
                )

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
            if (
                component in COMPONENT_NAMES
                and value != 0.0
                and component not in component_provenance
            ):
                raise ValueError(
                    f"Nonzero reward-integrity component {component!r} requires reward provenance."
                )

        object.__setattr__(self, "prompt_token_ids", prompt_token_ids)
        object.__setattr__(self, "response_token_ids", response_token_ids)
        for field_name, values in numeric_sequences.items():
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "reward_total", reward_total)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "adapter_sha256", adapter_sha256)
        object.__setattr__(self, "reward_components", MappingProxyType(components))
        object.__setattr__(self, "reward_component_evidence", MappingProxyType(component_evidence))
        object.__setattr__(
            self,
            "reward_component_provenance",
            MappingProxyType(component_provenance),
        )
        object.__setattr__(self, "trace_references", trace_references)
        object.__setattr__(self, "evidence_references", evidence_references)
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
        data: dict[str, object] = {
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
            "adapter_sha256": self.adapter_sha256,
            "policy_version": self.policy_version,
            "seed": self.seed,
            "trace_references": list(self.trace_references),
        }
        if self.evidence_references:
            data["evidence_references"] = [
                reference.to_dict() for reference in self.evidence_references
            ]
        if self.reward_component_provenance:
            data["reward_component_provenance"] = {
                component: _reward_provenance_to_dict(provenance)
                for component, provenance in self.reward_component_provenance.items()
            }
        return data

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
            reward_component_provenance=_component_provenance(
                data.get("reward_component_provenance"),
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
            adapter_sha256=_optional_sha256(data.get("adapter_sha256"), "adapter_sha256"),
            policy_version=_optional_string(data.get("policy_version"), "policy_version"),
            seed=_optional_int(data.get("seed"), "seed"),
            trace_references=_string_tuple(data.get("trace_references"), "trace_references"),
            evidence_references=_evidence_reference_tuple(
                data.get("evidence_references"),
                "evidence_references",
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
