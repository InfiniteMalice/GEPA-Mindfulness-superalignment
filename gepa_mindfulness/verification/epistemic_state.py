"""Inert temporal epistemic contracts; records confer neither reward nor authority.

These snapshots validate declared data, not evidence truth or causal chronology. No runtime
producer, estimator, event adapter or persistence operation is installed by importing this module.
"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from math import isfinite, sqrt
from typing import Any, TypeVar, cast

# Local
from mindful_trace_gepa.confidence import ConfidenceSource
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
from mindful_trace_gepa.logging_schema import EventEnvelope

from ..core.evidence import EvidenceReference
from .state import _require_exact_mapping, _restore_evidence_refs, _snapshot_evidence_refs

SCHEMA_VERSION = "epistemic-state-v1"
_RecordType = TypeVar("_RecordType", bound="_Record")


class Availability(str, Enum):
    """Numeric availability only; AVAILABLE does not mean verified or trusted."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class MismatchStatus(str, Enum):
    """Producer-declared diagnostic finding, never an inference about intent."""

    UNASSESSED = "unassessed"
    NONE = "none"
    MODEL_MISMATCH = "model_mismatch"
    REGIME_SHIFT_SUSPECTED = "regime_shift_suspected"
    INSUFFICIENT_MODEL = "insufficient_model"


class CorrelationTreatment(str, Enum):
    """Declared treatment of dependence; these labels do not execute a fusion algorithm."""

    UNRESOLVED_CORRELATION = "unresolved_correlation"
    KNOWN_COVARIANCE = "known_covariance"
    COVARIANCE_INTERSECTION = "covariance_intersection"
    CONSERVATIVE_BOUND = "conservative_bound"


@dataclass(frozen=True, slots=True)
class _Record:
    """Exact, versioned serialization shared only by these new record contracts."""

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON data after revalidating the current record fields."""
        self.__post_init__()
        return {"schema_version": SCHEMA_VERSION} | {
            item.name: _encode(getattr(self, item.name)) for item in fields(self)
        }

    def __post_init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_dict(cls: type[_RecordType], data: object) -> _RecordType:
        """Reject missing/unknown fields and restore explicitly typed nested records."""
        values = dict(
            _require_exact_mapping(
                data,
                {item.name for item in fields(cls)} | {"schema_version"},
                cls.__name__,
            )
        )
        version = values.pop("schema_version")
        if type(version) is not str or version != SCHEMA_VERSION:
            raise ValueError("unsupported epistemic schema_version")
        for name, value in tuple(values.items()):
            if name == "evidence_refs":
                values[name] = _restore_evidence_refs(value, cls.__name__)
            elif name in _NESTED_RECORDS and value is not None:
                values[name] = _NESTED_RECORDS[name].from_dict(value)
            elif name == "measurements":
                values[name] = tuple(EpistemicMeasurement.from_dict(v) for v in _array(value, name))
            elif name == "system":
                system = _require_exact_mapping(
                    value,
                    {"model_version", "harness_version"},
                    "system",
                )
                values[name] = EvaluatedSystemVersion(
                    cast(str, system["model_version"]),
                    cast(str, system["harness_version"]),
                )
            elif name in _ENUM_FIELDS:
                if type(value) is not str:
                    raise ValueError(f"{name} must be an exact enum string")
                values[name] = _ENUM_FIELDS[name](value)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class EpistemicContext(_Record):
    """Existing evaluation-unit coordinates with the existing system-version record."""

    run_id: str
    repeat_id: int | None
    system: EvaluatedSystemVersion

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        if self.repeat_id is not None and (
            type(self.repeat_id) is not int or not 0 <= self.repeat_id <= 9_007_199_254_740_991
        ):
            raise ValueError("repeat_id must be a nonnegative JSON-safe integer or null")
        if type(self.system) is not EvaluatedSystemVersion:
            raise ValueError("system must be EvaluatedSystemVersion")
        object.__setattr__(
            self,
            "system",
            EvaluatedSystemVersion(
                self.system.model_version,
                self.system.harness_version,
            ),
        )

    def validate_event(self, event: EventEnvelope) -> None:
        """Check only run/repeat/version identity; this does not validate causal references."""
        self.__post_init__()
        if type(event) is not EventEnvelope:
            raise ValueError("event must be EventEnvelope")
        expected = (
            self.run_id,
            self.repeat_id,
            self.system.model_version,
            self.system.harness_version,
        )
        actual = (event.run_id, event.repeat_id, event.model_version, event.harness_version)
        if any(type(a) is not type(b) or a != b for a, b in zip(actual, expected)):
            raise ValueError("event evaluation identity must match epistemic context")


@dataclass(frozen=True, slots=True)
class DiagonalState(_Record):
    """Numerical state in named units; variances are diagonal entries in squared units.

    Full covariance matrices are unsupported. A null variance vector means unavailable, not an
    identity matrix, zero covariance or independence between measurement sources.
    """

    representation_id: str
    dimensions: tuple[str, ...]
    values: tuple[float, ...]
    variances: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _text(self.representation_id, "representation_id")
        dimensions = _strings(self.dimensions, "dimensions", required=True)
        values = tuple(_number(v, "values") for v in _array(self.values, "values"))
        if len(values) != len(dimensions):
            raise ValueError("values must match dimensions")
        variances = self.variances
        if variances is not None:
            variances = tuple(_nonnegative(v, "variances") for v in _array(variances, "variances"))
            if len(variances) != len(dimensions):
                raise ValueError("diagonal variances must match dimensions")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "variances", variances)


@dataclass(frozen=True, slots=True, kw_only=True)
class _EvidenceRecord(_Record):
    evidence_refs: tuple[EvidenceReference, ...]
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", _snapshot_evidence_refs(self.evidence_refs))
        object.__setattr__(
            self, "provenance", _strings(self.provenance, "provenance", required=True)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EpistemicStateEstimate(_EvidenceRecord):
    """Separate normalized uncertainty diagnostics and optional numerical state.

    The estimator version identifies the producer's normalization contract. Values are not
    automatically probabilities, covariance entries or calibrated estimates of correctness.
    """

    estimate_id: str
    context: EpistemicContext
    estimator_version: str
    world_uncertainty: float | None
    model_uncertainty: float | None
    monitor_uncertainty: float | None
    state: DiagonalState | None = None
    action_id: str | None = None
    prediction_commit_id: str | None = None
    status: Availability = Availability.AVAILABLE

    def __post_init__(self) -> None:
        _EvidenceRecord.__post_init__(self)
        _text(self.estimate_id, "estimate_id")
        _text(self.estimator_version, "estimator_version")
        object.__setattr__(self, "context", _snapshot(self.context, EpistemicContext))
        _enum(self.status, Availability, "status")
        for name in ("action_id", "prediction_commit_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        numbers = []
        for name in ("world_uncertainty", "model_uncertainty", "monitor_uncertainty"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit(value, name))
                numbers.append(value)
        if self.state is not None:
            object.__setattr__(self, "state", _snapshot(self.state, DiagonalState))
        available = bool(numbers) or self.state is not None
        if available != (self.status is Availability.AVAILABLE):
            raise ValueError(
                "unavailable estimates require null numbers; available estimates need data"
            )
        if available and not self.evidence_refs:
            raise ValueError("available estimates require evidence_refs")


@dataclass(frozen=True, slots=True, kw_only=True)
class EpistemicMeasurement(_EvidenceRecord):
    """One declared scalar measurement; the source label does not certify verification."""

    measurement_id: str
    context: EpistemicContext
    source: ConfidenceSource
    target_dimension: str
    representation_id: str
    value: float | None
    uncertainty: float | None
    variance: float | None = None
    correlation_group: str | None = None
    status: Availability = Availability.AVAILABLE

    def __post_init__(self) -> None:
        _EvidenceRecord.__post_init__(self)
        for name in ("measurement_id", "target_dimension", "representation_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "context", _snapshot(self.context, EpistemicContext))
        _enum(self.source, ConfidenceSource, "source")
        _enum(self.status, Availability, "status")
        if self.correlation_group is not None:
            _text(self.correlation_group, "correlation_group")
        for name, validate in (
            ("value", _number),
            ("uncertainty", _unit),
            ("variance", _nonnegative),
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, validate(getattr(self, name), name))
        if self.status is Availability.UNAVAILABLE:
            if any(v is not None for v in (self.value, self.uncertainty, self.variance)):
                raise ValueError("unavailable measurements require null numbers, not zero")
        elif self.value is None or not self.evidence_refs:
            raise ValueError("available measurements require value and evidence_refs")


@dataclass(frozen=True, slots=True, kw_only=True)
class InnovationRecord(_EvidenceRecord):
    """Scalar comparison with unresolved event references; no causal or truth certification."""

    innovation_id: str
    context: EpistemicContext
    prediction_commit_id: str
    observation_id: str
    predicted_measurement: float
    actual_measurement: float
    mismatch_status: MismatchStatus = MismatchStatus.UNASSESSED
    innovation_variance: float | None = None
    normalization_basis: str | None = None

    def __post_init__(self) -> None:
        _EvidenceRecord.__post_init__(self)
        for name in ("innovation_id", "prediction_commit_id", "observation_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "context", _snapshot(self.context, EpistemicContext))
        _enum(self.mismatch_status, MismatchStatus, "mismatch_status")
        for name in ("predicted_measurement", "actual_measurement"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        _number(self.residual, "residual")
        if not self.evidence_refs:
            raise ValueError("innovation requires evidence_refs")
        if (self.innovation_variance is None) != (self.normalization_basis is None):
            raise ValueError(
                "normalization requires both innovation_variance and normalization_basis"
            )
        if self.innovation_variance is not None:
            variance = _number(self.innovation_variance, "innovation_variance")
            if variance <= 0:
                raise ValueError("innovation_variance must be positive")
            _text(self.normalization_basis, "normalization_basis")
            object.__setattr__(self, "innovation_variance", variance)
            _number(self.normalized_innovation, "normalized_innovation")

    @property
    def residual(self) -> float:
        """Signed difference actual minus predicted, in the measurement's units."""
        return self.actual_measurement - self.predicted_measurement

    @property
    def normalized_innovation(self) -> float | None:
        """Signed residual divided by declared standard deviation, when a basis exists."""
        if self.innovation_variance is None:
            return None
        return self.residual / sqrt(self.innovation_variance)


@dataclass(frozen=True, slots=True, kw_only=True)
class UncertaintyUpdateRecord(_EvidenceRecord):
    """A declared update with compatible snapshots; no estimator or persistence operation."""

    update_id: str
    prior_state: EpistemicStateEstimate
    measurements: tuple[EpistemicMeasurement, ...]
    posterior_state: EpistemicStateEstimate
    update_method: str
    correlation_treatment: CorrelationTreatment = CorrelationTreatment.UNRESOLVED_CORRELATION
    model_mismatch: MismatchStatus = MismatchStatus.UNASSESSED
    unresolved_hypotheses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _EvidenceRecord.__post_init__(self)
        _text(self.update_id, "update_id")
        _text(self.update_method, "update_method")
        _enum(self.correlation_treatment, CorrelationTreatment, "correlation_treatment")
        _enum(self.model_mismatch, MismatchStatus, "model_mismatch")
        object.__setattr__(
            self,
            "unresolved_hypotheses",
            _strings(
                self.unresolved_hypotheses,
                "unresolved_hypotheses",
            ),
        )
        prior = _snapshot(self.prior_state, EpistemicStateEstimate)
        posterior = _snapshot(self.posterior_state, EpistemicStateEstimate)
        measurements = tuple(
            _snapshot(v, EpistemicMeasurement)
            for v in _array(
                self.measurements,
                "measurements",
            )
        )
        if not measurements or len({v.measurement_id for v in measurements}) != len(measurements):
            raise ValueError("measurements must be nonempty with unique measurement_id values")
        if prior.estimate_id == posterior.estimate_id:
            raise ValueError("posterior requires a new estimate_id")
        if posterior.context != prior.context or any(
            v.context != prior.context for v in measurements
        ):
            raise ValueError("update context must match prior, measurements and posterior")
        _validate_state_dimensions(prior, posterior, measurements)
        required = set(prior.evidence_refs) | set(posterior.evidence_refs)
        for item in measurements:
            if item.status is Availability.UNAVAILABLE:
                raise ValueError("measurements used in an update must be available")
            required.update(item.evidence_refs)
        if not required.issubset(self.evidence_refs):
            raise ValueError(
                "update evidence_refs must retain prior, posterior and measurement evidence"
            )
        object.__setattr__(self, "prior_state", prior)
        object.__setattr__(self, "posterior_state", posterior)
        object.__setattr__(self, "measurements", measurements)


def _validate_state_dimensions(
    prior: EpistemicStateEstimate,
    posterior: EpistemicStateEstimate,
    measurements: tuple[EpistemicMeasurement, ...],
) -> None:
    states = tuple(v.state for v in (prior, posterior) if v.state is not None)
    if len(states) == 2 and (
        states[0].representation_id != states[1].representation_id
        or states[0].dimensions != states[1].dimensions
    ):
        raise ValueError("prior/posterior representation and dimensions must match")
    for state in states:
        for item in measurements:
            if item.representation_id != state.representation_id:
                raise ValueError("measurement representation must match numerical state")
            if item.target_dimension not in state.dimensions:
                raise ValueError("measurement target_dimension must exist in numerical state")


def _snapshot(value: object, kind: type[_RecordType]) -> _RecordType:
    if type(value) is not kind:
        raise ValueError(f"expected exact {kind.__name__}")
    return kind.from_dict(value.to_dict())


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonblank string without surrounding whitespace")
    return value


def _array(value: object, name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an array")
    return tuple(cast(list[Any] | tuple[Any, ...], value))


def _strings(value: object, name: str, *, required: bool = False) -> tuple[str, ...]:
    result = tuple(_text(v, name) for v in _array(value, name))
    if (required and not result) or len(set(result)) != len(result):
        raise ValueError(
            f"{name} must contain unique nonblank strings and cannot omit required data"
        )
    return result


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite built-in number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite built-in number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite built-in number")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _number(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _unit(value: object, name: str) -> float:
    result = _nonnegative(value, name)
    if result > 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _enum(value: object, kind: type[Enum], name: str) -> None:
    if type(value) is not kind:
        raise ValueError(f"{name} must be {kind.__name__}")


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (EvidenceReference, _Record)):
        return value.to_dict()
    if isinstance(value, EvaluatedSystemVersion):
        return {"model_version": value.model_version, "harness_version": value.harness_version}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


_NESTED_RECORDS: dict[str, type[_Record]] = {
    "context": EpistemicContext,
    "state": DiagonalState,
    "prior_state": EpistemicStateEstimate,
    "posterior_state": EpistemicStateEstimate,
}
_ENUM_FIELDS: dict[str, type[Enum]] = {
    "status": Availability,
    "source": ConfidenceSource,
    "mismatch_status": MismatchStatus,
    "model_mismatch": MismatchStatus,
    "correlation_treatment": CorrelationTreatment,
}
