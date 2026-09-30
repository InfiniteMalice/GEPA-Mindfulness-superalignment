"""Opt-in scalar random-walk diagnostics with causal validation and explicit mismatch handling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from math import sqrt
from typing import Any

from mindful_trace_gepa.confidence import ConfidenceSource
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .epistemic_reconciliation import (
    EpistemicReconciliation,
    OutcomeMeasurementBinding,
    make_epistemic_reconciliation_event,
)
from .epistemic_state import (
    DiagonalState,
    EpistemicMeasurement,
    EpistemicStateEstimate,
    InnovationRecord,
    MismatchStatus,
    UncertaintyUpdateRecord,
    _enum,
    _nonnegative,
    _number,
    _snapshot,
    _text,
    _unit,
)
from .state import _require_exact_mapping, parse_rfc3339_datetime

ESTIMATOR_VERSION = "scalar-random-walk-v1"


@dataclass(frozen=True, slots=True)
class ScalarEstimatorConfig:
    """Numerical model declarations, not evidence that semantic uncertainty obeys this model."""

    enabled: bool = False
    independent_noise: bool = False
    measurement_source: ConfidenceSource = ConfidenceSource.TOOL_RESULT
    measurement_provenance_ref: str = "scalar-monitor-v1"
    process_variance: float = 0.1
    process_variance_floor: float = 0.01
    max_process_variance: float = 1_000_000.0
    process_noise_growth: float = 4.0
    measurement_noise_growth: float = 4.0
    max_measurement_noise_scale: float = 100.0
    noise_decay: float = 2.0
    innovation_gate: float = 3.0
    regime_shift_after: int = 3
    insufficient_model_after: int = 6
    uncertainty_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("enabled", "independent_noise"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a built-in bool")
        if self.enabled and not self.independent_noise:
            raise ValueError("enabled estimator requires an explicit independent_noise declaration")
        _enum(self.measurement_source, ConfidenceSource, "measurement_source")
        _text(self.measurement_provenance_ref, "measurement_provenance_ref")
        for name in (
            "process_variance",
            "process_variance_floor",
            "max_process_variance",
            "process_noise_growth",
            "measurement_noise_growth",
            "max_measurement_noise_scale",
            "noise_decay",
            "innovation_gate",
            "uncertainty_scale",
        ):
            value = _nonnegative(getattr(self, name), name)
            if name != "process_variance" and value == 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "process_noise_growth",
            "measurement_noise_growth",
            "max_measurement_noise_scale",
            "noise_decay",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if max(self.process_variance, self.process_variance_floor) > self.max_process_variance:
            raise ValueError("process variance and floor must not exceed max_process_variance")
        for name in ("regime_shift_after", "insufficient_model_after"):
            value = getattr(self, name)
            if type(value) is not int or not 2 <= value <= 9_007_199_254_740_991:
                raise ValueError(f"{name} must be an integer from 2 through the JSON-safe limit")
        if self.insufficient_model_after <= self.regime_shift_after:
            raise ValueError("insufficient_model_after must exceed regime_shift_after")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact model configuration for audit/replay."""
        self.__post_init__()
        return asdict(self) | {"measurement_source": self.measurement_source.value}

    @classmethod
    def from_dict(cls, data: object) -> ScalarEstimatorConfig:
        """Reject unknown/missing fields and malformed configuration numbers."""
        values: dict[str, Any] = dict(
            _require_exact_mapping(data, {f.name for f in fields(cls)}, cls.__name__)
        )
        if type(values["measurement_source"]) is not str:
            raise ValueError("measurement_source must be an exact enum string")
        values["measurement_source"] = ConfidenceSource(values["measurement_source"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    """Diagnostic result; the event is returned to the host without writing or authorizing it."""

    reconciliation: EpistemicReconciliation
    event: EventEnvelope
    configuration: ScalarEstimatorConfig
    assimilated: bool
    gain: float
    process_variance: float
    measurement_variance: float
    outlier_count: int

    @property
    def estimate(self) -> EpistemicStateEstimate:
        """Return the computed posterior in the existing state contract."""
        return self.reconciliation.update.posterior_state

    @property
    def innovation(self) -> InnovationRecord:
        """Return the recorded gate residual with its pre-adaptation variance."""
        return self.reconciliation.bindings[0].innovation


class ScalarTemporalEstimator:
    """One calibrated scalar channel; explicit calls mutate only this instance's private state.

    No time-unit inference is made: Q is variance per observation step. Each returned event must
    remain in the host's next supplied history. Invalid steps leave the instance unchanged.
    """

    def __init__(
        self,
        initial_state: EpistemicStateEstimate,
        config: ScalarEstimatorConfig | None = None,
    ) -> None:
        supplied = config if config is not None else ScalarEstimatorConfig()
        if type(supplied) is not ScalarEstimatorConfig:
            raise ValueError("config must be ScalarEstimatorConfig")
        self._config = ScalarEstimatorConfig.from_dict(supplied.to_dict())
        self._estimate = _snapshot(initial_state, EpistemicStateEstimate)
        state = self._estimate.state
        if state is None or len(state.values) != 1 or state.variances is None:
            raise ValueError("scalar estimator requires one dimension with available covariance")
        self._q = self._config.process_variance
        self._r_scale = 1.0
        self._outliers = 0
        self._history_count = 0
        self._history_digest: str | None = None
        self._last_timestamp: str | None = None
        self._measurement_ids: set[str] = set()
        self._estimate_ids = {initial_state.estimate_id}
        self._evidence_ids = {ref.reference_id for ref in initial_state.evidence_refs}

    @property
    def estimate(self) -> EpistemicStateEstimate:
        """Return a detached snapshot; callers cannot mutate the estimator through it."""
        return _snapshot(self._estimate, EpistemicStateEstimate)

    def reconcile(
        self,
        events: Sequence[EventEnvelope],
        *,
        measurement: EpistemicMeasurement,
        binding: OutcomeMeasurementBinding,
        prediction_event_id: str,
        observation_event_id: str,
        verification_event_ids: tuple[str, ...],
        update_id: str,
        estimate_id: str,
        event_id: str,
        timestamp: str,
        **metadata: Any,
    ) -> EstimatorResult | None:
        """Calculate and validate one step, then atomically retain its state in memory.

        With enabled=False return None before inspecting inputs. Enabled calls raise ValueError
        for unsupported telemetry, replay, invalid ancestry or nonfinite arithmetic. A gate failure
        returns a rejected-mean diagnostic update, with explicit inflated noise and mismatch status.
        """
        cfg = self._config
        if not cfg.enabled:
            return None
        history = tuple(events)
        validate_action_bound_sequence(history)
        self._validate_history(history, prediction_event_id)
        measurement = _snapshot(measurement, EpistemicMeasurement)
        if type(binding) is not OutcomeMeasurementBinding:
            raise ValueError("binding must be OutcomeMeasurementBinding")
        binding = OutcomeMeasurementBinding.from_dict(binding.to_dict())
        state = self._estimate.state
        assert state is not None and state.variances is not None
        if (
            measurement.context != self._estimate.context
            or measurement.representation_id != state.representation_id
            or measurement.target_dimension != state.dimensions[0]
        ):
            raise ValueError("measurement context, representation and dimension must match state")
        if (
            measurement.source is not cfg.measurement_source
            or cfg.measurement_provenance_ref not in measurement.provenance
        ):
            raise ValueError("measurement must match the configured source and provenance contract")
        if measurement.variance is None or measurement.variance <= 0:
            raise ValueError("measurement requires a positive finite variance")
        evidence_ids = {ref.reference_id for ref in measurement.evidence_refs}
        if (
            measurement.measurement_id in self._measurement_ids
            or evidence_ids & self._evidence_ids
            or estimate_id in self._estimate_ids
        ):
            raise ValueError("measurement, evidence and estimate identities must not be reused")
        if binding.innovation.predicted_measurement != state.values[0]:
            raise ValueError("committed prediction must equal the scalar random-walk prior mean")
        observed = next((e for e in history if e.event_id == observation_event_id), None)
        if observed is None or observed.event_type != StructuredEventType.OUTCOME_OBSERVED.value:
            raise ValueError("estimator requires an earlier outcome_observed")

        predicted_variance = _nonnegative(state.variances[0] + self._q, "predicted variance")
        effective_r = _number(measurement.variance * self._r_scale, "measurement variance")
        innovation_variance = _number(predicted_variance + effective_r, "innovation variance")
        residual = _number(binding.innovation.residual, "residual")
        normalized = _number(residual / sqrt(innovation_variance), "normalized innovation")
        latched = self._outliers >= cfg.insufficient_model_after
        rejected = latched or abs(normalized) > cfg.innovation_gate
        count = min(self._outliers + 1, cfg.insufficient_model_after) if rejected else 0
        status = MismatchStatus.NONE
        q, r_scale = self._q, self._r_scale
        if rejected:
            q = _grow(
                max(q, cfg.process_variance_floor),
                cfg.process_noise_growth,
                cfg.max_process_variance,
            )
            r_scale = _grow(r_scale, cfg.measurement_noise_growth, cfg.max_measurement_noise_scale)
            effective_r = _number(measurement.variance * r_scale, "adapted measurement variance")
            mean = state.values[0]
            variance = _nonnegative(state.variances[0] + q, "inflated variance")
            gain = 0.0
            status = (
                MismatchStatus.INSUFFICIENT_MODEL
                if latched or count >= cfg.insufficient_model_after
                else (
                    MismatchStatus.REGIME_SHIFT_SUSPECTED
                    if count >= cfg.regime_shift_after
                    else MismatchStatus.MODEL_MISMATCH
                )
            )
            next_q, next_r_scale = q, r_scale
        else:
            gain = _unit(predicted_variance / innovation_variance, "gain")
            mean = _number(
                (1 - gain) * state.values[0] + gain * binding.innovation.actual_measurement,
                "posterior mean",
            )
            # Joseph form avoids subtracting nearly equal covariance values.
            variance = _nonnegative(
                (1 - gain) ** 2 * predicted_variance + gain**2 * effective_r, "posterior variance"
            )
            next_q = max(cfg.process_variance, q / cfg.noise_decay)
            next_r_scale = max(1.0, r_scale / cfg.noise_decay)
        config_ref = f"{ESTIMATOR_VERSION}:{_digest(cfg.to_dict())}"
        evidence = tuple(dict.fromkeys((*self._estimate.evidence_refs, *measurement.evidence_refs)))
        provenance = tuple(
            dict.fromkeys((*self._estimate.provenance, *measurement.provenance, config_ref))
        )
        posterior = EpistemicStateEstimate(
            estimate_id=estimate_id,
            context=self._estimate.context,
            estimator_version=ESTIMATOR_VERSION,
            world_uncertainty=_uncertainty(variance, cfg.uncertainty_scale),
            model_uncertainty=None,
            monitor_uncertainty=_uncertainty(effective_r, cfg.uncertainty_scale),
            state=DiagonalState(state.representation_id, state.dimensions, (mean,), (variance,)),
            action_id=observed.action_id,
            prediction_commit_id=binding.innovation.prediction_commit_id,
            evidence_refs=evidence,
            provenance=provenance,
        )
        innovation = replace(
            binding.innovation,
            innovation_variance=innovation_variance,
            normalization_basis=config_ref,
            mismatch_status=status,
        )
        update = UncertaintyUpdateRecord(
            update_id=update_id,
            prior_state=self._estimate,
            measurements=(measurement,),
            posterior_state=posterior,
            update_method=config_ref,
            model_mismatch=status,
            evidence_refs=evidence,
            provenance=provenance,
        )
        reconciliation = EpistemicReconciliation(
            update=update,
            prediction_event_id=prediction_event_id,
            observation_event_id=observation_event_id,
            verification_event_ids=verification_event_ids,
            bindings=(replace(binding, innovation=innovation),),
        )
        event = make_epistemic_reconciliation_event(
            reconciliation,
            event_id=event_id,
            timestamp=timestamp,
            **metadata,
        )
        full_history = (*history, event)
        validate_action_bound_sequence(full_history)
        digest = _digest([e.to_dict() for e in full_history])
        result = EstimatorResult(
            reconciliation,
            event,
            ScalarEstimatorConfig.from_dict(cfg.to_dict()),
            not rejected,
            gain,
            q,
            effective_r,
            count,
        )
        # Commit only after arithmetic, records, causal checks and serialization succeed.
        self._estimate = _snapshot(posterior, EpistemicStateEstimate)
        self._q, self._r_scale, self._outliers = next_q, next_r_scale, count
        self._history_count, self._history_digest = len(full_history), digest
        self._last_timestamp = timestamp
        self._measurement_ids.add(measurement.measurement_id)
        self._estimate_ids.add(estimate_id)
        self._evidence_ids.update(evidence_ids)
        return result

    def _validate_history(self, events: tuple[EventEnvelope, ...], prediction_id: str) -> None:
        if self._history_digest is not None:
            prefix = [e.to_dict() for e in events[: self._history_count]]
            if len(events) < self._history_count or _digest(prefix) != self._history_digest:
                raise ValueError("history must retain the complete previously reconciled prefix")
        prediction_index = next(
            (i for i, e in enumerate(events) if e.event_id == prediction_id), -1
        )
        if prediction_index < self._history_count:
            raise ValueError("prediction must be new and follow the previously reconciled history")
        if self._last_timestamp is not None:
            current = parse_rfc3339_datetime(events[prediction_index].timestamp, "timestamp")
            previous = parse_rfc3339_datetime(self._last_timestamp, "timestamp")
            if current < previous:
                raise ValueError("prediction timestamp must not precede previous reconciliation")


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _grow(value: float, factor: float, ceiling: float) -> float:
    return min(ceiling, min(value, ceiling / factor) * factor)


def _uncertainty(variance: float, scale: float) -> float:
    if variance >= scale:
        return _unit(1 / (1 + scale / variance), "normalized uncertainty")
    ratio = variance / scale
    return _unit(ratio / (1 + ratio), "normalized uncertainty")
