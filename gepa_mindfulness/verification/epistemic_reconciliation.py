"""Bind diagnostic updates to existing action-bound events; confer no reward or authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from mindful_trace_gepa._json_values import thaw_json_mapping
from mindful_trace_gepa.action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
    _make_payload_event,
)
from mindful_trace_gepa.confidence import ConfidenceSource
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .epistemic_state import (
    EpistemicMeasurement,
    EpistemicStateEstimate,
    InnovationRecord,
    UncertaintyUpdateRecord,
    _array,
    _number,
    _snapshot,
    _strings,
    _text,
)
from .interfaces import RelationalVerificationResult, VerificationLevel
from .state import _require_exact_mapping, parse_rfc3339_datetime

SCHEMA_VERSION = "epistemic-reconciliation-v1"
OutcomePath = tuple[str | int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class OutcomeMeasurementBinding:
    """World residual from explicit JSON paths; path semantics are a producer contract."""

    measurement_id: str
    innovation: InnovationRecord
    prediction_path: OutcomePath
    observation_path: OutcomePath
    verifier_event_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.measurement_id, "measurement_id")
        if self.verifier_event_id is not None:
            _text(self.verifier_event_id, "verifier_event_id")
        object.__setattr__(self, "innovation", _snapshot(self.innovation, InnovationRecord))
        for name in ("prediction_path", "observation_path"):
            path = _array(getattr(self, name), name)
            for key in path:
                if type(key) is not str and not (type(key) is int and 0 <= key <= 2**53 - 1):
                    raise ValueError(
                        f"{name} requires string keys or nonnegative safe integer indices"
                    )
            object.__setattr__(self, name, path)

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON after revalidation."""
        self.__post_init__()
        return dict(
            measurement_id=self.measurement_id,
            innovation=self.innovation.to_dict(),
            prediction_path=list(self.prediction_path),
            observation_path=list(self.observation_path),
            verifier_event_id=self.verifier_event_id,
        )

    @classmethod
    def from_dict(cls, data: object) -> OutcomeMeasurementBinding:
        """Restore exactly the fields in the versioned enclosing reconciliation."""
        values: dict[str, Any] = thaw_json_mapping(
            _require_exact_mapping(data, {f.name for f in fields(cls)}, cls.__name__)
        )
        values["innovation"] = InnovationRecord.from_dict(values["innovation"])
        return cls(**values)


@dataclass(frozen=True, slots=True, kw_only=True)
class EpistemicReconciliation:
    """Declared bindings; validate the full event sequence before claiming causal reconciliation."""

    update: UncertaintyUpdateRecord
    prediction_event_id: str
    observation_event_id: str
    verification_event_ids: tuple[str, ...]
    bindings: tuple[OutcomeMeasurementBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "update", _snapshot(self.update, UncertaintyUpdateRecord))
        _text(self.prediction_event_id, "prediction_event_id")
        _text(self.observation_event_id, "observation_event_id")
        object.__setattr__(
            self,
            "verification_event_ids",
            _strings(
                self.verification_event_ids,
                "verification_event_ids",
                required=True,
            ),
        )
        bindings = []
        for binding in _array(self.bindings, "bindings"):
            if type(binding) is not OutcomeMeasurementBinding:
                raise ValueError("bindings require exact OutcomeMeasurementBinding records")
            bindings.append(OutcomeMeasurementBinding.from_dict(binding.to_dict()))
        ids = [b.measurement_id for b in bindings]
        if len(set(ids)) != len(ids) or set(ids) != {
            m.measurement_id for m in self.update.measurements
        }:
            raise ValueError("bindings must cover each update measurement exactly once")
        if len({b.innovation.innovation_id for b in bindings}) != len(bindings):
            raise ValueError("bindings require unique innovation_id values")
        for binding in bindings:
            if binding.innovation.context != self.update.prior_state.context:
                raise ValueError("innovation context must match update context")
            if (
                binding.verifier_event_id is not None
                and binding.verifier_event_id not in self.verification_event_ids
            ):
                raise ValueError("binding verifier must occur in verification_event_ids")
        if len(set(self.parent_event_ids)) != len(self.parent_event_ids):
            raise ValueError("reconciliation requires distinct parent event IDs")
        posterior = self.update.posterior_state
        _text(posterior.action_id, "posterior action_id")
        _text(posterior.prediction_commit_id, "posterior prediction_commit_id")
        object.__setattr__(self, "bindings", tuple(bindings))

    @property
    def parent_event_ids(self) -> tuple[str, ...]:
        """The exact ordered parent references emitted by the adapter."""
        return (self.prediction_event_id, self.observation_event_id, *self.verification_event_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return detached, strictly versioned JSON."""
        self.__post_init__()
        return dict(
            schema_version=SCHEMA_VERSION,
            update=self.update.to_dict(),
            prediction_event_id=self.prediction_event_id,
            observation_event_id=self.observation_event_id,
            verification_event_ids=list(self.verification_event_ids),
            bindings=[b.to_dict() for b in self.bindings],
        )

    @classmethod
    def from_dict(cls, data: object) -> EpistemicReconciliation:
        """Reject missing/unknown fields and unsupported schema versions."""
        names = {f.name for f in fields(cls)} | {"schema_version"}
        values: dict[str, Any] = thaw_json_mapping(
            _require_exact_mapping(data, names, cls.__name__)
        )
        version = values.pop("schema_version")
        if type(version) is not str or version != SCHEMA_VERSION:
            raise ValueError("unsupported reconciliation schema_version")
        values["update"] = UncertaintyUpdateRecord.from_dict(values["update"])
        values["bindings"] = tuple(
            OutcomeMeasurementBinding.from_dict(b) for b in _array(values["bindings"], "bindings")
        )
        return cls(**values)


def make_epistemic_reconciliation_event(
    reconciliation: EpistemicReconciliation,
    **metadata: Any,
) -> EventEnvelope:
    """Wrap declared bindings; callers must still run validate_action_bound_sequence."""
    if type(reconciliation) is not EpistemicReconciliation:
        raise ValueError("expected exact EpistemicReconciliation")
    record = EpistemicReconciliation.from_dict(reconciliation.to_dict())
    context = record.update.prior_state.context
    event = _make_payload_event(
        StructuredEventType.EPISTEMIC_RECONCILIATION,
        record.to_dict(),
        metadata,
        run_id=context.run_id,
        repeat_id=context.repeat_id,
        model_version=context.system.model_version,
        harness_version=context.system.harness_version,
        action_id=record.update.posterior_state.action_id,
        parent_event_ids=record.parent_event_ids,
        evidence_refs=tuple(dict.fromkeys(r.reference_id for r in record.update.evidence_refs)),
    )
    context.validate_event(event)
    return event


def _validate_reconciliation_event(
    event: EventEnvelope,
    seen: Mapping[str, EventEnvelope],
) -> EpistemicReconciliation:
    """Resolve ancestry already validated by the sequence walker; never call standalone."""
    record = EpistemicReconciliation.from_dict(event.payload)
    update = record.update
    context = update.prior_state.context
    context.validate_event(event)
    if event.parent_event_ids != record.parent_event_ids:
        raise ValueError("reconciliation parent_event_ids must match payload")
    expected_refs = tuple(dict.fromkeys(r.reference_id for r in update.evidence_refs))
    if event.evidence_refs != expected_refs:
        raise ValueError("reconciliation evidence_refs must match payload")
    if type(event.action_id) is not str or event.action_id != update.posterior_state.action_id:
        raise ValueError("reconciliation action_id must match posterior")

    def resolve(event_id: str, kind: StructuredEventType) -> EventEnvelope:
        parent = seen.get(event_id)
        if parent is None or parent.event_type != kind.value:
            raise ValueError(f"reconciliation requires an earlier {kind.value}")
        context.validate_event(parent)
        return parent

    prediction_event = resolve(record.prediction_event_id, StructuredEventType.PREDICTION_COMMIT)
    observation_event = resolve(record.observation_event_id, StructuredEventType.OUTCOME_OBSERVED)
    prediction = PredictionCommit(**dict(prediction_event.payload))
    observation = OutcomeObservation(**dict(observation_event.payload))
    execution = resolve(observation_event.parent_event_ids[0], StructuredEventType.ACTION_EXECUTED)
    proposal = resolve(execution.parent_event_ids[0], StructuredEventType.ACTION_PROPOSED)
    action = ActionRecord(**dict(execution.payload))
    if (
        action.prediction_commit_id != prediction.prediction_commit_id
        or update.posterior_state.prediction_commit_id != prediction.prediction_commit_id
        or observation.action_id != event.action_id
    ):
        raise ValueError("reconciliation must bind the executed prediction and observed action")
    verifiers = {
        ref: resolve(ref, StructuredEventType.VERIFICATION_RESULT)
        for ref in record.verification_event_ids
    }
    for verifier in verifiers.values():
        if verifier.parent_event_ids != (observation_event.event_id,):
            raise ValueError("reconciliation verifier must bind the same observation")
    _validate_chronology(prediction_event, proposal, execution, observation_event, verifiers, event)
    _validate_prior_links(update.prior_state, prediction_event, execution, seen)

    # A prior is only declared here, but its cited evidence must have been committed prospectively.
    if not {r.reference_id for r in update.prior_state.evidence_refs}.issubset(
        prediction.evidence_refs
    ):
        raise ValueError("prior evidence must occur in the prediction commitment")
    allowed_refs = set(prediction.evidence_refs) | set(observation.evidence_refs)
    for verifier in verifiers.values():
        allowed_refs.update(verifier.evidence_refs)
    if not {r.reference_id for r in update.evidence_refs}.issubset(allowed_refs):
        raise ValueError("update evidence must occur in its causal inputs")
    measurements = {m.measurement_id: m for m in update.measurements}
    for binding in record.bindings:
        measurement = measurements[binding.measurement_id]
        innovation = binding.innovation
        if (
            innovation.prediction_commit_id != prediction.prediction_commit_id
            or innovation.observation_id != observation.observation_id
        ):
            raise ValueError("innovation IDs must match committed prediction and observation")
        predicted = _select_number(prediction.predicted_outcome, binding.prediction_path)
        actual = _select_number(observation.actual_outcome, binding.observation_path)
        if (
            innovation.predicted_measurement != predicted
            or innovation.actual_measurement != actual
            or measurement.value != actual
        ):
            raise ValueError("innovation and measurement values must match recorded outcomes")
        if set(innovation.evidence_refs) != set(measurement.evidence_refs):
            raise ValueError("innovation evidence must match measurement evidence")
        if not {r.reference_id for r in measurement.evidence_refs}.issubset(
            observation.evidence_refs
        ):
            raise ValueError("measurement evidence must occur in the observation")
        if binding.verifier_event_id is not None:
            _validate_external_measurement(measurement, verifiers[binding.verifier_event_id])
        elif measurement.source is ConfidenceSource.EXTERNAL_VERIFIER:
            raise ValueError("EXTERNAL_VERIFIER measurement requires a verifier event binding")
    return record


def _validate_prior_links(
    prior: EpistemicStateEstimate,
    prediction: EventEnvelope,
    execution: EventEnvelope,
    seen: Mapping[str, EventEnvelope],
) -> None:
    """Resolve declared prior IDs, allowing current-action conditioning or earlier history."""
    if prior.action_id is None and prior.prediction_commit_id is None:
        return
    earlier_ids: set[str] = set()
    for event_id in seen:
        if event_id == prediction.event_id:
            break
        earlier_ids.add(event_id)
    cutoff = parse_rfc3339_datetime(prediction.timestamp, "timestamp")

    def resolve_prior(
        identifier: str,
        field: str,
        kind: StructuredEventType,
        current: EventEnvelope,
    ) -> EventEnvelope:
        target = next(
            (
                event
                for event in seen.values()
                if event.event_type == kind.value and event.payload[field] == identifier
            ),
            None,
        )
        if target is None:
            raise ValueError(f"prior {field} requires a recorded {kind.value}")
        prior.context.validate_event(target)
        if target.event_id != current.event_id:
            timestamp = parse_rfc3339_datetime(target.timestamp, "prior timestamp")
            if target.event_id not in earlier_ids or timestamp > cutoff:
                raise ValueError("historical prior links must precede the current prediction")
        return target

    if prior.prediction_commit_id is not None:
        resolve_prior(
            prior.prediction_commit_id,
            "prediction_commit_id",
            StructuredEventType.PREDICTION_COMMIT,
            prediction,
        )
    if prior.action_id is not None:
        action = resolve_prior(
            prior.action_id, "action_id", StructuredEventType.ACTION_EXECUTED, execution
        )
        if (
            prior.prediction_commit_id is not None
            and action.payload["prediction_commit_id"] != prior.prediction_commit_id
        ):
            raise ValueError(
                "prior action_id and prediction_commit_id must match recorded ancestry"
            )


def _select_number(value: object, path: OutcomePath) -> float:
    """Traverse exact JSON keys/indices without coercing booleans or missing telemetry."""
    for key in path:
        if type(key) is str and isinstance(value, Mapping) and key in value:
            value = value[key]
        elif type(key) is int and isinstance(value, (list, tuple)) and key < len(value):
            value = value[key]
        else:
            raise ValueError("outcome path does not resolve to recorded telemetry")
    return _number(value, "selected outcome")


def _validate_chronology(
    prediction: EventEnvelope,
    proposal: EventEnvelope,
    execution: EventEnvelope,
    observation: EventEnvelope,
    verifiers: Mapping[str, EventEnvelope],
    event: EventEnvelope,
) -> None:
    chain = (prediction, proposal, execution, observation)
    times = [parse_rfc3339_datetime(e.timestamp, "timestamp") for e in chain]
    end = parse_rfc3339_datetime(event.timestamp, "timestamp")
    if times[0] >= times[2] or any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("prediction must predate execution and ancestry must be chronological")
    for verifier in verifiers.values():
        time = parse_rfc3339_datetime(verifier.timestamp, "timestamp")
        if not times[-1] <= time <= end:
            raise ValueError("verification must follow observation and precede reconciliation")


def _validate_external_measurement(
    measurement: EpistemicMeasurement,
    verifier: EventEnvelope,
) -> None:
    if any(not ref.is_observable for ref in measurement.evidence_refs):
        raise ValueError("externally verified measurements require observable evidence")
    if not set(verifier.verifier_refs).issubset(measurement.provenance):
        raise ValueError("measurement provenance must retain verifier evidence references")
    if set(verifier.payload) == set(VerificationResult.__dataclass_fields__):
        result = VerificationResult(**dict(verifier.payload))
        if not result.verified:
            raise ValueError("external measurement requires successful outcome verification")
        return
    if verifier.payload["verification_level"] != VerificationLevel.RELATIONAL_EVIDENCE.value:
        raise ValueError("local execution verification cannot certify an outcome measurement")
    relational = RelationalVerificationResult.from_dict(verifier.payload["result"])
    if relational.claimed_outcome_supported is not True or relational.provenance_intact is not True:
        raise ValueError("external measurement requires supported outcome and intact provenance")
    evidence = {b.field_name: set(b.evidence_refs) for b in relational.evidence_bindings}
    required = set(measurement.evidence_refs)
    for name in ("claimed_outcome_supported", "provenance_intact"):
        if not required.issubset(evidence.get(name, set())):
            raise ValueError(f"external measurement evidence must bind verifier field {name}")
