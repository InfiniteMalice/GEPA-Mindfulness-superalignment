"""Optional internal-state trajectory adapter interfaces."""

# Standard library
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Protocol

from ._continuity_validation import index_field, references, score, text_field


class MeasurementStatus(str, Enum):
    """Origin of bounded features; none of these statuses certifies intent."""

    MEASURED_INTERNAL = "measured_internal"
    DERIVED_PROXY = "derived_proxy"
    TRANSCRIPT_PROXY = "transcript_proxy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SoTStateSnapshot:
    """Experimental SoT-compatible summary, not a reproduction of the paper.

    All four scalars use adapter-declared [0, 1] normalization. The schema ID must
    identify that calibration (including any mapping of cosine consistency or entropy).
    Raw tensors and private reasoning are deliberately absent from this interface.
    """

    snapshot_id: str
    conversation_id: str
    turn_index: int
    adapter_name: str
    source_model_id: str
    backend_id: str
    layer_sources: tuple[str, ...]
    feature_schema: str
    provenance: tuple[str, ...]
    measurement_status: MeasurementStatus
    evidence_status: str
    source_kind: str
    local_organization: float | None = None
    progress_magnitude: float | None = None
    directional_consistency: float | None = None
    predictive_uncertainty: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "snapshot_id",
            "conversation_id",
            "adapter_name",
            "source_model_id",
            "backend_id",
            "feature_schema",
        ):
            text_field(getattr(self, name), name)
        index_field(self.turn_index, "turn_index")
        references(self.layer_sources, "layer_sources")
        references(self.provenance, "provenance")
        if not self.provenance:
            raise ValueError("provenance must identify the measurement or unavailability source")
        if type(self.measurement_status) is not MeasurementStatus:
            raise ValueError("measurement_status must be a MeasurementStatus")
        if self.evidence_status not in {"observed", "synthetic", "unverified", "unavailable"}:
            raise ValueError("unsupported evidence_status")
        if self.source_kind not in {"internal", "transcript", "synthetic", "unavailable"}:
            raise ValueError("unsupported source_kind")
        measured = self.measurement_status is MeasurementStatus.MEASURED_INTERNAL
        if measured and (
            self.source_kind != "internal"
            or self.evidence_status != "observed"
            or not self.layer_sources
        ):
            raise ValueError(
                "measured state requires observed internal telemetry and layer sources"
            )
        transcript = self.measurement_status is MeasurementStatus.TRANSCRIPT_PROXY
        if transcript != (self.source_kind == "transcript"):
            raise ValueError("transcript source must remain a transcript proxy")
        values = (
            self.local_organization,
            self.progress_magnitude,
            self.directional_consistency,
            self.predictive_uncertainty,
        )
        unavailable = self.measurement_status is MeasurementStatus.UNAVAILABLE
        if unavailable:
            if any(value is not None for value in values):
                raise ValueError("unavailable telemetry must be null, not zero")
            if self.source_kind != "unavailable" or self.evidence_status != "unavailable":
                raise ValueError("unavailable state requires unavailable source and evidence")
        else:
            if self.source_kind == "unavailable" or self.evidence_status == "unavailable":
                raise ValueError("available features cannot claim unavailable provenance")
            for value in values:
                score(value, "state feature")

    @property
    def feature_vector(self) -> tuple[float, ...] | None:
        """Return the four normalized features, or no vector when unavailable."""
        if self.measurement_status is MeasurementStatus.UNAVAILABLE:
            return None
        return tuple(
            float(value)
            for value in (
                self.local_organization,
                self.progress_magnitude,
                self.directional_consistency,
                self.predictive_uncertainty,
            )
            if value is not None
        )

    @property
    def comparison_key(self) -> tuple[object, ...]:
        """Identify the measurement space; differently calibrated spaces never compare."""
        return (
            self.adapter_name,
            self.source_model_id,
            self.backend_id,
            self.layer_sources,
            self.feature_schema,
            self.measurement_status,
            self.evidence_status,
            self.source_kind,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize bounded summaries only; missing telemetry remains JSON null."""
        result = asdict(self)
        result["measurement_status"] = self.measurement_status.value
        return result


class SoTStateAdapter(Protocol):
    """Backend extension beside InternalStateTrajectoryAdapter.

    A backend may consume its existing trajectory snapshot and ephemeral internal data.
    It must declare its calibration and actual measurement source; no backend is shipped
    or automatically substituted here. This protocol does not authorize tensor storage.
    """

    def extract_state(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        trajectory: InternalStateTrajectorySnapshot | None = None,
    ) -> SoTStateSnapshot:
        """Return a bounded state or an explicitly unavailable snapshot."""


@dataclass(frozen=True)
class InternalStateTrajectorySnapshot:
    """Normalized layer-wise feature snapshot."""

    snapshot_id: str
    conversation_id: str
    turn_index: int
    adapter_name: str
    layer_count: int
    normalized_layer_features: tuple[tuple[float, ...], ...]
    cache_available: bool
    hidden_states_available: bool
    transcript_fallback_used: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class InternalStateTrajectoryAdapter(Protocol):
    """Adapter interface for model-specific layer trajectory features."""

    def extract_layer_trajectory(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        prompt_text: str,
        conversation_history: tuple[str, ...],
        kv_cache: object | None = None,
        hidden_states: object | None = None,
        candidate_response: str | None = None,
    ) -> InternalStateTrajectorySnapshot:
        """Extract normalized layer features."""


class DeterministicInternalStateTrajectoryAdapter:
    """Synthetic deterministic adapter for CI fixtures."""

    def __init__(self, layer_count: int = 4) -> None:
        self.layer_count = layer_count

    def extract_layer_trajectory(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        prompt_text: str,
        conversation_history: tuple[str, ...],
        kv_cache: object | None = None,
        hidden_states: object | None = None,
        candidate_response: str | None = None,
    ) -> InternalStateTrajectorySnapshot:
        raw = _feature_rows(hidden_states, self.layer_count)
        rows = tuple(tuple(_clamp(value) for value in row) for row in raw)
        return InternalStateTrajectorySnapshot(
            snapshot_id=_snapshot_id(conversation_id, turn_index, repr(rows)),
            conversation_id=conversation_id,
            turn_index=turn_index,
            adapter_name="deterministic_internal_state",
            layer_count=len(rows),
            normalized_layer_features=rows,
            cache_available=kv_cache is not None,
            hidden_states_available=hidden_states is not None,
            transcript_fallback_used=False,
            metadata={
                "extension_points": (
                    "HuggingFaceInternalStateTrajectoryAdapter",
                    "VLLMInternalStateTrajectoryAdapter",
                    "TransformerLensInternalStateTrajectoryAdapter",
                )
            },
        )


class TranscriptOnlyTrajectoryAdapter:
    """Transcript-only trajectory features when internal state is unavailable."""

    def extract_layer_trajectory(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        prompt_text: str,
        conversation_history: tuple[str, ...],
        kv_cache: object | None = None,
        hidden_states: object | None = None,
        candidate_response: str | None = None,
    ) -> InternalStateTrajectorySnapshot:
        text = "\n".join(conversation_history + (prompt_text, candidate_response or ""))
        row = (
            _signal(text, "COMPONENT_"),
            _signal(text, "PARAMETER_"),
            _signal(text, "INTEGRATION_"),
        )
        rows = (row,)
        return InternalStateTrajectorySnapshot(
            snapshot_id=_snapshot_id(conversation_id, turn_index, text),
            conversation_id=conversation_id,
            turn_index=turn_index,
            adapter_name="transcript_only",
            layer_count=1,
            normalized_layer_features=rows,
            cache_available=False,
            hidden_states_available=False,
            transcript_fallback_used=True,
        )


def _feature_rows(hidden_states: object | None, layer_count: int) -> tuple[tuple[float, ...], ...]:
    if isinstance(hidden_states, dict):
        rows = hidden_states.get("features", ())
    else:
        rows = hidden_states or ()
    if not isinstance(rows, (list, tuple)):
        rows = ()
    normalized: list[tuple[float, ...]] = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            normalized.append(tuple(float(value) for value in row))
    while len(normalized) < layer_count:
        normalized.append((0.0, 0.0))
    return tuple(normalized[:layer_count])


def _signal(text: str, marker: str) -> float:
    return _clamp(text.lower().count(marker.lower()) / 3)


def _snapshot_id(conversation_id: str, turn_index: int, payload: str) -> str:
    digest = sha256(f"{conversation_id}:{turn_index}:{payload}".encode()).hexdigest()[:16]
    return f"state-{digest}"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "MeasurementStatus",
    "SoTStateAdapter",
    "SoTStateSnapshot",
    "DeterministicInternalStateTrajectoryAdapter",
    "InternalStateTrajectoryAdapter",
    "InternalStateTrajectorySnapshot",
    "TranscriptOnlyTrajectoryAdapter",
]
