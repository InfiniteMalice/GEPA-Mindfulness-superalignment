"""Optional internal-state trajectory adapter interfaces."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol


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
    "DeterministicInternalStateTrajectoryAdapter",
    "InternalStateTrajectoryAdapter",
    "InternalStateTrajectorySnapshot",
    "TranscriptOnlyTrajectoryAdapter",
]
