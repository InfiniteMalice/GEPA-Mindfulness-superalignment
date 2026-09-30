"""Freeze verified blind judgments before releasing a peer-discussion packet."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope, StructuredEventType

from .epistemic_reconciliation import EpistemicReconciliation
from .epistemic_state import EpistemicMeasurement, _snapshot, _strings, _text
from .scalar_fusion import MAX_SOURCES, ScalarFusionResult, _measurements, fuse_scalar_measurements
from .state import parse_rfc3339_datetime


@dataclass(frozen=True, slots=True)
class FrozenRoundZero:
    """Detached peer packet; the host must enforce the real transport/exposure boundary."""

    panel_id: str
    participant_ids: tuple[str, ...]
    measurements: tuple[EpistemicMeasurement, ...]
    reconciliation_event_ids: tuple[str, ...]
    released_at: str

    def to_dict(self) -> dict[str, Any]:
        """Export the original judgments with their causal-record references."""
        return dict(
            schema_version="frozen-round-zero-v1",
            panel_id=self.panel_id,
            participant_ids=list(self.participant_ids),
            measurements=[m.to_dict() for m in self.measurements],
            reconciliation_event_ids=list(self.reconciliation_event_ids),
            released_at=self.released_at,
        )


class VerifiedJudgmentPanel:
    """Sequential, in-memory host collector. No network access, votes, rewards or authority.

    Membership is fixed at construction. The collector releases no partial peer packet; all
    members must submit verified judgments before open_discussion seals and returns Round-0.
    Host-supplied participant identity does not authenticate a human, model or independent error.
    """

    def __init__(self, panel_id: str, participant_ids: tuple[str, ...]) -> None:
        _text(panel_id, "panel_id")
        participants = _strings(participant_ids, "participant_ids", required=True)
        if len(participants) > MAX_SOURCES:
            raise ValueError(f"panel supports at most {MAX_SOURCES} participants")
        self._panel_id = panel_id
        self._participants = participants
        self._judgments: dict[str, tuple[EpistemicMeasurement, str, str]] = {}
        self._history: dict[str, str] = {}
        self._released_at: str | None = None

    def add_judgment(
        self,
        participant_id: str,
        events: Sequence[EventEnvelope],
        *,
        reconciliation_event_id: str,
        measurement_id: str,
    ) -> None:
        """Validate causal evidence and retain a detached judgment only after all checks pass."""
        if self._released_at is not None:
            raise ValueError("Round-0 is sealed after peer release")
        _text(participant_id, "participant_id")
        _text(reconciliation_event_id, "reconciliation_event_id")
        _text(measurement_id, "measurement_id")
        if participant_id not in self._participants or participant_id in self._judgments:
            raise ValueError("participant must be declared and submit exactly once")
        history = tuple(events)
        validate_action_bound_sequence(history)
        snapshots = {
            e.event_id: json.dumps(
                e.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            for e in history
        }
        if any(k in self._history and self._history[k] != v for k, v in snapshots.items()):
            raise ValueError("submitted history conflicts with retained events")
        combined = self._history | snapshots
        # Fresh envelope IDs must not conceal reuse of an immutable causal-record identity.
        validate_action_bound_sequence(
            tuple(EventEnvelope(**json.loads(data)) for data in combined.values())
        )
        event = next((e for e in history if e.event_id == reconciliation_event_id), None)
        if event is None or event.event_type != StructuredEventType.EPISTEMIC_RECONCILIATION.value:
            raise ValueError("judgment requires a validated reconciliation event")
        record = EpistemicReconciliation.from_dict(event.payload)
        binding = next((b for b in record.bindings if b.measurement_id == measurement_id), None)
        if binding is None or binding.verifier_event_id is None:
            raise ValueError("judgment requires an explicit successful verifier binding")
        measurement = next(
            m for m in record.update.measurements if m.measurement_id == measurement_id
        )
        # Successful bound verification was enforced by the existing sequence validator above.
        _measurements(tuple(j[0] for j in self._judgments.values()) + (measurement,))
        self._judgments[participant_id] = (
            _snapshot(measurement, EpistemicMeasurement),
            event.event_id,
            event.timestamp,
        )
        self._history = combined

    def open_discussion(self, released_at: str) -> FrozenRoundZero:
        """Atomically seal the complete verified cohort before returning its first peer packet."""
        if self._released_at is not None:
            raise ValueError("Round-0 is already sealed")
        if len(self._judgments) != len(self._participants):
            raise ValueError("complete verified Round-0 is required before peer release")
        release = parse_rfc3339_datetime(released_at, "release timestamp")
        if any(
            release < parse_rfc3339_datetime(j[2], "judgment timestamp")
            for j in self._judgments.values()
        ):
            raise ValueError("release timestamp must not precede a verified judgment")
        packet = FrozenRoundZero(
            self._panel_id,
            self._participants,
            self._originals(),
            tuple(self._judgments[p][1] for p in self._participants),
            released_at,
        )
        self._released_at = released_at
        return packet

    def fuse_round_zero(self, *, estimate_id: str, **options: Any) -> ScalarFusionResult:
        """Fuse the frozen original judgments, preserving pre-discussion dissent."""
        self._require_release(options)
        return fuse_scalar_measurements(
            self._originals(),
            estimate_id=estimate_id,
            peer_exposed=False,
            **options,
        )

    def fuse_post_discussion(
        self,
        measurements: tuple[EpistemicMeasurement, ...],
        *,
        estimate_id: str,
        **options: Any,
    ) -> ScalarFusionResult:
        """Fuse one fresh declaration per participant in cohort order with peer exposure recorded.

        These later declarations are diagnostics; this method does not verify new outcome events.
        It cannot replace the frozen originals or reuse their identities as new evidence.
        """
        self._require_release(options)
        items = _measurements(measurements)
        originals = self._originals()
        if len(items) != len(originals) or {m.measurement_id for m in items} & {
            m.measurement_id for m in originals
        }:
            raise ValueError(
                "post-discussion measurements require one fresh identity per participant"
            )
        before, after = originals[0], items[0]
        if (before.context, before.representation_id, before.target_dimension) != (
            after.context,
            after.representation_id,
            after.target_dimension,
        ):
            raise ValueError("post-discussion measurements must retain the Round-0 target/context")
        return fuse_scalar_measurements(
            items, estimate_id=estimate_id, peer_exposed=True, **options
        )

    def _originals(self) -> tuple[EpistemicMeasurement, ...]:
        return tuple(
            _snapshot(self._judgments[p][0], EpistemicMeasurement) for p in self._participants
        )

    def _require_release(self, options: dict[str, Any]) -> None:
        if self._released_at is None:
            raise ValueError("open_discussion must first freeze and release the complete cohort")
        if "peer_exposed" in options:
            raise ValueError("panel controls peer_exposed; callers cannot override it")
