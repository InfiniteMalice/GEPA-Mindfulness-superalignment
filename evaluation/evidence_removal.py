"""Bounded whole-version removal experiments without inferred semantic penalties."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Third-party
# Local
from gepa_mindfulness.verification.artifact_evidence import ArtifactAccessRequest
from gepa_mindfulness.verification.artifact_records import (
    ArtifactDiagnosticRecord,
    ArtifactKey,
    artifact_key,
    record_tuple,
    unique,
)
from gepa_mindfulness.verification.debate_records import _record
from gepa_mindfulness.verification.diagnostic_records import _text, restore_records

from .evidence_topology import analyze_evidence_topology
from .evidence_topology_records import TopologyAssessment, TopologyCapture, TopologyProtocol


@dataclass(frozen=True)
class RemovalSlot(ArtifactDiagnosticRecord):
    """A predeclared exclusion of exactly one existing artifact/version."""

    slot_id: str
    removed_artifact: ArtifactKey
    protocol: TopologyProtocol
    schema_version = "removal-slot-v1"
    restorers = {"removed_artifact": artifact_key, "protocol": TopologyProtocol.from_dict}

    def __post_init__(self) -> None:
        _text(self.slot_id, "slot_id")
        object.__setattr__(self, "removed_artifact", artifact_key(self.removed_artifact))
        object.__setattr__(self, "protocol", _record(self.protocol, TopologyProtocol))
        if self.protocol.query.excluded_artifacts != (self.removed_artifact,):
            raise ValueError("removal slot must exclude exactly its artifact version")


def _held_fixed(protocol: TopologyProtocol) -> dict[str, Any]:
    value = protocol.to_dict()
    del value["protocol_id"]
    del value["query"]["request_id"]
    del value["query"]["excluded_artifacts"]
    for key in ("case", "robustness", "expected_actions"):
        del value["subject"][key]
    return value


@dataclass(frozen=True)
class RemovalPlan(ArtifactDiagnosticRecord):
    """A baseline and complete planned removal roster, including uncaptured conditions."""

    plan_id: str
    baseline: TopologyProtocol
    slots: tuple[RemovalSlot, ...]
    schema_version = "removal-plan-v1"
    restorers = {
        "baseline": TopologyProtocol.from_dict,
        "slots": lambda v: restore_records(v, RemovalSlot),
    }

    def __post_init__(self) -> None:
        _text(self.plan_id, "plan_id")
        object.__setattr__(self, "baseline", _record(self.baseline, TopologyProtocol))
        object.__setattr__(self, "slots", record_tuple(self.slots, RemovalSlot))
        if self.baseline.query.excluded_artifacts:
            raise ValueError("removal baseline must exclude nothing")
        if len(self.slots) > 64:
            raise ValueError("removal plan exceeds 64 slots")
        unique([s.slot_id for s in self.slots], "slot IDs")
        unique([s.removed_artifact for s in self.slots], "removed artifact versions")
        protocols = (self.baseline,) + tuple(s.protocol for s in self.slots)
        unique([p.protocol_id for p in protocols], "protocol IDs")
        unique([p.query.request_id for p in protocols], "request IDs")
        held = _held_fixed(self.baseline)
        if any(_held_fixed(s.protocol) != held for s in self.slots):
            raise ValueError("removal changes a held-fixed protocol field")


@dataclass(frozen=True)
class RemovalObservation(ArtifactDiagnosticRecord):
    """Raw capture and receipt attached to one planned removal, never inferred from topology."""

    slot_id: str
    capture: TopologyCapture | None
    assessment: TopologyAssessment | None
    schema_version = "removal-observation-v1"
    restorers = {
        "capture": lambda v: None if v is None else TopologyCapture.from_dict(v),
        "assessment": lambda v: None if v is None else TopologyAssessment.from_dict(v),
    }

    def __post_init__(self) -> None:
        _text(self.slot_id, "slot_id")
        for name, cls in (("capture", TopologyCapture), ("assessment", TopologyAssessment)):
            v = getattr(self, name)
            if v is not None:
                object.__setattr__(self, name, _record(v, cls))


def analyze_evidence_removals(
    plan: RemovalPlan,
    baseline_capture: TopologyCapture | None,
    observations: tuple[RemovalObservation, ...],
    *,
    baseline_assessment: TopologyAssessment | None = None,
    authorize: Callable[[ArtifactAccessRequest], bool] | None = None,
    authenticate: Callable[[TopologyAssessment], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Recompute access and support for every planned exclusion, retaining missing observations."""
    if enabled is not True:
        raise ValueError("removal analysis requires enabled=True")
    plan = _record(plan, RemovalPlan)
    observations = record_tuple(observations, RemovalObservation)
    unique([o.slot_id for o in observations], "observation slot IDs")
    indexed = {o.slot_id: o for o in observations}
    if not indexed.keys() <= {s.slot_id for s in plan.slots}:
        raise ValueError("observation references foreign removal slot")
    baseline = analyze_evidence_topology(
        plan.baseline,
        baseline_capture,
        assessment=baseline_assessment,
        authorize=authorize,
        authenticate=authenticate,
        enabled=True,
    )
    # Use the detached analysis capture rather than caller-owned mutable contents.
    bc = baseline["capture"]
    original_routes = set(baseline["structure"]["available_route_ids"])
    rows, missing = [], []
    for slot in plan.slots:
        observation = indexed.get(slot.slot_id)
        capture = observation.capture if observation is not None else None
        if capture is None:
            missing.append(slot.slot_id)
        if bc is not None and capture is not None and bc["condition"] != capture.condition:
            raise ValueError("removal captures must use the same retrieval condition")
        result = analyze_evidence_topology(
            slot.protocol,
            capture,
            assessment=observation.assessment if observation is not None else None,
            authorize=authorize,
            authenticate=authenticate,
            enabled=True,
        )
        current_routes = set(result["structure"]["available_route_ids"])
        action_changed = None
        if (
            bc is not None
            and bc["status"] == "observed"
            and capture is not None
            and (capture.status == "observed")
        ):
            action_changed = bc["actions"] != list(capture.actions)
        rows.append(
            dict(
                result,
                slot_id=slot.slot_id,
                removed_artifact=list(slot.removed_artifact),
                original_case=plan.baseline.subject.case.case_id,
                destination_case=slot.protocol.subject.case.case_id,
                lost_route_ids=sorted(original_routes - current_routes),
                surviving_route_ids=sorted(original_routes & current_routes),
                observed_action_changed=action_changed,
            )
        )
    return dict(
        schema_version="evidence-removal-analysis-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        plan_id=plan.plan_id,
        baseline=baseline,
        slots=rows,
        missing_slot_ids=sorted(missing),
    )
