"""Raw three-condition retrieval comparisons with content matching and explicit measured costs."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

# Third-party
# Local
from gepa_mindfulness.verification.artifact_evidence import ArtifactAccessRequest
from gepa_mindfulness.verification.artifact_records import (
    ArtifactDiagnosticRecord,
    payload_digest,
    record_tuple,
    unique,
)
from gepa_mindfulness.verification.debate_records import _record
from gepa_mindfulness.verification.diagnostic_records import _text, choice, restore_records

from .causal_records import canonical_json
from .evidence_topology import analyze_evidence_topology, summarize_topology_rows
from .evidence_topology_records import (
    CONDITIONS,
    TOPOLOGY_METRICS,
    TopologyAssessment,
    TopologyCapture,
    TopologyProtocol,
)


@dataclass(frozen=True)
class TopologyComparisonSlot(ArtifactDiagnosticRecord):
    """Planned raw protocol plus declared retrieval/index/checkpoint treatment."""

    run_id: str
    condition: str
    model_family: str
    index_version: str
    protocol: TopologyProtocol
    schema_version = "topology-comparison-slot-v1"
    restorers = {"protocol": TopologyProtocol.from_dict}

    def __post_init__(self) -> None:
        for name in ("run_id", "model_family", "index_version"):
            _text(getattr(self, name), name)
        choice(self.condition, "condition", CONDITIONS)
        object.__setattr__(self, "protocol", _record(self.protocol, TopologyProtocol))


def _pairing_key(slot: TopologyComparisonSlot) -> str:
    data = slot.protocol.to_dict()
    del data["protocol_id"]
    del data["query"]["request_id"]
    del data["subject"]["system"]["model_version"]
    return payload_digest(dict(model_family=slot.model_family, protocol=data))


@dataclass(frozen=True)
class TopologyComparisonPlan(ArtifactDiagnosticRecord):
    """Complete condition roster with protocols for planned runs whose captures may be absent."""

    experiment_id: str
    slots: tuple[TopologyComparisonSlot, ...]
    schema_version = "topology-comparison-plan-v1"
    restorers = {"slots": lambda v: restore_records(v, TopologyComparisonSlot)}

    def __post_init__(self) -> None:
        _text(self.experiment_id, "experiment_id")
        object.__setattr__(self, "slots", record_tuple(self.slots, TopologyComparisonSlot))
        unique([s.run_id for s in self.slots], "run IDs")
        if {s.condition for s in self.slots} != set(CONDITIONS):
            raise ValueError("comparison plan requires all three conditions")
        unique([(_pairing_key(s), s.condition) for s in self.slots], "condition per pairing key")


@dataclass(frozen=True)
class TopologyComparisonRun(ArtifactDiagnosticRecord):
    """Raw capture and semantic receipt only; saved report flags cannot authenticate anything."""

    run_id: str
    capture: TopologyCapture | None
    assessment: TopologyAssessment | None
    schema_version = "topology-comparison-run-v1"
    restorers = {
        "capture": lambda v: None if v is None else TopologyCapture.from_dict(v),
        "assessment": lambda v: None if v is None else TopologyAssessment.from_dict(v),
    }

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        for name, cls in (("capture", TopologyCapture), ("assessment", TopologyAssessment)):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _record(value, cls))


def _statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = [s["value"] for s in samples if s["value"] is not None]
    missing = len(samples) - len(values)
    return dict(
        samples=samples,
        known_count=len(values),
        missing_count=missing,
        partial=missing > 0,
        min=min(values) if values else None,
        max=max(values) if values else None,
        median=median(values) if values else None,
        mean=mean(values) if values else None,
    )


def _measurements(
    slots: list[TopologyComparisonSlot],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    latency, costs = [], []
    for slot in slots:
        c = reports[slot.run_id]["capture"]
        observed = c is not None and c["status"] == "observed"
        latency.append(dict(run_id=slot.run_id, value=c["latency_seconds"] if observed else None))
        costs.append(
            dict(
                run_id=slot.run_id,
                value=c["retrieval_cost"] if observed else None,
                unit=c["cost_unit"] if observed else None,
            )
        )
    units = sorted({s["unit"] for s in costs if s["unit"] is not None})
    known = sum(s["value"] is not None for s in costs)
    return dict(
        latency_seconds=_statistics(latency),
        retrieval_cost=dict(
            samples=costs,
            known_count=known,
            missing_count=len(costs) - known,
            partial=known < len(costs),
            by_unit={unit: _statistics([s for s in costs if s["unit"] == unit]) for unit in units},
        ),
    )


def _paired_rows(
    groups: dict[str, dict[str, TopologyComparisonSlot]],
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for key, slots in sorted(groups.items()):
        captures = {c: reports[s.run_id]["capture"] for c, s in slots.items()}
        all_observed = set(slots) == set(CONDITIONS) and all(
            c is not None and c["status"] == "observed" for c in captures.values()
        )
        complete: dict[str, bool] = {}
        for metric in TOPOLOGY_METRICS:
            complete[metric] = all_observed and all(
                reports[s.run_id]["metrics"][metric]["rate"] is not None for s in slots.values()
            )
        latency_ok = all_observed and all(
            c["latency_seconds"] is not None for c in captures.values()
        )
        cost_ok = all_observed and all(c["retrieval_cost"] is not None for c in captures.values())
        cost_ok = cost_ok and len({c["cost_unit"] for c in captures.values()}) == 1
        baseline = slots.get("existing_retrieval")
        for condition in CONDITIONS[1:]:
            treatment = slots.get(condition)
            if treatment is None:
                continue
            deltas = {
                metric: (
                    (
                        reports[treatment.run_id]["metrics"][metric]["rate"]
                        - reports[baseline.run_id]["metrics"][metric]["rate"]
                    )
                    if complete[metric] and baseline is not None
                    else None
                )
                for metric in TOPOLOGY_METRICS
            }
            result.append(
                dict(
                    pairing_digest=key,
                    condition=condition,
                    baseline_run_id=baseline.run_id if baseline else None,
                    treatment_run_id=treatment.run_id,
                    deltas=deltas,
                    reasons={
                        m: (
                            []
                            if complete[m]
                            else [
                                "requires three observed content-matched resolved conditions "
                                "for this metric"
                            ]
                        )
                        for m in TOPOLOGY_METRICS
                    },
                    latency_delta=(
                        (
                            captures[condition]["latency_seconds"]
                            - captures["existing_retrieval"]["latency_seconds"]
                        )
                        if latency_ok
                        else None
                    ),
                    latency_reason=(
                        None if latency_ok else "requires three known observed latencies"
                    ),
                    cost_delta=(
                        (
                            captures[condition]["retrieval_cost"]
                            - captures["existing_retrieval"]["retrieval_cost"]
                        )
                        if cost_ok
                        else None
                    ),
                    cost_unit=captures[condition]["cost_unit"] if cost_ok else None,
                    cost_reason=(
                        None if cost_ok else "requires three known observed costs in one unit"
                    ),
                )
            )
    return result


def compare_evidence_topology(
    plan: TopologyComparisonPlan,
    runs: tuple[TopologyComparisonRun, ...],
    *,
    authorize: Callable[[ArtifactAccessRequest], bool] | None = None,
    authenticate: Callable[[TopologyAssessment], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Reanalyze every planned run and compare only exact content matches from raw observations."""
    if enabled is not True:
        raise ValueError("topology comparison requires enabled=True")
    plan = _record(plan, TopologyComparisonPlan)
    runs = record_tuple(runs, TopologyComparisonRun)
    unique([r.run_id for r in runs], "observed run IDs")
    indexed = {r.run_id: r for r in runs}
    if not indexed.keys() <= {s.run_id for s in plan.slots}:
        raise ValueError("foreign comparison run ID")
    reports: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, TopologyComparisonSlot]] = {}
    for slot in plan.slots:
        run = indexed.get(slot.run_id)
        capture = run.capture if run is not None else None
        if capture is not None and capture.condition != slot.condition:
            raise ValueError("capture condition differs from planned condition")
        reports[slot.run_id] = analyze_evidence_topology(
            slot.protocol,
            capture,
            assessment=run.assessment if run is not None else None,
            authorize=authorize,
            authenticate=authenticate,
            enabled=True,
        )
        groups.setdefault(_pairing_key(slot), {})[slot.condition] = slot
    conditions = {}
    for condition in CONDITIONS:
        slots = [s for s in plan.slots if s.condition == condition]
        rows = [
            dict(
                row,
                opportunity_id=canonical_json([s.run_id, row["opportunity_id"]]),
                run_id=s.run_id,
            )
            for s in slots
            for row in reports[s.run_id]["rows"]
        ]
        conditions[condition] = dict(
            planned_runs=len(slots),
            rows=rows,
            **summarize_topology_rows(rows),
            **_measurements(slots, reports),
        )
    return dict(
        schema_version="evidence-topology-comparison-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        training_effect_established=False,
        caveat=(
            "Descriptive host observations and authored fixtures do not establish training effects."
        ),
        experiment_id=plan.experiment_id,
        plan=plan.to_dict(),
        runs=reports,
        conditions=conditions,
        paired=_paired_rows(groups, reports),
        missing_run_ids=sorted(
            s.run_id
            for s in plan.slots
            if s.run_id not in indexed or indexed[s.run_id].capture is None
        ),
    )
