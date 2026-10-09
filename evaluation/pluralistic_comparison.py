"""Raw-evidence comparisons across three declared augmentation conditions."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Third-party
# Local
from gepa_mindfulness.verification.debate_records import DebateSession, _digest, _record
from gepa_mindfulness.verification.diagnostic_records import _text, choice, restore_records
from semantic_intent_robustness.perspective_generation import PerspectiveGenerationContext
from semantic_intent_robustness.perspective_protocol import PerspectiveCapture, _items, _unique

from .causal_diagnostics import _metrics as causal_metrics
from .causal_diagnostics import _snapshot, evaluate_causal_suite
from .causal_records import (
    CausalCapture,
    MetricOpportunity,
    PairAdjudication,
    canonical_json,
    content_digest,
)
from .debate_analysis import analyze_debate
from .debate_records import DebateAssessment, DebateProtocol, DebateRecord, DebateVerification
from .pluralistic_diagnostics import analyze_pluralistic, summarize_pluralistic_rows
from .pluralistic_records import PLURALISTIC_METRICS, PluralisticAssessment, PluralisticProtocol
from .sensitive_debate import validate_debate_session
from .v5_records import SystemIdentity

CONDITIONS = ("plain_synthetic", "laundering_only", "laundering_debate_pluralistic")


@dataclass(frozen=True)
class ComparisonSlot(DebateRecord):
    """A planned run with host-declared treatment, holdout and model-family identity."""

    run_id: str
    condition: str
    family_id: str
    split_id: str
    curriculum_version: str
    model_family: str
    model_version: str
    harness_version: str
    seed: int
    repeat_id: int
    pair_digest: str
    schema_version = "pluralistic-comparison-slot-v1"

    def __post_init__(self) -> None:
        for name in ("run_id", "family_id", "split_id", "curriculum_version", "model_family"):
            _text(getattr(self, name), name)
        choice(self.condition, "condition", CONDITIONS)
        SystemIdentity(self.repeat_id, self.seed, self.model_version, self.harness_version)
        _digest(self.pair_digest)


@dataclass(frozen=True)
class ComparisonPlan(DebateRecord):
    """Full run roster, including planned conditions whose captures never arrive."""

    experiment_id: str
    slots: tuple[ComparisonSlot, ...]
    schema_version = "pluralistic-comparison-plan-v1"
    restorers = {"slots": lambda v: restore_records(v, ComparisonSlot)}

    def __post_init__(self) -> None:
        _text(self.experiment_id, "experiment_id")
        object.__setattr__(self, "slots", _items(self.slots, ComparisonSlot))
        _unique([s.run_id for s in self.slots], "run IDs")
        if {s.condition for s in self.slots} != set(CONDITIONS):
            raise ValueError("plan must contain all three conditions")


@dataclass(frozen=True)
class DebateAttachment(DebateRecord):
    """A raw PR-2 session and its optional independent semantic receipt."""

    protocol: DebateProtocol
    session: DebateSession
    assessment: DebateAssessment | None
    schema_version = "pluralistic-debate-attachment-v1"
    restorers = {
        "protocol": DebateProtocol.from_dict,
        "session": DebateSession.from_dict,
        "assessment": lambda v: None if v is None else DebateAssessment.from_dict(v),
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol", _record(self.protocol, DebateProtocol))
        object.__setattr__(self, "session", _record(self.session, DebateSession))
        if self.assessment is not None:
            object.__setattr__(self, "assessment", _record(self.assessment, DebateAssessment))
        validate_debate_session(self.protocol, self.session)


@dataclass(frozen=True)
class ComparisonRun(DebateRecord):
    """Captured inputs only; no saved analysis or acceptance flag can be submitted."""

    run_id: str
    protocol: PluralisticProtocol
    captures: tuple[CausalCapture, ...]
    perspective_capture: PerspectiveCapture | None
    assessment: PluralisticAssessment | None
    causal_protocol_id: str
    causal_opportunities: tuple[MetricOpportunity, ...]
    causal_judgments: tuple[PairAdjudication, ...]
    debate: DebateAttachment | None
    schema_version = "pluralistic-comparison-run-v1"
    restorers = {
        "protocol": PluralisticProtocol.from_dict,
        "captures": lambda v: tuple(CausalCapture.from_dict(x) for x in v),
        "perspective_capture": lambda v: None if v is None else PerspectiveCapture.from_dict(v),
        "assessment": lambda v: None if v is None else PluralisticAssessment.from_dict(v),
        "causal_opportunities": lambda v: tuple(MetricOpportunity.from_dict(x) for x in v),
        "causal_judgments": lambda v: tuple(PairAdjudication.from_dict(x) for x in v),
        "debate": lambda v: None if v is None else DebateAttachment.from_dict(v),
    }

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        _text(self.causal_protocol_id, "causal_protocol_id")
        object.__setattr__(self, "protocol", _record(self.protocol, PluralisticProtocol))
        object.__setattr__(self, "captures", _snapshot(self.captures, CausalCapture))
        object.__setattr__(
            self, "causal_opportunities", _snapshot(self.causal_opportunities, MetricOpportunity)
        )
        object.__setattr__(
            self, "causal_judgments", _snapshot(self.causal_judgments, PairAdjudication)
        )
        for name, cls in (
            ("perspective_capture", PerspectiveCapture),
            ("assessment", PluralisticAssessment),
            ("debate", DebateAttachment),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _record(value, cls))
        if any(
            o.metric == "semantic_laundering_susceptibility" and o.cohort == "benign_control"
            for o in self.causal_opportunities
        ):
            raise ValueError(
                "PR-1 requires cohort 'benign'; select it before obtaining causal receipts"
            )


def _evaluation_item_digest(protocol: PluralisticProtocol) -> str:
    """Match evaluation content and rubric independently of treatment/checkpoint identity."""
    pair = protocol.pair
    arms = [
        dict(
            case=v.case.to_dict(),
            robustness=v.robustness.to_dict(),
            turns=[t.to_dict() for t in v.turns],
            factors=v.factors,
            expected_actions=v.expected_actions,
        )
        for v in (pair.before, pair.after)
    ]
    return content_digest(
        dict(
            arms=arms,
            intervention_kind=pair.intervention_kind,
            claimed_equivalence=pair.claimed_equivalence,
            public_context=PerspectiveGenerationContext.from_plan(protocol.plan).to_dict(),
            rubric_id=protocol.rubric_id,
            evaluator_contract=protocol.evaluator.contract_id,
            opportunities=sorted(
                (o.metric, o.severity.value, o.cohort) for o in protocol.opportunities
            ),
        )
    )


def _pair_key(slot: ComparisonSlot, run: ComparisonRun) -> tuple[Any, ...]:
    pair = run.protocol.pair
    return (
        slot.family_id,
        slot.split_id,
        slot.model_family,
        slot.harness_version,
        slot.seed,
        slot.repeat_id,
        pair.before.case.case_id,
        pair.after.robustness.stripe_id,
        pair.after.robustness.subtype,
        _evaluation_item_digest(run.protocol),
    )


def _validate_slot(slot: ComparisonSlot, run: ComparisonRun) -> None:
    pair = run.protocol.pair
    if (
        pair.digest != slot.pair_digest
        or pair.family_id != slot.family_id
        or pair.after.system
        != SystemIdentity(slot.repeat_id, slot.seed, slot.model_version, slot.harness_version)
    ):
        raise ValueError("run does not match planned pair/family/system identity")


def _debate_boundary(run: ComparisonRun) -> bool:
    attachment = run.debate
    if attachment is None:
        return False
    pair = run.protocol.pair
    if attachment.protocol.subject != pair.after:
        raise ValueError("debate subject must match exact after variant")
    final = attachment.session.rounds[-1].after
    capture = next((c for c in run.captures if c.variant_id == pair.after.variant_id), None)
    if final is None or capture is None or capture.status != "observed":
        return False
    if capture.actions != (final.proposed_action,):
        raise ValueError("debate final action does not match exact captured final-action boundary")
    return all(r.status == "observed" for r in attachment.session.rounds)


def _scope_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [dict(r, opportunity_id=f"{run_id}/{r['opportunity_id']}", run_id=run_id) for r in rows]


def _condition_summary(slots: list[ComparisonSlot], rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [r for r in rows if r["status"] == "observed"]
    pluralistic = [
        item
        for r in observed
        for item in _scope_rows(r["pluralistic_report"]["metric_rows"], r["run_id"])
    ]
    causal = [
        item for r in observed for item in _scope_rows(r["causal_report"]["rows"], r["run_id"])
    ]
    pairs = [
        dict(p, pair_id=f"{r['run_id']}/{p['pair_id']}")
        for r in observed
        for p in r["causal_report"]["pairs"]
    ]
    return dict(
        planned_runs=len(slots),
        observed_runs=len(observed),
        missing_runs=[r["run_id"] for r in rows if r["status"] == "missing"],
        source_report_digests=[r["pluralistic_report"]["result_digest"] for r in observed],
        metric_rows=pluralistic,
        **summarize_pluralistic_rows(pluralistic),
        causal_metrics=causal_metrics(
            [r for r in causal if r["intervention_kind"] == "single_variable"],
            [p for p in pairs if p["intervention_kind"] == "single_variable"],
        ),
        causal_compound_metrics=causal_metrics(
            [r for r in causal if r["intervention_kind"] == "compound"],
            [p for p in pairs if p["intervention_kind"] == "compound"],
        ),
        causal_rows=causal,
    )


def _paired_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    unpaired = []
    for row in rows:
        if row["status"] == "missing":
            unpaired.append(dict(run_id=row["run_id"], reason="run missing"))
        elif row["pairing_key"] is None:
            unpaired.append(
                dict(run_id=row["run_id"], reason="before/after system identity differs")
            )
        else:
            group = groups.setdefault(tuple(row["pairing_key"]), {})
            condition = row["slot"]["condition"]
            if condition in group:
                raise ValueError("ambiguous duplicate pairing key within condition")
            group[condition] = row
    paired = []
    for key, group in sorted(groups.items(), key=lambda kv: canonical_json(kv[0])):
        missing = [c for c in CONDITIONS if c not in group]
        deltas: dict[str, Any] = {}
        complete_phases = not missing and group[CONDITIONS[-1]]["combined_protocol_complete"]
        for metric in PLURALISTIC_METRICS:
            summaries = {c: r["pluralistic_report"]["metrics"][metric] for c, r in group.items()}
            complete = complete_phases and all(
                s["rate"] is not None
                and not any(s[state] for state in ("unresolved", "missing", "censored"))
                for s in summaries.values()
            )
            deltas[metric] = (
                {c: summaries[c]["rate"] - summaries[CONDITIONS[0]]["rate"] for c in CONDITIONS[1:]}
                if complete
                else None
            )
        if missing:
            unpaired.extend(
                dict(run_id=r["run_id"], reason="matching condition missing")
                for r in group.values()
            )
        paired.append(
            dict(
                pairing_key=key,
                run_ids={c: r["run_id"] for c, r in group.items()},
                missing_conditions=missing,
                complete_phases=complete_phases,
                deltas=deltas,
            )
        )
    return paired, unpaired


def compare_pluralistic_conditions(
    plan: ComparisonPlan,
    runs: tuple[ComparisonRun, ...],
    *,
    authenticate_causal: Callable[[PairAdjudication], bool] | None = None,
    authenticate_pluralistic: Callable[[PluralisticAssessment], bool] | None = None,
    authenticate_debate_check: Callable[[DebateVerification], bool] | None = None,
    authenticate_debate_assessment: Callable[[DebateAssessment], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Recompute raw observations; matched fixture deltas establish no training effect."""
    if enabled is not True:
        raise ValueError("comparison requires enabled=True")
    plan = _record(plan, ComparisonPlan)
    runs = _items(runs, ComparisonRun)
    _unique([r.run_id for r in runs], "captured run IDs")
    indexed = {r.run_id: r for r in runs}
    if not set(indexed) <= {s.run_id for s in plan.slots}:
        raise ValueError("unplanned run")
    rows: list[dict[str, Any]] = []
    for slot in plan.slots:
        run = indexed.get(slot.run_id)
        if run is None:
            rows.append(dict(run_id=slot.run_id, slot=slot.to_dict(), status="missing"))
            continue
        _validate_slot(slot, run)
        debate_complete = _debate_boundary(run)
        pluralistic = analyze_pluralistic(
            run.protocol,
            run.captures,
            perspective_capture=run.perspective_capture,
            assessment=run.assessment,
            authenticate=authenticate_pluralistic,
            enabled=True,
        )
        causal = evaluate_causal_suite(
            (run.protocol.pair,),
            run.captures,
            run.causal_judgments,
            protocol_id=run.causal_protocol_id,
            opportunities=run.causal_opportunities,
            authenticate=authenticate_causal,
            enabled=True,
        )
        debate = (
            None
            if run.debate is None
            else analyze_debate(
                run.debate.protocol,
                run.debate.session,
                assessment=run.debate.assessment,
                authenticate_check=authenticate_debate_check,
                authenticate_assessment=authenticate_debate_assessment,
                enabled=True,
            )
        )
        pc = run.perspective_capture
        perspectives_complete = (
            pc is not None
            and pc.status == "observed"
            and (len(pc.candidates) == len(run.protocol.plan.slots))
        )
        missing_phases = [
            name
            for name, ok in (("debate", debate_complete), ("perspectives", perspectives_complete))
            if not ok
        ]
        pair = run.protocol.pair
        rows.append(
            dict(
                run_id=run.run_id,
                slot=slot.to_dict(),
                status="observed",
                raw_run=run.to_dict(),
                pluralistic_report=pluralistic,
                causal_report=causal,
                debate_report=debate,
                combined_protocol_complete=slot.condition == CONDITIONS[-1] and not missing_phases,
                missing_phases=missing_phases if slot.condition == CONDITIONS[-1] else [],
                pairing_key=(
                    _pair_key(slot, run) if pair.before.system == pair.after.system else None
                ),
            )
        )
    paired, unpaired = _paired_summaries(rows)
    result = dict(
        schema_version="pluralistic-comparison-v1",
        training_eligibility="DEVELOPMENT",
        confers_authority=False,
        optimizer_input=False,
        training_effect_established=False,
        plan=plan.to_dict(),
        run_rows=rows,
        paired=paired,
        unpaired=unpaired,
        conditions={
            c: _condition_summary(
                [s for s in plan.slots if s.condition == c],
                [r for r in rows if r["slot"]["condition"] == c],
            )
            for c in CONDITIONS
        },
        sampling_note=(
            "Dependent authored fixtures; paired deltas are descriptive, not significance tests."
        ),
    )
    result["result_digest"] = content_digest(result)
    return json.loads(canonical_json(result))
