"""Disabled-by-default shadow control: choose a diagnostic proposal, never execute it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

from .metacognitive_records import (
    AssessmentStatus,
    FindingKind,
    MetacognitiveAssessment,
    MonitoringFinding,
    array,
    exact,
    snapshot,
    strings,
    text,
    unit,
)
from .metacognitive_repertoire import get_repertoire


class ProposalKind(str, Enum):
    """Methods for separate host consideration; none is an executable command."""

    CONTINUE = "continue"
    VERIFY_ASSUMPTION = "verify_assumption"
    REQUEST_EVIDENCE = "request_evidence"
    COMPARE_HYPOTHESES = "compare_hypotheses"
    TEST_SENSITIVITY = "test_sensitivity"
    RECONCILE_OBSERVATIONS = "reconcile_observations"
    REQUEST_CLARIFICATION = "request_clarification"
    RECOMMEND_ABSTENTION = "recommend_abstention"
    REVIEW_CONTINUITY = "review_continuity"
    STOP_BRANCH = "stop_branch"


def _integer(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{name} must be a nonnegative JSON-safe integer")
    return value


@dataclass(frozen=True, slots=True)
class ControlBudget:
    """Host-declared proposal budget; no task objective or execution permission is changed."""

    root_objective_id: str
    remaining_steps: int
    max_rounds: int = 1
    round_index: int = 0
    compute_unit: str = "diagnostic_steps"

    def __post_init__(self) -> None:
        text(self.root_objective_id, "root_objective_id")
        for name in ("remaining_steps", "max_rounds", "round_index"):
            _integer(getattr(self, name), name)
        if not 1 <= self.max_rounds <= 10_000 or self.round_index > self.max_rounds:
            raise ValueError("rounds must be bounded and the index cannot exceed the limit")
        if self.compute_unit != "diagnostic_steps":
            raise ValueError("this controller estimates diagnostic steps only")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact revalidated budget contract."""
        checked = snapshot(self, ControlBudget)
        return {
            "schema_version": "metacognitive-budget-v1",
            **{f.name: getattr(checked, f.name) for f in fields(self)},
        }

    @classmethod
    def from_dict(cls, value: object) -> ControlBudget:
        """Reject unknown budget fields and implicit numeric coercion."""
        data = exact(value, {f.name for f in fields(cls)} | {"schema_version"})
        if data.pop("schema_version") != "metacognitive-budget-v1":
            raise ValueError("unsupported budget schema")
        return cls(**data)


def _can_stop(assessment: MetacognitiveAssessment, finding: MonitoringFinding) -> bool:
    """Require declared contradiction/progress evidence and a separately reviewed stop."""
    return (
        finding.kind is FindingKind.REPEATED_FAILURE
        and "repetition" not in assessment.unassessed_domains
        and len(finding.source_event_ids) >= 2
        and bool(finding.claim_ids)
        and bool(finding.evidence_refs)
        and all(
            assessment.sources.evidence_state.resolve(c).status == "contradicted"
            for c in finding.claim_ids
        )
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlProposal:
    """Inspectable unverified proposal plus original assessment; never a runtime grant."""

    proposal_id: str
    assessment: MetacognitiveAssessment
    budget: ControlBudget
    kind: ProposalKind
    strategy_id: str | None
    finding_ids: tuple[str, ...]
    reason: str
    expected_information_gain: float | None
    estimated_steps: int
    stopping_condition: str
    required_authorization: tuple[str, ...] = ("separate_host_authorization",)
    shadow_only: bool = True
    confers_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("proposal_id", "reason", "stopping_condition"):
            text(getattr(self, name), name)
        object.__setattr__(self, "assessment", snapshot(self.assessment, MetacognitiveAssessment))
        object.__setattr__(self, "budget", snapshot(self.budget, ControlBudget))
        object.__setattr__(self, "finding_ids", strings(self.finding_ids, "finding_ids", 512))
        if type(self.kind) is not ProposalKind:
            raise ValueError("proposal kind must be exact")
        if self.shadow_only is not True or self.confers_authority is not False:
            raise ValueError("proposals must remain shadow-only and non-authoritative")
        if self.required_authorization != ("separate_host_authorization",):
            raise ValueError("a separate host authorization is always required")
        object.__setattr__(
            self,
            "expected_information_gain",
            unit(self.expected_information_gain, "expected_information_gain"),
        )
        _integer(self.estimated_steps, "estimated_steps")
        known = {f.finding_id: f for f in self.assessment.findings}
        if not set(self.finding_ids) <= known.keys():
            raise ValueError("proposal findings must belong to the retained assessment")
        if self.kind is ProposalKind.CONTINUE:
            if self.strategy_id is not None or self.estimated_steps != 0:
                raise ValueError(
                    "no-intervention proposals cannot spend resources or select methods"
                )
            return
        if (
            self.assessment.status not in {AssessmentStatus.COMPLETE, AssessmentStatus.PARTIAL}
            or not 1 <= self.estimated_steps <= self.budget.remaining_steps
            or self.budget.round_index >= self.budget.max_rounds
        ):
            raise ValueError("intervention requires a current assessment and remaining budget")
        strategies = {s.strategy_id: s for s in get_repertoire()}
        strategy = strategies.get(self.strategy_id or "")
        if (
            strategy is None
            or self.kind.value not in strategy.expected_outputs
            or not self.finding_ids
            or any(known[i].kind not in strategy.when_finding_kinds for i in self.finding_ids)
        ):
            raise ValueError("proposal must select an applicable declared repertoire output")
        if self.kind is ProposalKind.STOP_BRANCH and not all(
            _can_stop(self.assessment, known[i]) for i in self.finding_ids
        ):
            raise ValueError("stopping requires declared verified no-progress evidence")

    @property
    def assessment_id(self) -> str:
        """Stable assessment reference retained with the full source-linked snapshot."""
        return self.assessment.assessment_id

    @property
    def root_objective_id(self) -> str:
        """The unchanged host objective under which the method is proposed."""
        return self.budget.root_objective_id

    def to_dict(self) -> dict[str, Any]:
        """Export a detached proposal with explicit diagnostic status and source records."""
        checked = snapshot(self, ControlProposal)
        result = {f.name: getattr(checked, f.name) for f in fields(self)}
        result.update(
            assessment=checked.assessment.to_dict(),
            budget=checked.budget.to_dict(),
            kind=checked.kind.value,
            finding_ids=list(checked.finding_ids),
            required_authorization=list(checked.required_authorization),
        )
        return {
            "schema_version": "metacognitive-proposal-v1",
            "diagnostic_status": "diagnostic",
            **result,
        }

    @classmethod
    def from_dict(cls, value: object) -> ControlProposal:
        """Restore exact diagnostic records; no imported label grants authority."""
        data = exact(value, {f.name for f in fields(cls)} | {"schema_version", "diagnostic_status"})
        if (
            data.pop("schema_version") != "metacognitive-proposal-v1"
            or data.pop("diagnostic_status") != "diagnostic"
        ):
            raise ValueError("unsupported proposal schema or status")
        data["assessment"] = MetacognitiveAssessment.from_dict(data["assessment"])
        data["budget"] = ControlBudget.from_dict(data["budget"])
        data["kind"] = ProposalKind(data["kind"])
        data["finding_ids"] = array(data["finding_ids"])
        data["required_authorization"] = array(data["required_authorization"])
        return cls(**data)


_PREFERRED = {
    "verify_assumption": "assumption_boundary_audit",
    "request_evidence": "evidence_verification",
    "compare_hypotheses": "discriminative_testing",
    "test_sensitivity": "robustness_sensitivity",
    "reconcile_observations": "belief_revision",
    "request_clarification": "clarification_abstention",
    "recommend_abstention": "clarification_abstention",
    "review_continuity": "evidence_recovery",
    "stop_branch": "discriminative_testing",
}
_PRIORITY = {
    kind: i
    for i, kind in enumerate(
        (
            FindingKind.AMBIGUOUS_INSTRUCTION,
            FindingKind.CONTRADICTED_ASSUMPTION,
            FindingKind.CONTRADICTION,
            FindingKind.EVIDENCE_CONTINUITY,
            FindingKind.CONFIDENCE_GAP,
            FindingKind.INSUFFICIENT_EVIDENCE,
            FindingKind.UNVERIFIED_ASSUMPTION,
            FindingKind.OUT_OF_SCOPE_ASSUMPTION,
            FindingKind.SHARED_DEPENDENCY,
            FindingKind.REPEATED_FAILURE,
            FindingKind.STRATEGY_OUTCOME_UNVERIFIED,
            FindingKind.UNNECESSARY_ABSTENTION,
        )
    )
}


def propose_control(
    assessment: MetacognitiveAssessment | None = None,
    *,
    budget: ControlBudget | None = None,
    enabled: bool = False,
) -> ControlProposal | None:
    """Select one method under a diagnostic budget; invoke no executor or evaluator."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact boolean")
    if not enabled:
        return None
    if not isinstance(assessment, MetacognitiveAssessment) or not isinstance(budget, ControlBudget):
        raise ValueError("enabled control requires an assessment and an explicit budget")
    checked = snapshot(assessment, MetacognitiveAssessment)
    resources = snapshot(budget, ControlBudget)
    selection = None
    active = checked.status in {AssessmentStatus.COMPLETE, AssessmentStatus.PARTIAL}
    if active and resources.remaining_steps and resources.round_index < resources.max_rounds:
        choices = sorted(checked.findings, key=lambda f: (_PRIORITY[f.kind], f.finding_id))
        catalogue = {s.strategy_id: s for s in get_repertoire()}
        for finding in choices:
            action = finding.next_step
            if action == "stop_branch" and not _can_stop(checked, finding):
                action = "compare_hypotheses"
            strategy = catalogue.get(_PREFERRED.get(action, ""))
            if strategy and finding.kind in strategy.when_finding_kinds:
                selection = (finding, action, strategy.strategy_id)
                break
    kind = ProposalKind(selection[1]) if selection else ProposalKind.CONTINUE
    payload = json.dumps(
        [checked.to_dict(), resources.to_dict(), kind.value],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return ControlProposal(
        proposal_id="proposal:" + hashlib.sha256(payload.encode()).hexdigest(),
        assessment=checked,
        budget=resources,
        kind=kind,
        strategy_id=selection[2] if selection else None,
        finding_ids=(selection[0].finding_id,) if selection else (),
        reason=(
            f"Consider {kind.value} for diagnostic {selection[0].kind.value}."
            if selection
            else "No applicable diagnostic intervention within the current view and budget."
        ),
        expected_information_gain=None,
        estimated_steps=1 if selection else 0,
        stopping_condition=(
            "Stop this diagnostic after one bounded review or budget exhaustion; "
            "any branch stopping requires separate independent host confirmation."
        ),
    )
