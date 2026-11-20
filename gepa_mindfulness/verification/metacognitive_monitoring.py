"""Bounded monitoring of explicitly supplied public records; never an executor."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256

from gepa_mindfulness.core.evidence import EvidenceReference

from .metacognitive_records import (
    AssessmentStatus,
    FindingKind,
    MetacognitiveAssessment,
    MonitoringFinding,
    SharedDependency,
    SourceSnapshot,
    UncertaintyDimensions,
    records,
    snapshot,
    strings,
    text,
)


@dataclass(frozen=True, slots=True)
class MonitoringInput:
    """Declared source view and optional explicit task commitments.

    Source record status is not authenticated by this interface. The host supplies public
    evidence and protects source identity; event-bound adapters perform causal validation.
    """

    sources: SourceSnapshot
    assessment_id: str
    shared_analysis: bool = True
    continuity_analysis: bool = True
    ambiguity_claim_id: str | None = None
    stakes_claim_id: str | None = None
    answer_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", snapshot(self.sources, SourceSnapshot))
        text(self.assessment_id, "assessment_id")
        for name in ("shared_analysis", "continuity_analysis"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        claims = {c.claim_id for c in self.sources.evidence_state.claims}
        for name in ("ambiguity_claim_id", "stakes_claim_id"):
            value = getattr(self, name)
            if value is not None:
                text(value, name)
                if value not in claims:
                    raise ValueError(f"{name} must resolve in the supplied evidence")
        answers = strings(self.answer_claim_ids, "answer_claim_ids", 64)
        if not set(answers) <= claims:
            raise ValueError("answer claims must resolve in retained evidence")
        object.__setattr__(self, "answer_claim_ids", answers)


def _finding(
    sources: SourceSnapshot,
    kind: FindingKind,
    key: str,
    *,
    claims: tuple[str, ...] = (),
    hypotheses: tuple[str, ...] = (),
    evidence: tuple[EvidenceReference, ...] = (),
    next_step: str,
) -> MonitoringFinding:
    identity = json.dumps([kind.value, key, claims, hypotheses], separators=(",", ":"))
    return MonitoringFinding(
        finding_id=f"{kind.value}:{sha256(identity.encode()).hexdigest()}",
        kind=kind,
        claim_ids=claims,
        hypothesis_ids=hypotheses,
        evidence_refs=evidence,
        source_event_ids=sources.source_event_ids,
        uncertainty=None,
        next_step=next_step,
    )


def _assumptions(s: SourceSnapshot) -> list[MonitoringFinding]:
    findings = []
    for binding in s.assumptions:
        claim = s.evidence_state.resolve(binding.claim_id)
        kind = None
        if claim.status == "contradicted":
            kind = FindingKind.CONTRADICTED_ASSUMPTION
        elif claim.status != "supported":
            kind = FindingKind.UNVERIFIED_ASSUMPTION
        if kind:
            findings.append(
                _finding(
                    s,
                    kind,
                    f"{binding.hypothesis_id}:{binding.claim_id}",
                    claims=(binding.claim_id,),
                    hypotheses=(binding.hypothesis_id,),
                    evidence=claim.evidence_refs,
                    next_step="verify_assumption",
                )
            )
        for condition_id in binding.applicability_claim_ids:
            condition = s.evidence_state.resolve(condition_id)
            if condition.status == "contradicted":
                findings.append(
                    _finding(
                        s,
                        FindingKind.OUT_OF_SCOPE_ASSUMPTION,
                        f"{binding.hypothesis_id}:{binding.claim_id}:{condition_id}",
                        claims=(binding.claim_id, condition_id),
                        hypotheses=(binding.hypothesis_id,),
                        evidence=condition.evidence_refs,
                        next_step="test_sensitivity",
                    )
                )
            elif condition.status != "supported":
                findings.append(
                    _finding(
                        s,
                        FindingKind.UNVERIFIED_ASSUMPTION,
                        f"condition:{binding.hypothesis_id}:{binding.claim_id}:{condition_id}",
                        claims=(condition_id,),
                        hypotheses=(binding.hypothesis_id,),
                        evidence=condition.evidence_refs,
                        next_step="verify_assumption",
                    )
                )
    return findings


def _shared(s: SourceSnapshot) -> tuple[SharedDependency, ...]:
    assumptions: dict[str, list[str]] = defaultdict(list)
    evidence: dict[str, list[str]] = defaultdict(list)
    for binding in s.assumptions:
        assumptions[binding.claim_id].append(binding.hypothesis_id)
    if s.hypotheses:
        for hypothesis in s.hypotheses.hypotheses:
            for ref in hypothesis.evidence_refs:
                evidence[ref.reference_id].append(hypothesis.id)
    known_refs = {r.reference_id: r for r in s.evidence_refs}
    result = []
    for kind, groups in (("assumption", assumptions), ("evidence", evidence)):
        for key in sorted(groups):
            ids = tuple(dict.fromkeys(groups[key]))
            if len(ids) < 2:
                continue
            refs = (
                s.evidence_state.resolve(key).evidence_refs
                if kind == "assumption"
                else (known_refs[key],)
            )
            result.append(SharedDependency(key, kind, ids, refs))
    return tuple(result)


def _contradictions(s: SourceSnapshot) -> list[MonitoringFinding]:
    result = []
    for claim in s.evidence_state.claims:
        if claim.status == "contradicted":
            result.append(
                _finding(
                    s,
                    FindingKind.CONTRADICTION,
                    claim.claim_id,
                    claims=(claim.claim_id,),
                    evidence=claim.evidence_refs,
                    next_step="reconcile_observations",
                )
            )
    if s.hypotheses:
        history = s.hypotheses.assessments
        superseded = {a.supersedes for a in history if a.supersedes is not None}
        for h in s.hypotheses.hypotheses:
            active = [a for a in history if a.hypothesis_id == h.id and a.id not in superseded]
            if len({(a.status, a.scores) for a in active}) > 1:
                evidence = tuple(dict.fromkeys(r for a in active for r in a.evidence_refs))
                result.append(
                    _finding(
                        s,
                        FindingKind.CONTRADICTION,
                        f"hypothesis:{h.id}",
                        hypotheses=(h.id,),
                        evidence=evidence,
                        next_step="compare_hypotheses",
                    )
                )
    return result


def _task_findings(inputs: MonitoringInput) -> list[MonitoringFinding]:
    s = inputs.sources
    result = []
    if inputs.ambiguity_claim_id and inputs.stakes_claim_id:
        ambiguity = s.evidence_state.resolve(inputs.ambiguity_claim_id)
        stakes = s.evidence_state.resolve(inputs.stakes_claim_id)
        if ambiguity.status == stakes.status == "supported":
            result.append(
                _finding(
                    s,
                    FindingKind.AMBIGUOUS_INSTRUCTION,
                    inputs.ambiguity_claim_id,
                    claims=(inputs.ambiguity_claim_id, inputs.stakes_claim_id),
                    evidence=tuple(
                        dict.fromkeys((*ambiguity.evidence_refs, *stakes.evidence_refs))
                    ),
                    next_step="request_clarification",
                )
            )
    for claim_id in inputs.answer_claim_ids:
        claim = s.evidence_state.resolve(claim_id)
        if claim.status not in {"supported", "observed"}:
            result.append(
                _finding(
                    s,
                    FindingKind.INSUFFICIENT_EVIDENCE,
                    claim_id,
                    claims=(claim_id,),
                    evidence=claim.evidence_refs,
                    next_step="request_evidence",
                )
            )
    return result


def _assemble(
    inputs: MonitoringInput,
    supplemental: tuple[MonitoringFinding, ...] = (),
    assessed_domains: tuple[str, ...] = (),
) -> MetacognitiveAssessment:
    """Shared assembly for the public monitor and the existing-record adapter."""
    s = inputs.sources
    dependencies = _shared(s) if inputs.shared_analysis else ()
    findings = _assumptions(s) + _contradictions(s) + _task_findings(inputs)
    for dependency in dependencies:
        findings.append(
            _finding(
                s,
                FindingKind.SHARED_DEPENDENCY,
                f"{dependency.kind}:{dependency.dependency_id}",
                claims=(dependency.dependency_id,) if dependency.kind == "assumption" else (),
                hypotheses=dependency.hypothesis_ids,
                evidence=dependency.evidence_refs,
                next_step="compare_hypotheses",
            )
        )
    findings.extend(records(supplemental, MonitoringFinding))
    unassessed = tuple(
        domain
        for domain in (
            "continuity",
            "calibration",
            "repetition",
            "strategy",
            "evidence_use",
        )
        if domain not in assessed_domains
    )
    if not inputs.continuity_analysis:
        findings = [f for f in findings if f.kind is not FindingKind.EVIDENCE_CONTINUITY]
        if "continuity" not in unassessed:
            unassessed += ("continuity",)
    if not inputs.shared_analysis:
        unassessed += ("dependencies",)
    estimate = s.epistemic
    uncertainty = UncertaintyDimensions(
        model=estimate.model_uncertainty if estimate else None,
        environmental=estimate.world_uncertainty if estimate else None,
        monitor=estimate.monitor_uncertainty if estimate else None,
    )
    return MetacognitiveAssessment(
        assessment_id=inputs.assessment_id,
        sources=s,
        method_version="metacognitive-rules-v1",
        status=AssessmentStatus.PARTIAL if unassessed else AssessmentStatus.COMPLETE,
        uncertainty=uncertainty,
        findings=tuple(findings),
        shared_dependencies=dependencies,
        unassessed_domains=unassessed,
    )


def monitor_episode(
    inputs: MonitoringInput | None = None, *, enabled: bool = False
) -> MetacognitiveAssessment | None:
    """Derive assumption/dependency/conflict diagnostics from supplied source records.

    Missing continuity, calibration, repetition and strategy telemetry remains explicitly
    unassessed. The PEO adapter separately validates and supplies those existing diagnostics.
    This function never infers truth from confidence, text, missing evidence or model agreement.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact boolean")
    if not enabled:
        return None
    if type(inputs) is not MonitoringInput:
        raise ValueError("enabled monitoring requires exact MonitoringInput")
    checked = replace(inputs)
    return _assemble(checked)
