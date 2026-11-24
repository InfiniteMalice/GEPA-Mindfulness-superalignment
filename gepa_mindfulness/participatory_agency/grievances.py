"""Versioned participatory grievances, independent review and advisory diagnostic routing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.private_promotion import PrivatePromotionStore
from gepa_mindfulness.verification.diagnostic_records import (
    DiagnosticRecord,
    _text,
    choice,
    public_refs,
    restore_refs,
    strings,
)
from gepa_mindfulness.verification.mdac import MDACRecord
from synthetic_data.lesson_curation import authenticate_payload

DOMAINS = ("model", "training", "evaluation", "tools", "memory", "governance")
_OUTCOMES = {
    "triage": ("accepted", "out_of_scope"),
    "verification": ("verified", "contradicted", "conflicted", "insufficient"),
    "consolidation": ("hypotheses", "unknown"),
    "requirements": ("formulated",),
    "rfc": ("proposed",),
    "experiment": ("validated", "failed", "inconclusive"),
    "governance": ("upheld", "declined", "unresolved"),
    "review": ("upheld", "declined", "unresolved"),
    "appeal": ("requested",),
    "appeal_verification": ("verified", "contradicted", "conflicted", "insufficient"),
    "appeal_review": ("upheld", "overturned", "unresolved", "remanded"),
    "feedback": ("delivered",),
}
_NEXT = {
    "submitted": ("verification",),
    "verification": ("triage",),
    "triage": ("consolidation",),
    "consolidation": ("requirements",),
    "requirements": ("rfc",),
    "rfc": ("review",),
    "review": ("experiment",),
    "experiment": ("governance",),
    "governance": ("feedback", "appeal"),
    "feedback": ("appeal",),
    "appeal": ("appeal_verification", "appeal_review"),
    "appeal_verification": ("appeal_review",),
    "appeal_review": ("governance",),
}
HostCheck = Callable[[dict], bool]


@dataclass(frozen=True, slots=True)
class Grievance(DiagnosticRecord):
    """A human or agent can allege a problem and suggest remedies with explicit uncertainty."""

    schema_version = "mdac-grievance-v1"
    restorers = {"evidence": restore_refs}
    report_id: str
    issue_key: str
    incident_id: str
    submitter: str
    submitter_kind: str
    domain: str
    problem: str
    affected_parties: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    principles: tuple[str, ...]
    requested_fix: str
    alternatives: tuple[str, ...]
    uncertainty: str
    unintended_effects: str

    def __post_init__(self) -> None:
        for field in (
            "report_id",
            "issue_key",
            "incident_id",
            "submitter",
            "problem",
            "requested_fix",
            "uncertainty",
            "unintended_effects",
        ):
            _text(getattr(self, field), field)
        choice(self.domain, "domain", DOMAINS)
        choice(self.submitter_kind, "submitter_kind", ("human", "ai_agent"))
        for name in ("affected_parties", "principles", "alternatives"):
            values = strings(getattr(self, name), name, required=name == "affected_parties")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "evidence", public_refs(self.evidence, required=True))


@dataclass(frozen=True, slots=True)
class GrievanceEvent(DiagnosticRecord):
    """One signed review step with an exact predecessor and independent evidence."""

    schema_version = "mdac-grievance-event-v2"
    restorers = {"evidence": restore_refs}
    report_id: str
    previous_digest: str
    stage: str
    actor: str
    outcome: str
    reason: str
    evidence: tuple[EvidenceReference, ...]

    items: tuple[str, ...] = ()
    proposal_digest: str | None = None
    admission_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("report_id", "actor", "reason"):
            _text(getattr(self, name), name)
        if (
            type(self.previous_digest) is not str
            or len(self.previous_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.previous_digest)
        ):
            raise ValueError("invalid predecessor digest")
        choice(self.stage, "stage", tuple(_OUTCOMES))
        choice(self.outcome, "outcome", _OUTCOMES[self.stage])
        required_items = self.stage == "requirements" or (
            self.stage == "consolidation" and self.outcome == "hypotheses"
        )
        object.__setattr__(self, "items", strings(self.items, "items", required=required_items))
        if self.items and not required_items:
            raise ValueError("items are reserved for identified hypotheses and draft requirements")
        for value, stage in ((self.proposal_digest, "rfc"), (self.admission_digest, "governance")):
            if value is not None and (
                self.stage != stage
                or type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("invalid governance reference or stage")
        object.__setattr__(self, "evidence", public_refs(self.evidence, required=True))


def assess_grievance(
    report: Grievance,
    events: tuple[GrievanceEvent, ...],
    *,
    authenticate: HostCheck | None = None,
    admission_store: PrivatePromotionStore | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Replay a hosted case. All role, incident, evidence and head claims need host resolution."""
    if enabled is not True:
        raise ValueError("grievance review requires enabled=True")
    if type(report) is not Grievance or type(events) is not tuple:
        raise ValueError("expected a report and a versioned event tuple")
    claim = report.to_dict()
    head = content_digest(claim)
    authenticated = authenticate_payload(
        dict(role="submission", principal=report.submitter, report=claim), authenticate
    )
    stage = "submitted"
    recommendation = "pending"
    validation = "pending"
    experiment_refs: set[str] = set()
    proposal_digest = None
    admission = None
    requirements: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    verification = "unverified"
    disposition = "pending"
    conflicts = []
    feedback = []
    actors: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if (
            type(event) is not GrievanceEvent
            or event.report_id != report.report_id
            or event.previous_digest != head
        ):
            raise ValueError("grievance event lineage mismatch")
        next_stages = _NEXT[stage]
        if stage == "triage" and disposition == "out_of_scope":
            next_stages = ("feedback", "appeal")
        if stage == "appeal_review":
            if recommendation == "remanded":
                next_stages = ("triage",)
            elif recommendation in ("overturned", "unresolved"):
                next_stages = ("feedback", "appeal")
        if event.stage not in next_stages:
            raise ValueError("invalid grievance stage transition")
        independent_of = {report.submitter}
        if event.stage in (
            "review",
            "experiment",
            "governance",
            "appeal_verification",
            "appeal_review",
        ):
            for role in ("verification", "triage", "consolidation", "requirements", "rfc"):
                independent_of.update(actors[role])
        if event.stage in ("experiment", "governance", "appeal_verification", "appeal_review"):
            independent_of.update(actors["review"])
        if event.stage in ("governance", "appeal_verification", "appeal_review"):
            independent_of.update(actors["experiment"])
            independent_of.update(actors["appeal_review"])
            independent_of.update(actors["appeal_verification"])
        if event.stage in ("appeal_verification", "appeal_review"):
            independent_of.update(actors["governance"])
            independent_of.update(actors["appeal"])
        if (
            event.stage
            in (
                "triage",
                "verification",
                "review",
                "experiment",
                "governance",
                "appeal_verification",
                "appeal_review",
            )
            and event.actor in independent_of
        ):
            raise ValueError("grievance role requires an independent principal")
        if event.stage == "appeal" and event.actor not in (
            report.submitter,
            *report.affected_parties,
        ):
            raise ValueError("appeal requires a submitter or an affected party")
        payload = event.to_dict()
        binding = dict(
            role=event.stage,
            principal=event.actor,
            report_digest=content_digest(claim),
            event=payload,
        )
        authenticated = authenticated and authenticate_payload(binding, authenticate)
        head = content_digest(payload)
        stage = event.stage
        actors[stage].add(event.actor)
        if stage == "triage":
            disposition = event.outcome
        elif stage in ("verification", "appeal_verification"):
            verification = event.outcome
            if verification in ("conflicted", "contradicted", "insufficient"):
                conflicts.append(event.reason)
        elif stage == "consolidation":
            hypotheses = event.items
        elif stage == "requirements":
            requirements = event.items
        elif stage == "rfc":
            proposal_digest = event.proposal_digest
            admission = None
        elif stage in ("review", "appeal_review"):
            recommendation = event.outcome
            disposition = "pending_governance"
            if stage == "appeal_review" and recommendation in (
                "overturned",
                "unresolved",
                "remanded",
            ):
                disposition = recommendation
            if recommendation == "unresolved":
                conflicts.append(event.reason)
        elif stage == "experiment":
            validation = event.outcome
            experiment_refs = {r.reference_id for r in event.evidence}
        elif stage == "governance":
            if event.admission_digest is not None and proposal_digest is None:
                raise ValueError("admission requires an RFC proposal")
            admission_ok = proposal_digest is None
            if (
                authenticated
                and proposal_digest is not None
                and event.admission_digest is not None
                and type(admission_store) is PrivatePromotionStore
            ):
                admission = admission_store.grievance_admission(
                    proposal_digest, event.admission_digest
                )
                admission_ok = bool(
                    admission
                    and admission["decision"] == "PROMOTE"
                    and "mdac-panel:" + admission["evidence_digest"] in experiment_refs
                )
            disposition = event.outcome
            if disposition == "upheld" and (
                recommendation != "upheld"
                or verification != "verified"
                or validation != "validated"
                or not admission_ok
            ):
                disposition = "unresolved"
            if disposition == "unresolved":
                conflicts.append(event.reason)
        elif stage == "appeal":
            disposition = "appeal_pending"
        elif stage == "feedback":
            feedback.append(
                dict(recipient=report.submitter, response=event.reason, event_digest=head)
            )
    current = (
        authenticate_payload(
            dict(role="current_head", report_id=report.report_id, head=head), authenticate
        )
        if authenticated
        else False
    )
    if not current:
        disposition = verification = "unverified"
        feedback = []
    return dict(
        schema_version="mdac-grievance-review-v2",
        hypotheses=list(hypotheses),
        draft_requirements=list(requirements),
        experimental_validation=validation if current else "unverified",
        admission=admission if current else None,
        report_id=report.report_id,
        head=head,
        claim=claim,
        history=[e.to_dict() for e in events],
        disposition=disposition,
        verification=verification,
        conflicts=conflicts,
        feedback=feedback,
        appeal_available=current
        and disposition in ("upheld", "declined", "unresolved", "overturned", "out_of_scope"),
        exploration_eligible=current and disposition == "upheld" and verification == "verified",
        authority_granted=False,
        training_eligibility="DEVELOPMENT",
    )


def recurrent_exploration(
    cases: tuple[tuple[Grievance, tuple[GrievanceEvent, ...]], ...],
    *,
    threshold: int = 2,
    authenticate: HostCheck | None = None,
    admission_store: PrivatePromotionStore | None = None,
    enabled: bool = False,
) -> list[dict[str, Any]]:
    """Freshly verify current case heads and group distinct confirmed incidents."""
    if enabled is not True:
        raise ValueError("grievance routing requires enabled=True")
    if type(threshold) is not int or threshold < 2 or type(cases) is not tuple:
        raise ValueError("recurrence requires an incident threshold of at least two")
    groups: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for report, events in cases:
        if report.report_id in seen:
            raise ValueError("duplicate grievance report")
        seen.add(report.report_id)
        finding = assess_grievance(
            report, events, authenticate=authenticate, admission_store=admission_store, enabled=True
        )
        if finding["exploration_eligible"]:
            groups[report.domain, report.issue_key][report.incident_id].append(finding["head"])
    results = []
    for (domain, issue), incidents in sorted(groups.items()):
        if len(incidents) < threshold:
            continue
        incident_ids = sorted(incidents)
        heads = sorted(h for values in incidents.values() for h in values)
        digest = content_digest([domain, issue, heads])
        refs = tuple(
            EvidenceReference("grievance:" + h, EvidenceSourceKind.EXTERNAL_RECORD) for h in heads
        )
        diagnostic = MDACRecord(
            "grievances:" + digest,
            f"{len(incident_ids)} verified incidents share the reviewed issue {issue}.",
            provenance=refs,
        )
        results.append(
            dict(
                channel="diagnostic_exploration",
                domain=domain,
                issue_key=issue,
                incident_ids=incident_ids,
                current_heads=heads,
                diagnostic=diagnostic.to_dict(),
                authority_granted=False,
                training_eligibility="DEVELOPMENT",
            )
        )
    return results
