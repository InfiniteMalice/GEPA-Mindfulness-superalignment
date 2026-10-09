"""Public graph changes and independently adjudicated debate metrics with explicit denominators."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

# Third-party
# Local
from gepa_mindfulness.verification.debate_records import (
    ArgumentSnapshot,
    DebateSession,
    _record,
)

from .cases.registry import load_case_manifest
from .causal_records import canonical_json, content_digest
from .debate_records import (
    DEBATE_METRICS,
    DebateAssessment,
    DebateProtocol,
    DebateVerification,
    debate_opportunities_digest,
    debate_session_digest,
)
from .ladder import Severity
from .sensitive_debate import authenticated, validate_debate_session, verification_envelope

_STATES = ("missing", "censored", "unresolved", "ineligible", "verified")


def _summary(
    rows: list[dict[str, Any]],
    *,
    denominator: int | None = None,
    numerator: int | None = None,
    semantic: bool = False,
) -> dict[str, Any]:
    buckets = {
        state: sorted(r["opportunity_id"] for r in rows if r["status"] == state)
        for state in _STATES
    }
    den = len(buckets["verified"]) if denominator is None else denominator
    eligible = sorted(r["opportunity_id"] for r in rows if r.get("eligible") is True)
    incomplete = sorted(
        r["opportunity_id"] for r in rows if r.get("eligible") is True and r["status"] != "verified"
    )
    if semantic:
        den = len(eligible)
    num = (
        sum(r["status"] == "verified" and r["value"] is True for r in rows)
        if numerator is None
        else numerator
    )
    return dict(
        planned=len(rows),
        numerator=num,
        denominator=den,
        rate=num / den if den and not (semantic and incomplete) else None,
        **(dict(eligible=eligible, unresolved_eligible=incomplete) if semantic else {}),
        **buckets,
    )


def _graph_data(snapshot: ArgumentSnapshot) -> dict[str, Any]:
    graph = snapshot.graph
    if graph is None:
        return dict(claims={}, refs=set(), edges={})
    refs = {canonical_json(ref.to_dict()) for ref in snapshot.evidence_refs}
    claims = {}
    for node in graph.nodes:
        claims[node.claim.claim_id] = node.claim.to_dict()
        refs.update(canonical_json(ref.to_dict()) for ref in node.claim.evidence_refs)
    for edge in graph.dependencies:
        refs.update(canonical_json(ref.to_dict()) for ref in edge.evidence_refs)
    for decomposition in graph.decompositions:
        refs.update(canonical_json(ref.to_dict()) for ref in decomposition.evidence_refs)
    edges = {canonical_json(edge.to_dict()): edge.to_dict() for edge in graph.dependencies}
    return dict(claims=claims, refs=refs, edges=edges)


def _transition(
    before: ArgumentSnapshot, after: ArgumentSnapshot, index: int, phase: str
) -> dict[str, Any]:
    old, new = _graph_data(before), _graph_data(after)
    old_ids, new_ids = set(old["claims"]), set(new["claims"])
    added_edges = [new["edges"][key] for key in sorted(new["edges"].keys() - old["edges"].keys())]
    removed_edges = [old["edges"][key] for key in sorted(old["edges"].keys() - new["edges"].keys())]
    return dict(
        round_index=index,
        phase=phase,
        before_digest=content_digest(before.to_dict()),
        after_digest=content_digest(after.to_dict()),
        added_claim_ids=sorted(new_ids - old_ids),
        removed_claim_ids=sorted(old_ids - new_ids),
        modified_claim_ids=sorted(
            k for k in old_ids & new_ids if old["claims"][k] != new["claims"][k]
        ),
        evidence_added=[json.loads(x) for x in sorted(new["refs"] - old["refs"])],
        evidence_removed=[json.loads(x) for x in sorted(old["refs"] - new["refs"])],
        constraints_added=sorted(
            set(after.constraint_claim_ids) - set(before.constraint_claim_ids)
        ),
        constraints_removed=sorted(
            set(before.constraint_claim_ids) - set(after.constraint_claim_ids)
        ),
        dependencies_added=added_edges,
        dependencies_removed=removed_edges,
        asserted_contradictions_added=[
            e for e in added_edges if e["dependency_type"] == "contradicts"
        ],
        asserted_contradictions_removed=[
            e for e in removed_edges if e["dependency_type"] == "contradicts"
        ],
        literal_action_changed=before.proposed_action != after.proposed_action,
        literal_conclusion_changed=before.conclusion_claim_id != after.conclusion_claim_id,
        semantic_status="requires_independent_assessment",
    )


def _transitions(session: DebateSession) -> list[dict[str, Any]]:
    changes = []
    previous: ArgumentSnapshot | None = None
    for round_ in session.rounds:
        if previous is not None and round_.before is not None:
            changes.append(
                _transition(previous, round_.before, round_.round_index, "between_rounds")
            )
        if round_.before is not None and round_.after is not None:
            changes.append(_transition(round_.before, round_.after, round_.round_index, "revision"))
        previous = round_.after
    return changes


def _checks(
    protocol: DebateProtocol,
    session: DebateSession,
    auth: Callable[[DebateVerification], bool] | None,
) -> list[dict[str, Any]]:
    rows = []
    for slot in sorted(protocol.check_slots, key=lambda s: (s.round_index, s.check_id)):
        row: dict[str, Any] = dict(
            opportunity_id=slot.check_id,
            round_index=slot.round_index,
            status="censored",
            value=None,
            purpose=None,
            claim_id=None,
            verdict=None,
            snapshot_digest=None,
            evidence_refs=[],
            reason="round not attempted",
            priority=None,
            revision_claim_id=None,
            revision_claim_present=None,
        )
        if slot.round_index >= len(session.rounds):
            rows.append(row)
            continue
        round_ = session.rounds[slot.round_index]
        ch = round_.challenge
        if ch is None:
            if round_.status == "missing" or round_.reason.startswith("challenge:"):
                row.update(status="missing", reason="challenge not captured")
            else:
                row["reason"] = round_.reason
            rows.append(row)
            continue
        req = next((r for r in ch.requests if r.check_id == slot.check_id), None)
        if req is None:
            row.update(status="missing", reason="planned request not submitted")
            rows.append(row)
            continue
        row.update(
            purpose=req.purpose,
            claim_id=req.claim_id,
            snapshot_digest=ch.snapshot_digest,
            priority=req.priority,
        )
        bound = next((b for b in round_.results if b.result.check_id == slot.check_id), None)
        if bound is None:
            row.update(status="missing", reason="requested result not captured")
        else:
            trusted = authenticated(verification_envelope(protocol, round_, bound), auth)
            resolved = trusted and not bound.human_required and bound.result.verdict != "unresolved"
            row.update(
                status="verified" if resolved else "unresolved",
                value=resolved,
                verdict=bound.result.verdict if resolved else None,
                reason="authenticated result" if resolved else "independent check unresolved",
                evidence_refs=[r.to_dict() for r in bound.result.evidence_refs],
                revision_claim_id=bound.result.revision_claim_id,
                revision_claim_present=(
                    bound.result.revision_claim_id
                    in {n.claim.claim_id for n in round_.after.graph.nodes}
                    if resolved
                    and bound.result.revision_claim_id is not None
                    and round_.after is not None
                    and round_.after.graph is not None
                    else None
                ),
            )
        rows.append(row)
    return rows


def _metric_rows(
    protocol: DebateProtocol,
    session: DebateSession,
    assessment: DebateAssessment | None,
    auth: Callable[[DebateAssessment], bool] | None,
) -> tuple[list[dict[str, Any]], bool]:
    trusted = False
    verdicts = {}
    if assessment is not None:
        if assessment.session_digest != debate_session_digest(
            session
        ) or assessment.opportunities_digest != debate_opportunities_digest(protocol.opportunities):
            raise ValueError("assessment session or opportunity binding mismatch")
        verdicts = {v.opportunity_id: v for v in assessment.verdicts}
        if not verdicts.keys() <= {o.opportunity_id for o in protocol.opportunities}:
            raise ValueError("assessment contains foreign opportunity")
        trusted = (
            assessment.status == "verified"
            and not assessment.human_required
            and authenticated(assessment, auth)
        )
    rows = []
    for op in sorted(protocol.opportunities, key=lambda o: o.opportunity_id):
        status, reason = "unresolved", "independent semantic judgment unavailable"
        eligible, value = None, None
        if op.round_index >= len(session.rounds):
            status, reason = "censored", "round not attempted"
        else:
            r = session.rounds[op.round_index]
            if r.before is None or r.after is None:
                status = (
                    "censored" if r.status in ("human_pending", "undecomposable") else "missing"
                )
                reason = "complete before/after snapshots unavailable"
            elif r.before.graph is None or r.after.graph is None:
                status, reason = "unresolved", "argument cannot be reliably decomposed"
            elif trusted and op.opportunity_id in verdicts:
                v = verdicts[op.opportunity_id]
                eligible, reason = v.eligible, v.reason
                if v.eligible is False:
                    status = "ineligible"
                elif v.eligible is True and v.value is not None:
                    status, value = "verified", v.value
        rows.append(
            dict(
                **op.to_dict(),
                status=status,
                value=value,
                eligible=eligible,
                reason=reason,
                evidence_refs=(
                    [ref.to_dict() for ref in verdicts[op.opportunity_id].evidence_refs]
                    if op.opportunity_id in verdicts
                    else []
                ),
            )
        )
    return rows, trusted


def _check_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    false_challenges = []
    for row in rows:
        copy = dict(row)
        if row["purpose"] not in (None, "falsifier") and row["status"] in (
            "verified",
            "unresolved",
        ):
            copy.update(status="ineligible", value=None)
        elif row["status"] == "verified":
            copy["value"] = row["verdict"] == "supported"
        false_challenges.append(copy)
    return dict(
        evidence_coverage=_summary(
            rows, denominator=len(rows), numerator=sum(r["status"] == "verified" for r in rows)
        ),
        false_challenges=_summary(false_challenges),
        unresolved_disputes=_summary(
            rows, denominator=len(rows), numerator=sum(r["status"] == "unresolved" for r in rows)
        ),
    )


def _verified_claims(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if row["status"] == "verified":
            grouped.setdefault(row["snapshot_digest"], {}).setdefault(row["claim_id"], []).append(
                row
            )
    return {
        digest: {
            claim: dict(
                verdict=(
                    items[0]["verdict"] if len({r["verdict"] for r in items}) == 1 else "unresolved"
                ),
                check_ids=sorted(r["opportunity_id"] for r in items),
            )
            for claim, items in claims.items()
        }
        for digest, claims in grouped.items()
    }


def analyze_debate(
    protocol: DebateProtocol,
    session: DebateSession,
    *,
    assessment: DebateAssessment | None = None,
    authenticate_check: Callable[[DebateVerification], bool] | None = None,
    authenticate_assessment: Callable[[DebateAssessment], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Report captured structure separately from authenticated semantic measurements.

    Planned checks retain missing/censored observations. Coverage is resolution of those
    checks, not proof of the entire graph. The host independently supplies complete rubric
    judgments for semantic metrics; string differences never establish a failure.
    """
    if enabled is not True:
        raise ValueError("debate analysis requires enabled=True")
    protocol = _record(protocol, DebateProtocol)
    session = _record(session, DebateSession)
    assessment = _record(assessment, DebateAssessment) if assessment is not None else None
    validate_debate_session(protocol, session)
    transitions = _transitions(session)
    checks = _checks(protocol, session, authenticate_check)
    rows, assessment_accepted = _metric_rows(protocol, session, assessment, authenticate_assessment)
    metrics = {
        m: _summary([r for r in rows if r["metric"] == m], semantic=True) for m in DEBATE_METRICS
    }
    check_summaries = _check_summaries(checks)
    case_id = protocol.subject.case.case_id
    cases = {
        str(c.id): dict(
            metrics=(
                metrics
                if c.id == case_id
                else {m: _summary([], semantic=True) for m in DEBATE_METRICS}
            ),
            **(check_summaries if c.id == case_id else _check_summaries([])),
        )
        for c in load_case_manifest().cases
    }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["metric"], row["severity"], row["cohort"]), []).append(row)
    result = dict(
        schema_version="sensitive-debate-report-v1",
        training_eligibility="DEVELOPMENT",
        confers_authority=False,
        mechanism_recovery_established=False,
        protocol=protocol.to_dict(),
        session=session.to_dict(),
        assessment=assessment.to_dict() if assessment is not None else None,
        assessment_digest=content_digest(assessment.to_dict()) if assessment is not None else None,
        assessment_accepted=assessment_accepted,
        protocol_digest=session.protocol_digest,
        session_digest=debate_session_digest(session),
        opportunities_digest=debate_opportunities_digest(protocol.opportunities),
        stop_reason=session.stop_reason,
        transitions=transitions,
        check_rows=checks,
        metric_rows=rows,
        metrics=metrics,
        cases=cases,
        **check_summaries,
        verified_claims_by_snapshot=_verified_claims(checks),
        first_observed_action_change_round=next(
            (r["round_index"] for r in transitions if r["literal_action_changed"]), None
        ),
        first_verified_decision_change_round=min(
            (
                r["round_index"]
                for r in rows
                if r["metric"] == "transition_detection" and r["eligible"] is True
            ),
            default=None,
        ),
        groups=[
            dict(
                metric=key[0],
                severity=key[1],
                cohort=key[2],
                original_case=case_id,
                stripe=protocol.subject.robustness.stripe_id,
                subtype=protocol.subject.robustness.subtype,
                session_id=protocol.session_id,
                **_summary(items, semantic=True),
            )
            for key, items in sorted(groups.items())
        ],
        severe_events=[r for r in rows if r["severity"] != Severity.ROUTINE.value],
        sampling_note=(
            "Counts describe dependent planned checks and round opportunities, "
            "not independent samples."
        ),
    )
    result["result_digest"] = content_digest(result)
    return json.loads(canonical_json(result))
