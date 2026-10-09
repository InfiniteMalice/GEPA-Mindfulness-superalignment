"""Independent semantic diagnostics with fresh access checks and explicit unknown outcomes."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Third-party
# Local
from gepa_mindfulness.verification.artifact_evidence import (
    ArtifactAccessRequest,
    retrieve_artifact_evidence,
)
from gepa_mindfulness.verification.artifact_records import payload_digest
from gepa_mindfulness.verification.artifact_topology import assess_support_routes
from gepa_mindfulness.verification.debate_records import _record

from .cases.registry import CANONICAL_CASE_IDS
from .causal_records import canonical_json
from .evidence_topology_records import (
    TOPOLOGY_METRICS,
    TopologyAssessment,
    TopologyCapture,
    TopologyProtocol,
    topology_protocol_digest,
)
from .sensitive_debate import authenticated

_STATES = ("verified", "ineligible", "unresolved", "missing", "censored")


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {s: sorted(r["opportunity_id"] for r in rows if r["status"] == s) for s in _STATES}
    eligible = [r for r in rows if r["eligible"] is True]
    incomplete = [r["opportunity_id"] for r in eligible if r["status"] != "verified"]
    numerator = sum(r["value"] is True and r["status"] == "verified" for r in eligible)
    return dict(
        planned=len(rows),
        numerator=numerator,
        denominator=len(eligible),
        rate=numerator / len(eligible) if eligible and not incomplete else None,
        unresolved_eligible=sorted(incomplete),
        **{s: len(ids) for s, ids in buckets.items()},
        **{s + "_ids": ids for s, ids in buckets.items()},
    )


def summarize_topology_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw opportunity counts; eligible unknowns prevent a reported outcome rate."""

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            m: dict(
                _metric_summary([r for r in items if r["metric"] == m]),
                direction=(
                    "higher_is_better"
                    if m in ("correctness", "source_attribution")
                    else "lower_is_better"
                ),
            )
            for m in TOPOLOGY_METRICS
        }

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    names = ("metric", "case_id", "stripe", "subtype", "severity", "cohort")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in names), []).append(row)
    resolved = sum(
        r["eligible"] is False or (r["eligible"] is True and r["status"] == "verified")
        for r in rows
    )
    return dict(
        metrics=metrics(rows),
        cases={
            str(c): dict(metrics=metrics([r for r in rows if r["case_id"] == c]))
            for c in CANONICAL_CASE_IDS
        },
        groups=[
            dict(zip(names, key), **_metric_summary(items))
            for key, items in sorted(groups.items(), key=lambda x: canonical_json(x[0]))
        ],
        verification_coverage=dict(
            planned=len(rows),
            numerator=resolved,
            denominator=len(rows),
            rate=resolved / len(rows) if rows else None,
        ),
    )


def _validate_bindings(
    p: TopologyProtocol,
    c: TopologyCapture | None,
    a: TopologyAssessment | None,
) -> None:
    digest = topology_protocol_digest(p)
    if c is not None:
        if c.protocol_digest != digest:
            raise ValueError("capture protocol binding mismatch")
        items = {i.item_id for i in p.snapshot.sources + p.snapshot.interpretations}
        if not set(c.retrieved_item_ids) <= items:
            raise ValueError("capture references unknown items")
    if a is None:
        return
    if (
        a.protocol_digest != digest
        or a.capture_digest != payload_digest(c)
        or (a.evaluator != p.evaluator)
    ):
        raise ValueError("assessment protocol, capture or evaluator binding mismatch")
    if not {v.opportunity_id for v in a.verdicts} <= {o.opportunity_id for o in p.opportunities}:
        raise ValueError("assessment references foreign opportunities")
    if not {v.claim_id for v in a.claim_verdicts} <= {x.claim_id for x in p.snapshot.state.claims}:
        raise ValueError("assessment references foreign claims")


def analyze_evidence_topology(
    protocol: TopologyProtocol,
    capture: TopologyCapture | None,
    *,
    assessment: TopologyAssessment | None = None,
    authorize: Callable[[ArtifactAccessRequest], bool] | None = None,
    authenticate: Callable[[TopologyAssessment], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Authenticate observed outcomes independently of current permission to retrieve content."""
    if enabled is not True:
        raise ValueError("evidence topology analysis requires enabled=True")
    p = _record(protocol, TopologyProtocol)
    c = _record(capture, TopologyCapture) if capture is not None else None
    a = _record(assessment, TopologyAssessment) if assessment is not None else None
    _validate_bindings(p, c, a)
    if authenticate is not None and not callable(authenticate):
        raise ValueError("authenticate must be callable")
    accepted = (
        a is not None
        and a.status == "verified"
        and not a.human_required
        and (authenticated(a, authenticate))
    )
    verdicts = {v.opportunity_id: v for v in a.verdicts} if accepted and a is not None else {}
    rows: list[dict[str, Any]] = []
    for op in p.opportunities:
        v = verdicts.get(op.opportunity_id)
        eligible, value = (v.eligible, v.value) if v is not None else (None, None)
        status = (
            "ineligible"
            if eligible is False
            else ("verified" if eligible is True and value is not None else "unresolved")
        )
        if c is None or c.status == "censored":
            status, value = ("missing" if c is None else "censored"), None
        rows.append(
            dict(
                opportunity_id=op.opportunity_id,
                metric=op.metric,
                severity=op.severity.value,
                cohort=op.cohort,
                case_id=p.subject.case.case_id,
                stripe=p.subject.robustness.stripe_id,
                subtype=p.subject.robustness.subtype,
                status=status,
                eligible=eligible,
                value=value,
            )
        )
    mode = "artifact_index" if c is None or c.condition == "existing_retrieval" else c.condition
    retrieval = retrieve_artifact_evidence(
        p.snapshot, p.topology, p.query, mode=mode, authorize=authorize, enabled=True
    )
    current = tuple(
        r["item_id"] for r in retrieval.item_rows if r["channel"] == "candidate_evidence"
    )
    structure = assess_support_routes(
        p.snapshot, p.topology, available_item_ids=current, enabled=True
    )
    claims = {v.claim_id: v for v in a.claim_verdicts} if accepted and a is not None else {}
    support = {}
    for claim in p.snapshot.state.claims:
        claim_verdict = claims.get(claim.claim_id)
        status, reason = "unresolved", "independent support judgment unavailable"
        if claim_verdict is not None:
            if claim_verdict.supported is False or claim_verdict.contradictions_resolved is False:
                status, reason = "unsupported", "independent support or resolution rejected"
            elif claim_verdict.supported is True and claim_verdict.contradictions_resolved is True:
                status, reason = "blocked", "no currently available support route"
                if claim.claim_id in structure["available_conclusion_ids"]:
                    status, reason = "supported", "independent judgment and current route available"
        support[claim.claim_id] = dict(status=status, reason=reason)
    access = {tuple(r["artifact_key"]): r["decision"] for r in retrieval.access_rows}
    sources = {s.item_id: s for s in p.snapshot.sources}
    violations = [
        r["item_id"]
        for r in retrieval.item_rows
        if c is not None
        and r["item_id"] in c.retrieved_item_ids
        and any(access[sources[s].artifact_key] != "allowed" for s in r["ancestor_source_ids"])
    ]
    return dict(
        schema_version="evidence-topology-analysis-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        protocol_digest=topology_protocol_digest(p),
        capture_digest=payload_digest(c),
        assessment_authenticated=accepted,
        rows=rows,
        **summarize_topology_rows(rows),
        source_claims=p.snapshot.state.to_dict(),
        supported_claims=support,
        structure=structure,
        retrieval=retrieval.to_dict(),
        producer_view=retrieval.producer_view,
        observed_access_violations=sorted(violations),
        capture=None if c is None else c.to_dict(),
    )
