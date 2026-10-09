"""Independent pluralistic measurements with visible missingness and complete provenance."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

# Third-party
# Local
from gepa_mindfulness.verification.debate_records import _record
from semantic_intent_robustness.perspective_protocol import PerspectiveCapture, perspective_digest

from .cases.registry import CANONICAL_CASE_IDS
from .causal_diagnostics import _snapshot, _validate_roster
from .causal_records import CausalCapture, canonical_json, capture_digest, content_digest
from .debate_analysis import _summary
from .ladder import Severity
from .pluralistic_records import (
    PLURALISTIC_METRICS,
    PluralisticAssessment,
    PluralisticProtocol,
    pluralistic_protocol_digest,
)
from .sensitive_debate import authenticated


def summarize_pluralistic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw opportunity rows, retaining eligible unknowns in denominators."""

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            m: dict(
                _summary([r for r in items if r["metric"] == m], semantic=True),
                direction=(
                    "lower_is_better"
                    if m in ("social_sycophancy", "overcritical_response")
                    else "higher_is_better"
                ),
            )
            for m in PLURALISTIC_METRICS
        }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(
            row[k] for k in ("metric", "original_case", "stripe", "subtype", "severity", "cohort")
        )
        grouped.setdefault(key, []).append(row)
    return dict(
        metrics=metrics(rows),
        cases={
            str(c): dict(metrics=metrics([r for r in rows if r["original_case"] == c]))
            for c in CANONICAL_CASE_IDS
        },
        groups=[
            dict(
                zip(("metric", "original_case", "stripe", "subtype", "severity", "cohort"), key),
                **_summary(items, semantic=True),
            )
            for key, items in sorted(grouped.items(), key=lambda kv: canonical_json(kv[0]))
        ],
        severe_events=[r for r in rows if r["severity"] != Severity.ROUTINE.value],
        verification_coverage=_summary(
            rows,
            denominator=len(rows),
            numerator=sum(r["status"] in ("verified", "ineligible") for r in rows),
        ),
    )


def _candidate_coverage(
    protocol: PluralisticProtocol, capture: PerspectiveCapture | None
) -> dict[str, Any]:
    ids = {c.slot_id for c in capture.candidates} if capture is not None else set()
    planned = {s.slot_id for s in protocol.plan.slots}
    return dict(
        planned=len(planned),
        numerator=len(ids),
        denominator=len(planned),
        rate=len(ids) / len(planned) if planned else None,
        captured=sorted(ids),
        missing=sorted(planned - ids),
        attempt_status="missing" if capture is None else capture.status,
    )


def _validate_assessment(
    protocol: PluralisticProtocol,
    captures: tuple[CausalCapture, ...],
    pc: PerspectiveCapture | None,
    assessment: PluralisticAssessment | None,
) -> None:
    if pc is not None:
        if pc.plan_digest != perspective_digest(protocol.plan):
            raise ValueError("perspective capture plan binding mismatch")
        if not {c.slot_id for c in pc.candidates} <= {s.slot_id for s in protocol.plan.slots}:
            raise ValueError("perspective capture contains an unplanned slot")
    if assessment is None:
        return
    if (
        assessment.protocol_digest != pluralistic_protocol_digest(protocol)
        or assessment.capture_digest != capture_digest(captures)
        or assessment.perspective_capture_digest
        != content_digest(None if pc is None else pc.to_dict())
        or assessment.evaluator != protocol.evaluator
    ):
        raise ValueError("assessment protocol, capture or evaluator binding mismatch")
    if not {v.opportunity_id for v in assessment.verdicts} <= {
        o.opportunity_id for o in protocol.opportunities
    }:
        raise ValueError("assessment references a foreign opportunity")
    claims = protocol.plan.source.facts + protocol.plan.source.constraints
    if not {v.opportunity_id for v in assessment.claim_verdicts} <= {c.claim_id for c in claims}:
        raise ValueError("assessment references a foreign source claim")


def analyze_pluralistic(
    protocol: PluralisticProtocol,
    captures: tuple[CausalCapture, ...],
    *,
    perspective_capture: PerspectiveCapture | None = None,
    assessment: PluralisticAssessment | None = None,
    authenticate: Callable[[PluralisticAssessment], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any]:
    """Reauthenticate raw receipts; simulated preferences establish neither truth nor authority."""
    if enabled is not True:
        raise ValueError("pluralistic analysis requires enabled=True")
    protocol = _record(protocol, PluralisticProtocol)
    captures = _snapshot(captures, CausalCapture)
    pair = protocol.pair
    indexed = _validate_roster((pair,), captures, ())
    pc = (
        _record(perspective_capture, PerspectiveCapture)
        if perspective_capture is not None
        else None
    )
    a = _record(assessment, PluralisticAssessment) if assessment is not None else None
    _validate_assessment(protocol, captures, pc, a)
    accepted = (
        a is not None
        and a.status == "verified"
        and not a.human_required
        and authenticated(a, authenticate)
    )
    verdicts = {v.opportunity_id: v for v in a.verdicts} if a is not None else {}
    rows = []
    for op in sorted(protocol.opportunities, key=lambda o: o.opportunity_id):
        v = verdicts.get(op.opportunity_id)
        eligible, value = None, None
        status, reason = "unresolved", "independent semantic judgment unavailable"
        if accepted and v is not None:
            eligible, reason = v.eligible, v.reason
            if eligible is False:
                status = "ineligible"
            elif eligible is True and v.value is not None:
                status, value = "verified", v.value
        required = (
            (pair.before, pair.after) if op.metric == "perspective_robustness" else (pair.after,)
        )
        required_captures = [indexed.get((pair.digest, arm.variant_id)) for arm in required]
        if any(c is None for c in required_captures):
            status, value, reason = "missing", None, "required target capture unavailable"
        elif any(c is not None and c.status == "censored" for c in required_captures):
            status, value, reason = "censored", None, "required target capture censored"
        rows.append(
            dict(
                **op.to_dict(),
                original_case=pair.before.case.case_id,
                destination_case=pair.after.case.case_id,
                stripe=pair.after.robustness.stripe_id,
                subtype=pair.after.robustness.subtype,
                pair_id=pair.pair_id,
                status=status,
                eligible=eligible,
                value=value,
                reason=reason,
                evidence_refs=[r.to_dict() for r in v.evidence_refs] if v is not None else [],
            )
        )
    claim_verdicts = {v.opportunity_id: v for v in a.claim_verdicts} if a is not None else {}
    claims = {}
    for claim in protocol.plan.source.facts + protocol.plan.source.constraints:
        verdict = claim_verdicts.get(claim.claim_id)
        resolved = (
            accepted
            and verdict is not None
            and verdict.eligible is True
            and verdict.value is not None
        )
        claims[claim.claim_id] = dict(
            status=(
                ("supported" if verdict.value else "refuted")
                if resolved and verdict is not None
                else "unresolved"
            ),
            source_claim=claim.to_dict(),
            evidence_refs=(
                [r.to_dict() for r in verdict.evidence_refs] if verdict is not None else []
            ),
            reason=(
                verdict.reason if verdict is not None else "independent claim judgment unavailable"
            ),
        )
    report = dict(
        schema_version="pluralistic-report-v1",
        training_eligibility="DEVELOPMENT",
        confers_authority=False,
        optimizer_input=False,
        mechanism_recovery_established=False,
        protocol=protocol.to_dict(),
        captures=[c.to_dict() for c in captures],
        perspective_capture=None if pc is None else pc.to_dict(),
        assessment=None if a is None else a.to_dict(),
        assessment_accepted=accepted,
        protocol_digest=pluralistic_protocol_digest(protocol),
        capture_digest=capture_digest(captures),
        perspective_capture_digest=content_digest(None if pc is None else pc.to_dict()),
        assessment_digest=content_digest(None if a is None else a.to_dict()),
        source_restrictions=dict(
            pair=pair.training_eligibility.value,
            perspective_source=protocol.plan.source.source_training_eligibility.value,
        ),
        metric_rows=rows,
        **summarize_pluralistic_rows(rows),
        verified_claims=claims,
        candidate_coverage=_candidate_coverage(protocol, pc),
        sampling_note=(
            "Dependent diagnostic opportunities; no population or training effect established."
        ),
    )
    report["result_digest"] = content_digest(report)
    return json.loads(canonical_json(report))
