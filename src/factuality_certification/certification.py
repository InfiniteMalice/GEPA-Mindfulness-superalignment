"""Main certification pipeline."""

from __future__ import annotations

from .abstention_policy import detect_abstention, detect_refusal
from .claim_extraction import extract_atomic_claims
from .config import FactualityCertificationConfig
from .constraints import constraint_scores
from .evidence_matching import match_claims_to_evidence
from .integrations import to_13_case_features
from .logging_schema import make_hash
from .overrefusal_guard import find_scoped_alternative
from .types import CertificationResult, EvidenceItem


def _overall_from_action(action: str, fallback: str) -> str:
    if action == "refuse":
        return "should_refuse"
    if action == "abstain":
        return "should_abstain"
    if action in {"answer_with_qualifications", "answer_partially", "ask_clarifying_question"}:
        return "partial"
    return fallback


def certify_answer(
    prompt: str,
    answer: str,
    evidence: list[EvidenceItem] | None = None,
    context: str | None = None,
    trace_summary: str | None = None,
    config: FactualityCertificationConfig | None = None,
    safety_context: dict[str, bool] | None = None,
) -> CertificationResult:
    """Certify answer factuality against evidence/context with optional safety policy context."""
    cfg = config or FactualityCertificationConfig()
    if not cfg.enabled or cfg.mode == "off":
        return CertificationResult(
            mode=cfg.mode,
            overall_label="certified",
            hallucination_risk=0.0,
            overrefusal_risk=0.0,
            useful_answer_retention_score=1.0,
            claim_support=[],
            recommended_action="answer",
        )
    ev = list(evidence or [])
    if context and context.strip():
        existing_ids = {item.id for item in ev}
        context_id = "context"
        suffix = 1
        while context_id in existing_ids:
            context_id = f"context_{suffix}"
            suffix += 1
        ev.append(
            EvidenceItem(
                id=context_id,
                text=context,
                source="context",
                quality_score=0.8,
                retrieval_score=1.0,
            )
        )

    claims = extract_atomic_claims(answer, cfg)
    supports = match_claims_to_evidence(claims, ev, cfg)
    contradicted = sum(1 for s in supports if s.support_label == "contradicted")
    unsupported = sum(1 for s in supports if s.support_label == "unsupported")
    partial_supported = sum(1 for s in supports if s.support_label == "partially_supported")
    total = max(1, len(supports))
    hallucination_risk = min(
        1.0, (contradicted + unsupported * 0.8 + partial_supported * 0.4) / total
    )

    scoped = find_scoped_alternative(prompt, answer, claims, supports, safety_context)
    abstained = detect_abstention(answer)
    refused = detect_refusal(answer)
    overrefusal = (
        (refused or abstained) and scoped.scoped_answer_possible and not scoped.refusal_required
    )

    action = "answer"
    overall = "certified"
    warnings: list[str] = []
    if contradicted > 0:
        overall = "uncertified"
        action = "answer_with_qualifications"
        warnings.append("Contradicted claim(s) detected.")
    elif unsupported > 0 or partial_supported > 0:
        overall = "partial" if (unsupported + partial_supported) < total else "uncertified"
        action = (
            "answer_partially"
            if (unsupported + partial_supported) < total
            else "answer_with_qualifications"
        )
    if abstained:
        overall = "should_abstain"
        action = "abstain"
    if refused:
        overall = "should_refuse"
        action = "refuse"
    if (
        cfg.mode in {"advisory", "shadow"}
        and (abstained or refused or action in {"refuse", "abstain"})
        and scoped.scoped_answer_possible
    ):
        action = scoped.action
        overall = _overall_from_action(action, fallback="partial")
    if cfg.mode == "gated" and scoped.refusal_required:
        action = "refuse"
        overall = "should_refuse"

    ctr = constraint_scores(
        supports, is_refusal=(action == "refuse"), is_abstention=(action == "abstain")
    )

    evidence_by_id = {item.id: item for item in ev}
    claim_to_current_evidence_ids: dict[str, list[str]] = {}
    for support in supports:
        current_ids = [
            eid
            for eid in support.evidence_ids
            if evidence_by_id.get(eid) is not None and bool(evidence_by_id[eid].timestamp)
        ]
        claim_to_current_evidence_ids[support.claim_id] = current_ids

    logs = {
        "prompt_hash": make_hash(prompt),
        "answer_hash": make_hash(answer),
        "atomic_claim_ids": [c.id for c in claims],
        "current_source_claim_ids": [c.id for c in claims if c.requires_current_source],
        "support_labels": {s.claim_id: s.support_label for s in supports},
        "evidence_ids": {s.claim_id: s.evidence_ids for s in supports},
        "contradiction_scores": {s.claim_id: s.contradiction_score for s in supports},
        "recommended_action": action,
        "counterfactual_action": scoped.action if cfg.mode == "shadow" else None,
        "overrefusal_detected": overrefusal,
        "scoped_answer_possible": scoped.scoped_answer_possible,
        "refusal_required": scoped.refusal_required,
        "thought_trace_aligned": True if trace_summary is None else bool(trace_summary.strip()),
        "has_references": any(bool(e.citation or e.source) for e in ev),
        "has_current_references": any(bool(e.timestamp) for e in ev),
        "current_claim_to_evidence_map": claim_to_current_evidence_ids,
    }
    res = CertificationResult(
        mode=cfg.mode,
        overall_label=overall,
        hallucination_risk=hallucination_risk,
        overrefusal_risk=1.0 if overrefusal else 0.0,
        useful_answer_retention_score=1.0 if action.startswith("answer") else 0.5,
        claim_support=supports,
        recommended_action=action,
        warnings=warnings,
        metrics=ctr,
        logs=logs,
    )
    res.logs["schema_13_case"] = to_13_case_features(res)
    return res


def positive_only_reward_features(result: CertificationResult) -> dict[str, float]:
    """Return positive-only thought/certification reward features."""
    return {
        "identified_uncertainty": (
            1.0 if result.recommended_action in {"abstain", "answer_with_qualifications"} else 0.0
        ),
        "narrowed_answer_scope": (
            1.0
            if result.recommended_action in {"answer_partially", "answer_with_qualifications"}
            else 0.0
        ),
        "used_supporting_evidence": (
            1.0
            if any(
                s.evidence_ids and s.support_label in {"supported", "partially_supported"}
                for s in result.claim_support
            )
            else 0.0
        ),
        "corrected_unsupported_claim": (
            1.0 if result.overall_label in {"partial", "should_abstain"} else 0.0
        ),
        "chose_calibrated_abstention": 1.0 if result.recommended_action == "abstain" else 0.0,
        "avoided_overrefusal": 1.0 if result.overrefusal_risk == 0.0 else 0.0,
        "separated_supported_from_unsupported_claims": (
            1.0
            if (
                any(
                    s.support_label in {"supported", "partially_supported"}
                    for s in result.claim_support
                )
                and any(
                    s.support_label in {"unsupported", "contradicted"} for s in result.claim_support
                )
            )
            else 0.0
        ),
        "cited_current_evidence_when_needed": (
            1.0
            if (
                result.logs.get("has_current_references", False)
                and result.logs.get("current_source_claim_ids")
                and all(
                    bool(result.logs.get("current_claim_to_evidence_map", {}).get(claim_id, []))
                    for claim_id in result.logs.get("current_source_claim_ids", [])
                )
            )
            else 0.0
        ),
    }
