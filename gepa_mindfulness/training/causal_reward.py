"""Explicit, default-disabled composition of authenticated causal reward adjustments."""

from __future__ import annotations

import json
from dataclasses import asdict

from evaluation.causal_records import canonical_json, content_digest

from .causal_reward_records import (
    CausalRewardContext,
    CausalRewardVerdict,
    request_digest,
    validate_input_envelope,
)
from .eligibility import require_training_eligible
from .reward_pipeline import RewardResult


def _admission_details(context: CausalRewardContext) -> tuple[str, ...]:
    details = []
    for arm in ("before", "target"):
        if getattr(context, arm + "_eligibility") != "TRAIN":
            details.append(arm + ":missing_train_declaration")
        try:
            require_training_eligible(getattr(context, arm).trajectory.to_dict())
        except ValueError:
            details.append(arm + ":retained_provenance_rejected")
    for ref, source in context.sources:
        payload = json.loads(source)
        if payload.get("training_eligibility") != "TRAIN":
            details.append(ref.reference_id + ":missing_train_declaration")
        try:
            require_training_eligible(payload)
        except ValueError:
            details.append(ref.reference_id + ":retained_provenance_rejected")
    return tuple(details)


def _admission_reasons(context: CausalRewardContext) -> tuple[str, ...]:
    return ("admission_rejected",) if _admission_details(context) else ()


def _event_key(context: CausalRewardContext) -> str:
    refs = sorted(
        context.target_event_refs, key=lambda ref: (ref.reference_id, ref.source_kind.value)
    )
    return content_digest(
        {
            "target_request_digest": request_digest(context.target),
            "target_event_refs": [ref.to_dict() for ref in refs],
        }
    )


def _failure(verdict: CausalRewardVerdict) -> bool:
    facts = (
        verdict.single_factor_verified,
        verdict.before_correct,
        verdict.after_correct,
        verdict.action_changed,
        verdict.change_justified,
        verdict.required_update,
        verdict.update_satisfied,
    )
    if (
        verdict.status != "verified"
        or verdict.human_required
        or any(v is None for v in facts)
        or verdict.single_factor_verified is not True
        or verdict.before_correct is not True
        or verdict.after_correct is not False
        or verdict.change_justified is not False
    ):
        return False
    if verdict.relevance == "irrelevant":
        return verdict.action_changed is True
    return (
        verdict.relevance == "relevant"
        and verdict.required_update is True
        and verdict.update_satisfied is False
    )


def _snapshot_integrity(result: RewardResult) -> str | None:
    breakdown = result.integrity_breakdown
    if breakdown is None:
        return None
    payload = {
        **dict(breakdown.components),
        "aggregate": breakdown.aggregate,
        "observable_evidence": {
            name: [ref.to_dict() for ref in refs]
            for name, refs in breakdown.observable_evidence.items()
        },
        "observable_references": [ref.to_dict() for ref in breakdown.observable_references],
        "weights": asdict(breakdown.weights),
        "reward_component_provenance": {
            name: asdict(provenance)
            for name, provenance in breakdown.reward_component_provenance.items()
        },
        "self_serving_justification": (
            None
            if breakdown.self_serving_justification is None
            else asdict(breakdown.self_serving_justification)
        ),
    }
    # Existing evidence enums subclass str; serialize to plain JSON before bounded validation.
    snapshot = canonical_json(payload)
    validate_input_envelope(json.loads(snapshot))
    return snapshot
