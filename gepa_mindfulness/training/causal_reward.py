"""Explicit, default-disabled composition of authenticated causal reward adjustments."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite
from typing import NoReturn

from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.reward_integrity import RewardObservation

from .causal_reward_records import (
    CausalRewardAccounting,
    CausalRewardContext,
    CausalRewardPolicy,
    CausalRewardUseRequest,
    CausalRewardVerdict,
    JsonObject,
    _bindings,
    _read_request,
    _request_wire,
    request_digest,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .eligibility import require_training_eligible
from .reward_pipeline import RewardPipeline, RewardResult


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


_REASON_ORDER = (
    "missing_context",
    "missing_verdict",
    "unresolved_verdict",
    "authentication_rejected",
    "admission_rejected",
    "use_not_authorized",
    "severe_violation",
    "severe_unknown",
    "accounting_unknown",
    "already_counted",
    "not_attributable",
    "target_correct",
    "adjustment_applied",
)


def _accepted(callback: Callable[[JsonObject], bool] | None, payload: JsonObject) -> bool:
    if callback is None:
        return False
    validate_input_envelope(payload)
    snapshot = canonical_json(payload)
    detached = json.loads(snapshot)
    try:
        result = callback(detached)
        validate_input_envelope(detached)
        return result is True and canonical_json(detached) == snapshot
    except Exception:
        # Host exceptions are not evidence and may contain credentials or private text.
        return False


@dataclass(frozen=True, init=False)
class CausalRewardDecision:
    """Immutable audit result; optimizer access is restricted to an admitted live call.

    Constructed only by CausalRewardAdapter.score. Exported reports are DEVELOPMENT
    evidence and cannot restore the private live authorization state.
    """

    base_result: RewardResult
    baseline_total: float
    deduction: float
    diagnostic_total: float
    disposition: str
    reasons: tuple[str, ...]
    severe_status: str
    audit: str
    _authorized: bool

    def __init__(self) -> None:
        raise TypeError("causal decisions are produced only by CausalRewardAdapter.score")

    def optimizer_reward(self) -> float:
        """Return this transaction's scalar or raise if any optimizer gate is unresolved."""
        if not self._authorized:
            raise ValueError("causal optimizer access withheld: " + ", ".join(self.reasons))
        return self.diagnostic_total

    def to_dict(self) -> JsonObject:
        """Export audit-only data without a restorable optimizer authorization."""
        return {
            "schema_version": "causal-reward-decision-v1",
            "training_eligibility": "DEVELOPMENT",
            "baseline_total": self.baseline_total,
            "deduction": self.deduction,
            "diagnostic_total": self.diagnostic_total,
            "disposition": self.disposition,
            "reasons": list(self.reasons),
            "severe_status": self.severe_status,
            "audit": json.loads(self.audit),
        }

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "live decisions cannot be restored; export to_dict and rescore original inputs"
        )


def _decision(
    result: RewardResult,
    *,
    baseline: float,
    deduction: float,
    authorized: bool,
    reasons: set[str],
    severe_status: str,
    audit: JsonObject,
) -> CausalRewardDecision:
    total = baseline if deduction == 0 else baseline - deduction
    if not isfinite(total):
        raise ValueError("causal adjusted result must be finite")
    decision = object.__new__(CausalRewardDecision)
    values = {
        "base_result": result,
        "baseline_total": baseline,
        "deduction": deduction,
        "diagnostic_total": total,
        "disposition": "applied" if deduction else ("unchanged" if authorized else "withheld"),
        "reasons": tuple(reason for reason in _REASON_ORDER if reason in reasons),
        "severe_status": severe_status,
        "audit": canonical_json(audit),
        "_authorized": authorized,
    }
    for name, value in values.items():
        object.__setattr__(decision, name, value)
    return decision


class CausalRewardAdapter:
    """Compose a bounded causal deduction after fresh semantic and use authentication.

    Args:
        pipeline: Existing reward pipeline, called once for the target request.
        policy: Explicit experiment configuration, disabled by default.
        authenticate_verdict: Host verifier of the complete context, policy and verdict.
        authorize_use: Host verifier of admission, configuration and semantic overlap.
    """

    def __init__(
        self,
        pipeline: RewardPipeline,
        *,
        policy: CausalRewardPolicy,
        authenticate_verdict: Callable[[JsonObject], bool] | None = None,
        authorize_use: Callable[[JsonObject], bool] | None = None,
    ) -> None:
        if type(policy) is not CausalRewardPolicy:
            raise ValueError("expected CausalRewardPolicy")
        self.pipeline = pipeline
        self._policy = CausalRewardPolicy.from_dict(policy.to_dict())
        self.authenticate_verdict = authenticate_verdict
        self.authorize_use = authorize_use

    @property
    def policy(self) -> CausalRewardPolicy:
        """Return the immutable policy selected when this adapter was constructed."""
        return self._policy

    def score(
        self,
        target_request: RewardRequest,
        *,
        observation: RewardObservation | None = None,
        context: CausalRewardContext | None = None,
        verdict: CausalRewardVerdict | None = None,
        accounting: CausalRewardAccounting | None = None,
    ) -> RewardResult | CausalRewardDecision:
        """Score one target; disabled calls return the exact existing pipeline result.

        Enabled calls snapshot input before provider or host execution. Missing evidence
        withholds optimizer use; malformed records and inconsistent bindings raise ValueError.
        Call optimizer_reward explicitly on an admitted decision; diagnostic totals alone
        never authorize training.
        """
        policy = self.policy
        if not policy.enabled or policy.weight == 0:
            return self.pipeline.score(target_request, observation=observation)
        policy = CausalRewardPolicy.from_dict(policy.to_dict())
        for record, kind in (
            (context, CausalRewardContext),
            (verdict, CausalRewardVerdict),
            (accounting, CausalRewardAccounting),
        ):
            if record is not None and type(record) is not kind:
                raise ValueError("expected exact causal input record")
        context = None if context is None else CausalRewardContext.from_dict(context.to_dict())
        verdict = None if verdict is None else CausalRewardVerdict.from_dict(verdict.to_dict())
        accounting = (
            None if accounting is None else CausalRewardAccounting.from_dict(accounting.to_dict())
        )
        target_wire = _request_wire(target_request)
        target_snapshot = canonical_json(target_wire)
        detached_target = _read_request(json.loads(target_snapshot))
        envelope: JsonObject = {
            "target_request": target_wire,
            "policy": policy.to_dict(),
            "context": None if context is None else context.to_dict(),
            "verdict": None if verdict is None else verdict.to_dict(),
            "accounting": None if accounting is None else accounting.to_dict(),
        }
        validate_input_envelope(envelope)
        if context is not None:
            if request_digest(detached_target) != request_digest(context.target):
                raise ValueError("target request binding mismatch")
            _bindings(context, policy, verdict, accounting)
        overlay_enabled = self.pipeline.overlay_enabled
        result = self.pipeline.score(detached_target, observation=observation)
        if canonical_json(_request_wire(detached_target)) != target_snapshot:
            raise ValueError("reward provider mutated the detached target request")
        baseline = result.total
        if type(baseline) not in (int, float) or not isfinite(baseline):
            raise ValueError("baseline total must be finite")
        integrity = _snapshot_integrity(result)
        if context is not None:
            _bindings(context, policy, verdict, accounting, baseline)
        reasons: set[str] = set()
        if context is None:
            reasons.add("missing_context")
        if verdict is None:
            reasons.add("missing_verdict")
        resolved = (
            verdict is not None and verdict.status == "verified" and not verdict.human_required
        )
        if verdict is not None and not resolved:
            reasons.add("unresolved_verdict")
        authenticated = False
        if context is not None and verdict is not None:
            authenticated = _accepted(
                self.authenticate_verdict,
                {
                    "context": context.to_dict(),
                    "policy": policy.to_dict(),
                    "verdict": verdict.to_dict(),
                },
            )
        if not authenticated:
            reasons.add("authentication_rejected")
        admission_details = () if context is None else _admission_details(context)
        admitted = context is not None and not admission_details
        if not admitted:
            reasons.add("admission_rejected")
        use_authorized, event_key = False, None
        if context is not None:
            event_key = _event_key(context)
            if verdict is not None and accounting is not None:
                use_request = CausalRewardUseRequest(
                    context=context,
                    verdict=verdict,
                    accounting=accounting,
                    policy=policy,
                    baseline_total=baseline,
                    integrity_snapshot=integrity,
                    event_key=event_key,
                )
                use_authorized = _accepted(self.authorize_use, use_request.to_dict())
        if not use_authorized:
            reasons.add("use_not_authorized")
        severe_status = "unknown"
        if authenticated and resolved and verdict is not None:
            if verdict.severe_violation is True:
                severe_status = "verified_violation"
            elif verdict.severe_violation is False:
                severe_status = "verified_clear"
        if severe_status == "unknown":
            reasons.add("severe_unknown")
        elif severe_status == "verified_violation":
            reasons.add("severe_violation")
        required_layers = {"provider"}
        if overlay_enabled or result.integrity_breakdown is not None:
            required_layers.add("reward_integrity")
        layers = {} if accounting is None else dict(accounting.layers)
        accounting_known = required_layers.issubset(layers) and "unknown" not in layers.values()
        if not accounting_known:
            reasons.add("accounting_unknown")
        counted = "already_counted" in layers.values()
        if counted:
            reasons.add("already_counted")
        failure = verdict is not None and _failure(verdict)
        if verdict is not None:
            if verdict.after_correct is True:
                reasons.add("target_correct")
            elif not failure:
                reasons.add("not_attributable")
        authorized = (
            authenticated
            and resolved
            and admitted
            and use_authorized
            and severe_status == "verified_clear"
            and accounting_known
        )
        deduction = (
            min(policy.weight, policy.max_adjustment)
            if authorized and failure and not counted
            else 0.0
        )
        if deduction:
            reasons.add("adjustment_applied")
        audit = {
            **envelope,
            "baseline_total": baseline,
            "integrity_snapshot": None if integrity is None else json.loads(integrity),
            "event_key": event_key,
            "admission_details": list(admission_details),
            "gates": {
                "authenticated": authenticated,
                "resolved": resolved,
                "admitted": admitted,
                "use_authorized": use_authorized,
                "accounting_known": accounting_known,
                "already_counted": counted,
                "failure": failure,
                "severe_status": severe_status,
            },
        }
        return _decision(
            result,
            baseline=baseline,
            deduction=deduction,
            authorized=authorized,
            reasons=reasons,
            severe_status=severe_status,
            audit=audit,
        )
