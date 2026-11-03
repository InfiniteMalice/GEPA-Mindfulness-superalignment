"""Opt-in repeat suppression subordinate to existing one-use runtime authority."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification import (
    AuthorityGrantRegistry,
    AuthorizationDecision,
    TrustedClock,
    consume_authorization,
)
from gepa_mindfulness.verification.interfaces import (
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)
from mindful_trace_gepa import ActionRecord, PredictionCommit
from mindful_trace_gepa.logging_schema import EventEnvelope

from .actionopd import DecisionKind
from .causal_reward import _accepted
from .causal_reward_records import _json_value, _text, validate_input_envelope
from .deloop import _EXEMPT, LoopStep, audit_loops

ContextReview = Callable[[dict[str, object]], bool]
RepeatVerifier = Callable[[dict[str, object]], RelationalVerificationResult | None]


@dataclass(frozen=True)
class LoopGuardConfig:
    """Host-selected procedural scope; never an authorization or moral policy."""

    enabled: bool = False
    policy_id: str = ""
    task_id: str = ""
    run_id: str = ""
    harness_version: str = ""
    reviewer_id: str = ""
    reviewer_version: str = ""
    reviewer_contract_id: str = ""

    def __post_init__(self) -> None:
        """Require explicit bounded identity fields only when the guard is enabled."""
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        for field in fields(self):
            if field.name == "enabled":
                continue
            value = getattr(self, field.name)
            if self.enabled:
                _text(value, field.name)
            elif type(value) is not str:
                raise ValueError("scope fields must be strings")

    @property
    def reviewer(self) -> TrustedEvaluatorContract:
        """Retain installed host reviewer provenance without certifying its identity."""
        return TrustedEvaluatorContract(
            self.reviewer_id, self.reviewer_version, self.reviewer_contract_id
        )

    def to_dict(self) -> dict[str, object]:
        """Export a closed versioned configuration without executable roles."""
        self.__post_init__()
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["schema_version"] = "live-loop-config-v1"
        validate_input_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> LoopGuardConfig:
        """Restore only exact v1 settings, never a runtime grant."""
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or value["schema_version"] != "live-loop-config-v1"
        ):
            raise ValueError("invalid live loop configuration")
        return cls(**{name: value[name] for name in names})


@dataclass(frozen=True)
class GuardedExecution:
    """Host executor invocation/result; neither task success nor independent verification."""

    invoked: bool
    gate: dict[str, object] | None
    result: object = None


def loop_guard_action_scope(
    pending: LoopStep,
    prediction: PredictionCommit,
    config: LoopGuardConfig,
) -> str:
    """Bind an existing owner-issued grant to exact pending inputs and guard scope.

    This is a scope digest, not a grant. The runtime owner must authenticate the
    intended action and enroll its existing policy/grant using this scope.
    """
    if (
        type(pending) is not LoopStep
        or type(prediction) is not PredictionCommit
        or type(config) is not LoopGuardConfig
    ):
        raise ValueError("exact pending step, prediction and guard config required")
    value = {
        "schema_version": "live-loop-action-scope-v1",
        "pending": pending.to_dict(),
        "prediction": prediction.to_dict(),
        "config": config.to_dict(),
    }
    validate_input_envelope(value)
    return "live-loop:" + content_digest(value)


def _finish(report: dict[str, object], route: str, reason: str) -> dict[str, object]:
    """Return detached bounded diagnostic data; callers cannot restore permission from it."""
    report.update(route=route, reason=reason)
    validate_input_envelope(report)
    return _json_value(canonical_json(report))


def _reserve_review(report: dict[str, object], refs: tuple[EvidenceReference, ...]) -> None:
    """Reserve a maximum-size allowed relational result before either reviewer runs."""
    names = (
        "task_fit",
        "dependencies_satisfied",
        "contradiction_status",
        "provenance_intact",
        "authorization_scope_valid",
        "claimed_outcome_supported",
        "repeated_failed_route",
    )
    sizing = RelationalVerificationResult(
        report["repeat_action_id"],
        True,
        True,
        "contradicted",
        True,
        True,
        True,
        True,
        refs,
        tuple(VerificationEvidenceBinding(name, refs) for name in names),
    )
    validate_input_envelope(
        report
        | {
            "route": "blocked_repeat",
            "reason": "x" * 256,
            "diagnostic_subtype": "x" * 128,
            "repeat_verification": sizing.to_dict(),
        }
    )


def inspect_repeat_guard(
    events: tuple[EventEnvelope, ...],
    steps: tuple[LoopStep, ...],
    pending: LoopStep | None,
    prediction: PredictionCommit | None,
    action: ActionRecord,
    *,
    config: LoopGuardConfig = LoopGuardConfig(),
    authenticate_context: ContextReview | None = None,
    verify_repeat: RepeatVerifier | None = None,
) -> dict[str, object] | None:
    """Inspect a pending decision against completed PEO observations, never authorize it.

    Host context review authenticates history, state version, prediction and any
    claimed exception. Structural matches alone cannot block a repeat. Installed
    roles and source stores remain trusted; this is not a Python sandbox.
    """
    if type(config) is not LoopGuardConfig:
        raise ValueError("expected exact LoopGuardConfig")
    config = LoopGuardConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    if (
        type(events) is not tuple
        or type(steps) is not tuple
        or type(pending) is not LoopStep
        or type(prediction) is not PredictionCommit
        or type(action) is not ActionRecord
    ):
        raise ValueError("exact completed history, pending step, prediction and action required")
    current = LoopStep.from_dict(pending.to_dict())
    committed = PredictionCommit(**prediction.to_dict())
    proposed = ActionRecord(**action.to_dict())
    predicted = committed.to_dict()["predicted_outcome"]
    if not current.context_refs:
        raise ValueError("current observable state/context references required")
    if (
        proposed.action_id != current.request.trajectory.trajectory_id
        or proposed.action_class != current.action.kind.value
        or proposed.prediction_commit_id != committed.prediction_commit_id
        or type(predicted) is not dict
        or canonical_json(predicted.get("deloop")) != canonical_json(current.commitment())
        or not {ref.reference_id for ref in current.context_refs} <= set(committed.evidence_refs)
    ):
        raise ValueError("pending action/prediction/context binding mismatch")
    if proposed.authorization_scope != loop_guard_action_scope(current, committed, config):
        raise ValueError("runtime authorization scope does not bind the exact pending action")
    if config.reviewer_id in {
        current.request.trajectory.model_identifier,
        current.request.trajectory.policy_version,
    }:
        raise ValueError("student cannot review its own repeated action")
    audit = None
    source = ()
    if events or steps:
        audit = audit_loops(events, steps, task_id=config.task_id, enabled=True)
        if (
            audit["run_id"] != config.run_id
            or audit["harness_version"] != config.harness_version
            or audit["model_version"] != current.request.trajectory.policy_version
        ):
            raise ValueError("current/source model, harness or run scope mismatch")
        source = tuple(LoopStep.from_dict(item) for item in audit["source_steps"])
        if proposed.action_id in {step.request.trajectory.trajectory_id for step in source}:
            raise ValueError("pending action identity was already executed")
        if any(
            event.event_type == "prediction_commit"
            and event.payload["prediction_commit_id"] == committed.prediction_commit_id
            for event in events
        ):
            raise ValueError("pending prediction cannot rewrite a committed source prediction")
    refs_by_id = {}
    for step in (*source, current):
        for ref in step.request.observable_references:
            if refs_by_id.setdefault(ref.reference_id, ref) != ref:
                raise ValueError("conflicting source evidence identity")
    refs = tuple(refs_by_id.values())
    candidate_digest = content_digest(current.action.to_dict())
    matching = (
        []
        if audit is None
        else [
            index
            for index, row in enumerate(audit["entries"])
            if row["action_digest"] == candidate_digest
            and source[index].state_version == current.state_version
        ]
    )
    repeated_index = matching[-1] if matching else None
    row = None if repeated_index is None else audit["entries"][repeated_index]
    # An intervening informative/novel action invalidates a stale loop prediction.
    fresh = repeated_index is not None and any(
        not later["raw_repeat"] for later in audit["entries"][repeated_index + 1 :]
    )
    suspect = (
        row is not None
        and row["classification"] == "unresolved_repeat"
        and not fresh
        and current.repeat_context not in _EXEMPT
    )
    raw_events = _json_value(canonical_json([event.to_dict() for event in events]))
    report = {
        "schema_version": "live-loop-guard-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "source_reference": "REF-DELOOP",
        "overlay_id": "repeated_action_audit",
        "config": config.to_dict(),
        "pending": current.to_dict(),
        "prediction": committed.to_dict(),
        "action": proposed.to_dict(),
        "source_events": raw_events,
        "source_audit": audit,
        "source_references": [ref.to_dict() for ref in refs],
        "context_authenticated": False,
        "repeat_action_id": None if row is None else row["action_id"],
        "required_observation_refs": [],
        "repeat_verification": None,
        "diagnostic_subtype": None,
        "route": "escalate",
        "reason": "context_unverified",
    }
    if suspect:
        previous = audit["entries"][row["previous_index"]]
        report["required_observation_refs"] = sorted(
            set(previous["observation_evidence_refs"]) | set(row["observation_evidence_refs"])
        )
        if current.action.kind is DecisionKind.CLARIFICATION:
            report["diagnostic_subtype"] = (
                "repeated_clarification_after_answer"
                if current.repeat_context == "answered_clarification"
                else "repeated_clarification_unchanged_state"
            )
        elif current.action.kind is DecisionKind.VERIFICATION:
            report["diagnostic_subtype"] = "verification_restart_without_new_observation"
        _reserve_review(report, refs)
    # Reserve final route/reason growth even when no relational review is needed.
    validate_input_envelope(
        report
        | {
            "route": "blocked_repeat",
            "reason": "x" * 256,
            "diagnostic_subtype": "x" * 128,
        }
    )
    if not _accepted(authenticate_context, report):
        return _finish(report, "escalate", "context_unverified")
    report["context_authenticated"] = True
    if not suspect:
        reason = "authenticated_recheck" if current.repeat_context in _EXEMPT else "no_current_loop"
        return _finish(report, "continue", reason)
    if not row["observation_verified"] or not previous["observation_verified"]:
        return _finish(report, "escalate", "repeat_observation_unverified")
    encoded = canonical_json(report)
    payload = _json_value(encoded)
    try:
        supplied = verify_repeat(payload) if callable(verify_repeat) else None
        validate_input_envelope(payload)
        if canonical_json(payload) != encoded or type(supplied) is not RelationalVerificationResult:
            return _finish(report, "escalate", "repeat_review_unavailable")
        checked = RelationalVerificationResult.from_dict(supplied.to_dict())
    except Exception:
        return _finish(report, "escalate", "repeat_review_unavailable")
    if (
        checked.action_id != row["action_id"]
        or len(set(checked.evidence_refs)) != len(checked.evidence_refs)
        or not set(checked.evidence_refs) <= set(refs)
        or any(
            len(set(binding.evidence_refs)) != len(binding.evidence_refs)
            or not set(binding.evidence_refs) <= set(refs)
            for binding in checked.evidence_bindings
        )
    ):
        return _finish(report, "escalate", "repeat_review_scope_mismatch")
    report["repeat_verification"] = checked.to_dict()
    bound = {
        ref.reference_id
        for binding in checked.evidence_bindings
        if binding.field_name == "repeated_failed_route"
        for ref in binding.evidence_refs
    }
    if not set(report["required_observation_refs"]) <= bound:
        return _finish(report, "escalate", "repeat_review_omits_observations")
    supported = (
        checked.provenance_intact
        and checked.authorization_scope_valid
        and checked.contradiction_status == "none"
    )
    if supported and checked.repeated_failed_route:
        return _finish(report, "blocked_repeat", "verified_unproductive_repeat")
    if (
        supported
        and not checked.repeated_failed_route
        and checked.task_fit
        and checked.dependencies_satisfied
        and checked.claimed_outcome_supported
    ):
        return _finish(report, "continue", "independently_supported_repeat")
    return _finish(report, "escalate", "repeat_review_unresolved")


def execute_guarded_action(
    action: ActionRecord,
    authorization: AuthorizationDecision,
    *,
    grant_registry: AuthorityGrantRegistry,
    clock: TrustedClock,
    execute: Callable[[ActionRecord], object],
    config: LoopGuardConfig = LoopGuardConfig(),
    events: tuple[EventEnvelope, ...] = (),
    steps: tuple[LoopStep, ...] = (),
    pending: LoopStep | None = None,
    prediction: PredictionCommit | None = None,
    authenticate_context: ContextReview | None = None,
    verify_repeat: RepeatVerifier | None = None,
    action_author_id: str | None = None,
    action_executor_id: str | None = None,
) -> GuardedExecution:
    """Suppress reviewed loops, then delegate only after existing authority consumption.

    The synchronous host executor must execute exactly the authenticated committed
    action and log its actual PEO consequences. This function does not sandbox that
    host, certify task completion, or replace independent outcome verification.
    """
    if not callable(execute):
        raise ValueError("synchronous host executor required")
    if type(config) is not LoopGuardConfig:
        raise ValueError("expected exact LoopGuardConfig")
    if config.enabled and type(authorization) is not AuthorizationDecision:
        raise PermissionError("existing issued runtime authorization required")
    if config.enabled and config.reviewer_id in {
        authorization.principal_id,
        action_author_id,
        action_executor_id,
    }:
        raise ValueError("loop reviewer must be independent of action authors/executors")
    gate = inspect_repeat_guard(
        events,
        steps,
        pending,
        prediction,
        action,
        config=config,
        authenticate_context=authenticate_context,
        verify_repeat=verify_repeat,
    )
    if gate is not None and gate["route"] != "continue":
        return GuardedExecution(False, gate)
    consume_authorization(
        authorization,
        action,
        grant_registry=grant_registry,
        clock=clock,
        action_author_id=action_author_id,
        action_executor_id=action_executor_id,
    )
    return GuardedExecution(True, gate, execute(ActionRecord(**action.to_dict())))
