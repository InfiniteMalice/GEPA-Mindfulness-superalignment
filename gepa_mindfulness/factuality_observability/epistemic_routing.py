"""Opt-in System-One proposals constrained by causal evidence and host policy.

Callbacks are trusted host code, but their outputs confer no execution authority.
The host authenticates evidence, declares relevance and enforces callback timeouts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from time import perf_counter
from typing import Any, cast

from gepa_mindfulness.verification.epistemic_reconciliation import EpistemicReconciliation
from gepa_mindfulness.verification.epistemic_state import Availability, MismatchStatus
from gepa_mindfulness.verification.interfaces import RelationalVerificationResult
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from mindful_trace_gepa.event_sequence import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope

from .routing import RoutingContext, RoutingDecision, choose_routing_action
from .schemas import RecommendedAction as Action


def _number(value: object, name: str, maximum: float) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a number")
    number = cast(float, value)
    if not math.isfinite(number) or not 0 <= number <= maximum:
        raise ValueError(f"{name} must be finite and between 0 and {maximum}")


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """Host thresholds for diagnostics; these are not calibrated probabilities."""

    enabled: bool = False
    uncertainty_threshold: float = 0.6
    high_risk_threshold: float = 0.8
    max_state_age_seconds: float = 300.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        _number(self.uncertainty_threshold, "uncertainty_threshold", 1)
        _number(self.high_risk_threshold, "high_risk_threshold", 1)
        _number(self.max_state_age_seconds, "max_state_age_seconds", 86400)


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutingRequest:
    """Complete prefix ending at a proposal; relevance is a trusted host declaration."""

    events: tuple[EventEnvelope, ...]
    decision_event_id: str
    context: RoutingContext
    context_changed: bool = False


@dataclass(frozen=True, slots=True)
class RoutingFeatures:
    """Bounded public features shared by all candidate backends; no private reasoning."""

    input_digest: str
    decision_event_id: str
    reconciliation_event_id: str | None
    world_uncertainty: float | None
    model_uncertainty: float | None
    monitor_uncertainty: float | None
    mismatch: bool
    verified_evidence: bool
    baseline_action: Action
    required_action: Action
    allowed_actions: tuple[Action, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Detach public features for a classifier, JEV/CLM adapter or audit."""
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True, slots=True)
class RoutingBackend:
    """Named/versioned host callback; call once, then validate its bounded proposal."""

    name: str
    version: str
    propose: Callable[[RoutingFeatures], Action]

    def __post_init__(self) -> None:
        for value in (self.name, self.version):
            if type(value) is not str or not value.strip() or len(value) > 128:
                raise ValueError(
                    "backend name and version must be nonblank, at most 128 characters"
                )
        if not callable(self.propose):
            raise ValueError("backend.propose must be callable")


LEGACY_BACKEND = RoutingBackend("legacy-router", "v1", lambda features: features.baseline_action)
UNCERTAINTY_BACKEND = RoutingBackend(
    "uncertainty-rules", "v1", lambda features: features.required_action
)


@dataclass(frozen=True, slots=True)
class RoutingAssessment:
    """Serializable routing audit; ACCEPT is only a continuation proposal."""

    features: RoutingFeatures
    backend_name: str
    backend_version: str
    proposed_action: Action | None
    action: Action
    backend_error: str | None
    latency_ms: float

    @property
    def decision_event_id(self) -> str:
        return self.features.decision_event_id

    @property
    def reconciliation_event_id(self) -> str | None:
        return self.features.reconciliation_event_id

    @property
    def input_digest(self) -> str:
        return self.features.input_digest

    @property
    def overridden(self) -> bool:
        return self.proposed_action != self.action

    def to_dict(self) -> dict[str, Any]:
        """Return detached public diagnostics, with an explicit non-authority marker."""
        return json.loads(
            json.dumps(
                asdict(self)
                | {
                    "schema_version": "epistemic-routing-v1",
                    "overridden": self.overridden,
                    "confers_authority": False,
                }
            )
        )

    def routing_decision(self) -> RoutingDecision:
        """Adapt to the existing router output; targets come only from the fixed action set."""
        targets = {
            Action.ACCEPT: "final_output",
            Action.DOWNGRADE_CONFIDENCE: "trace_queue",
            Action.DECOMPOSE_AND_VERIFY: "atomic_fact_pipeline",
            Action.RETRIEVE_MORE: "retrieval_stack",
            Action.ROUTE_EXTERNAL: "specialized_verifier",
            Action.ABSTAIN: "abstention_handler",
            Action.CLARIFY: "clarification_handler",
            Action.ESCALATE: "human_review",
        }
        return RoutingDecision(
            self.action, ["epistemic-routing-v1", self.features.reason], targets[self.action]
        )


def _context(context: RoutingContext) -> RoutingContext:
    if type(context) is not RoutingContext:
        raise ValueError("context must be RoutingContext")
    snapshot = replace(context)
    for name in ("operational_confidence", "claim_complexity", "domain_risk", "guessing_pressure"):
        _number(getattr(snapshot, name), name, 1)
    for name in (
        "has_provenance",
        "trace_worthy",
        "abstention_viable",
        "verification_required",
        "representation_sensitive",
    ):
        if type(getattr(snapshot, name)) is not bool:
            raise ValueError(f"{name} must be boolean")
    if type(snapshot.base_case_label) is not int or not 1 <= snapshot.base_case_label <= 17:
        raise ValueError("base_case_label must be a canonical integer case 1..17")
    if type(snapshot.verification_budget) is not int or snapshot.verification_budget < 0:
        raise ValueError("verification_budget must be a nonnegative integer")
    return snapshot


def _same_unit(left: EventEnvelope, right: EventEnvelope) -> bool:
    return all(
        getattr(left, key) == getattr(right, key)
        for key in (
            "run_id",
            "repeat_id",
            "conversation_id",
            "model_version",
            "harness_version",
            "case_version",
            "case_id",
            "stripe_id",
            "stripe_subtype",
            "seed",
        )
    )


def _verified(record: EpistemicReconciliation, by_id: dict[str, EventEnvelope]) -> bool:
    """Require typed, nonconflicting findings whose field evidence covers the state."""
    required = {
        "task_fit": True,
        "dependencies_satisfied": True,
        "contradiction_status": "none",
        "provenance_intact": True,
        "authorization_scope_valid": True,
        "claimed_outcome_supported": True,
        "repeated_failed_route": False,
    }
    evidence = record.update.posterior_state.evidence_refs
    if not evidence or any(not ref.is_observable for ref in evidence):
        return False
    for event_id in record.verification_event_ids:
        event = by_id[event_id]
        if event.payload.get("verification_level") != "relational_evidence":
            return False
        result = RelationalVerificationResult.from_dict(event.payload["result"])
        if any(getattr(result, name) != value for name, value in required.items()):
            return False
        for name in required.keys() - {"repeated_failed_route"}:
            bound = {
                ref
                for b in result.evidence_bindings
                if b.field_name == name
                for ref in b.evidence_refs
            }
            if not set(evidence) <= bound:
                return False
    return True


def prepare_routing(request: RoutingRequest, policy: RoutingPolicy) -> RoutingFeatures:
    """Validate the prefix and compute gates before exposing detached features to a backend."""
    policy = replace(policy)
    if not policy.enabled:
        raise ValueError("epistemic routing requires enabled=True")
    context = _context(request.context)
    if type(request.context_changed) is not bool:
        raise ValueError("context_changed must be boolean")
    events = request.events
    if type(events) is not tuple or not 1 <= len(events) <= 10000:
        raise ValueError("events must be a tuple of 1..10000 envelopes")
    validate_action_bound_sequence(events)
    decision = events[-1]
    if decision.event_id != request.decision_event_id or decision.event_type != "action_proposed":
        raise ValueError("events must end at decision_event_id, an action_proposed event")
    if not decision.conversation_id or type(decision.checkpoint_step) is not int:
        raise ValueError("decision requires conversation_id and checkpoint_step")
    decision_time = parse_rfc3339_datetime(decision.timestamp, "decision.timestamp")
    times = [parse_rfc3339_datetime(e.timestamp, "event.timestamp") for e in events]
    if times != sorted(times):
        raise ValueError("routing prefix must be chronological")
    for event in events[:-1]:
        if parse_rfc3339_datetime(event.timestamp, "event.timestamp") > decision_time:
            raise ValueError("routing prefix cannot contain future events")
        if _same_unit(event, decision) and (
            type(event.checkpoint_step) is not int
            or event.checkpoint_step > decision.checkpoint_step
        ):
            raise ValueError("routing prefix requires earlier checkpoint steps")
    reconciliations = [
        e
        for e in events[:-1]
        if e.event_type == "epistemic_reconciliation" and _same_unit(e, decision)
    ]
    selected = reconciliations[-1] if reconciliations else None
    world = model = monitor = None
    mismatch, verified = True, False
    state_reason = "missing_state"
    if selected is not None:
        record = EpistemicReconciliation.from_dict(selected.payload)
        estimate = record.update.posterior_state
        estimate.context.validate_event(decision)
        by_id = {e.event_id: e for e in events}
        pending, ancestors = list(selected.parent_event_ids), set()
        while pending:
            event_id = pending.pop()
            if event_id not in ancestors:
                ancestors.add(event_id)
                pending.extend(by_id[event_id].parent_event_ids)
        foreign_evidence = any(not _same_unit(by_id[key], decision) for key in ancestors)
        expired = any(
            by_id[key].valid_until is not None
            and decision_time >= parse_rfc3339_datetime(by_id[key].valid_until, "valid_until")
            for key in (*ancestors, selected.event_id)
        )
        not_yet_valid = any(
            by_id[key].valid_from is not None
            and decision_time < parse_rfc3339_datetime(by_id[key].valid_from, "valid_from")
            for key in (*ancestors, selected.event_id)
        )
        age = decision_time - parse_rfc3339_datetime(selected.timestamp, "state.timestamp")
        observations = [
            e
            for e in events
            if e.event_type == "outcome_observed" and e.action_id == estimate.action_id
        ]
        # Legacy verifier envelopes may omit action_id. Both verifier schemas have
        # a validated direct observation parent; use that ancestry for completeness.
        selected_index = events.index(selected)
        changed_evidence = observations[-1].event_id != record.observation_event_id or any(
            e.event_type == "verification_result"
            and by_id[e.parent_event_ids[0]].action_id == estimate.action_id
            and e.event_id not in record.verification_event_ids
            and (e.parent_event_ids == (record.observation_event_id,) or i > selected_index)
            for i, e in enumerate(events)
        )
        if foreign_evidence:
            state_reason = "foreign_evidence"
        elif expired or not_yet_valid:
            state_reason = "evidence_outside_validity_window"
        elif selected.superseded_by or changed_evidence:
            state_reason = "unreconciled_evidence"
        elif age.total_seconds() > policy.max_state_age_seconds:
            state_reason = "stale_state"
        elif estimate.status != Availability.AVAILABLE:
            state_reason = "unavailable_state"
        else:
            state_reason = "current_state"
            world, model, monitor = (
                estimate.world_uncertainty,
                estimate.model_uncertainty,
                estimate.monitor_uncertainty,
            )
            mismatch = record.update.model_mismatch != MismatchStatus.NONE or any(
                b.innovation.mismatch_status != MismatchStatus.NONE for b in record.bindings
            )
            verified = _verified(record, {e.event_id: e for e in events})
    baseline = choose_routing_action(context).recommended_action
    if context.verification_budget == 0:
        required, reason = baseline, "budget_exhausted"
    elif context.representation_sensitive or context.verification_required:
        required, reason = Action.ROUTE_EXTERNAL, "mandatory_verification"
    elif context.domain_risk >= policy.high_risk_threshold or not decision.payload["reversible"]:
        required, reason = Action.ROUTE_EXTERNAL, "high_impact_review"
    elif request.context_changed or state_reason != "current_state":
        required, reason = Action.ROUTE_EXTERNAL, (
            "context_changed" if request.context_changed else state_reason
        )
    elif monitor is None or monitor >= policy.uncertainty_threshold:
        required, reason = Action.ROUTE_EXTERNAL, "independent_monitor_verification"
    elif model is None or model >= policy.uncertainty_threshold:
        required, reason = Action.ROUTE_EXTERNAL, "model_verification_reduce_autonomy"
    elif mismatch:
        required, reason = Action.DECOMPOSE_AND_VERIFY, "reconsider_hypothesis"
    elif world is None or world >= policy.uncertainty_threshold:
        required, reason = Action.RETRIEVE_MORE, "acquire_evidence"
    elif not verified:
        required, reason = Action.ROUTE_EXTERNAL, "insufficient_verified_evidence"
    else:
        required, reason = baseline, "legacy_route_with_verified_state"
    allowed: list[Action] = (
        list(Action) if required == Action.ACCEPT else [required, Action.ESCALATE]
    )
    if context.abstention_viable:
        allowed.append(Action.ABSTAIN)
    else:
        allowed = [action for action in allowed if action != Action.ABSTAIN]
    digest = sha256(
        json.dumps(
            {
                "events": [e.to_dict() for e in events],
                "context": asdict(context),
                "policy": asdict(policy),
                "context_changed": request.context_changed,
                "schema_version": "epistemic-routing-v1",
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return RoutingFeatures(
        digest,
        decision.event_id,
        selected.event_id if selected else None,
        world,
        model,
        monitor,
        mismatch,
        verified,
        baseline,
        required,
        tuple(dict.fromkeys(allowed)),
        reason,
    )


def route_epistemic(
    request: RoutingRequest,
    *,
    policy: RoutingPolicy = RoutingPolicy(),
    backend: RoutingBackend = UNCERTAINTY_BACKEND,
) -> RoutingAssessment | None:
    """Propose one route when enabled; never execute it or change host authority.

    Disabled routing returns None without invoking the callback or reading the event prefix.
    Invalid inputs raise ValueError. Callback failures/malformed outputs recommend escalation.
    Latency measures only the callback; a hard timeout belongs in the host's backend adapter.
    """
    policy = replace(policy)
    if not policy.enabled:
        return None
    features = prepare_routing(request, policy)
    return evaluate_routing_backend(features, backend)


def evaluate_routing_backend(
    features: RoutingFeatures,
    backend: RoutingBackend,
) -> RoutingAssessment:
    """Benchmark one callback against host-prepared features; not an authorization API."""
    features = replace(features)
    backend = replace(backend)
    started = perf_counter()
    error = None
    proposed = None
    try:
        candidate = backend.propose(replace(features))
        if type(candidate) is not Action:
            error = "invalid_proposal"
        else:
            proposed = candidate
    except Exception as exc:
        error = type(exc).__name__
    latency = (perf_counter() - started) * 1000
    action = (
        Action.ESCALATE
        if error
        else proposed if proposed in features.allowed_actions else features.required_action
    )
    assert action is not None
    return RoutingAssessment(
        features, backend.name, backend.version, proposed, action, error, latency
    )
