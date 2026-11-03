"""Default-off bounded decision/review cascade and existing-authority host dispatch."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.training.causal_reward_records import _text, validate_input_envelope
from gepa_mindfulness.verification import (
    AuthorityGrantRegistry,
    AuthorizationDecision,
    TrustedClock,
    consume_authorization,
)
from mindful_trace_gepa import ActionRecord
from synthetic_data.lesson_curation import authenticate_payload

from .decision_routing import (
    DecisionBackend,
    PublicDecisionContext,
    _probability,
    evaluate_decision_backend,
    prepare_decision_packet,
)
from .epistemic_routing import RoutingPolicy, RoutingRequest, prepare_routing
from .schemas import RecommendedAction as Action


@dataclass(frozen=True)
class DecisionWorkflowConfig:
    """Closed procedural enablement, never a runtime grant or learned moral boundary."""

    enabled: bool = False
    workflow_id: str = ""

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.workflow_id) is not str:
            raise ValueError("explicit boolean and workflow identity required")
        if self.enabled:
            _text(self.workflow_id, "workflow_id")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": "decision-workflow-config-v1",
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    @classmethod
    def from_dict(cls, value: object) -> DecisionWorkflowConfig:
        validate_input_envelope(value)
        if type(value) is not dict or set(value) != {"schema_version", "enabled", "workflow_id"}:
            raise ValueError("invalid decision workflow config")
        if value["schema_version"] != "decision-workflow-config-v1":
            raise ValueError("unsupported decision workflow schema")
        return cls(value["enabled"], value["workflow_id"])


@dataclass(frozen=True)
class DecisionWorkflowResult:
    """Actual handler invocation; an arbitrary handler return is not verified task success."""

    invoked: bool
    _report_json: str
    result: object = None

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._report_json)


def _backend_identity(backend: DecisionBackend) -> dict[str, str]:
    if type(backend) is not DecisionBackend:
        raise ValueError("configured exact decision backend required")
    checked = replace(backend)
    return {name: getattr(checked, name) for name in ("name", "version", "model_version", "kind")}


def run_decision_workflow(
    request: RoutingRequest,
    *,
    config: DecisionWorkflowConfig = DecisionWorkflowConfig(),
    public: PublicDecisionContext | None = None,
    backend: DecisionBackend | None = None,
    confidence_threshold: float | None = None,
    policy: RoutingPolicy = RoutingPolicy(),
    review_backend: DecisionBackend | None = None,
    review_confidence_threshold: float | None = None,
    authenticate: Callable[[dict], bool] | None = None,
    authorize: (
        Callable[[dict], tuple[ActionRecord, AuthorizationDecision, AuthorityGrantRegistry]] | None
    ) = None,
    handlers: Mapping[Action, Callable[[ActionRecord], object]] | None = None,
    clock: TrustedClock | None = None,
    action_author_id: str | None = None,
    action_executor_id: str | None = None,
) -> DecisionWorkflowResult | None:
    """Run at most two decision calls, then consume a fresh exact-scope owner-issued grant.

    Installed callbacks and handlers are trusted host code, not a Python sandbox.
    The host authenticates source relevance, identities, evidence and spending,
    enforces timeouts, executes the committed action and records PEO consequences.
    Generative review cannot replace independent verification or runtime authority.
    """
    if type(config) is not DecisionWorkflowConfig:
        raise ValueError("exact DecisionWorkflowConfig required")
    config = DecisionWorkflowConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    if type(policy) is not RoutingPolicy or not policy.enabled:
        raise ValueError("existing routing guard must be explicitly enabled")
    if not callable(authenticate) or not callable(authorize) or clock is None:
        raise ValueError("source authentication, runtime authorization and clock required")
    if not isinstance(handlers, Mapping) or any(
        type(action) is not Action or not callable(callback)
        for action, callback in handlers.items()
    ):
        raise ValueError("configured existing-action handlers required")
    handlers = dict(handlers)
    identity = _backend_identity(backend)
    if review_backend is not None:
        review_identity = _backend_identity(review_backend)
        if review_backend.kind != "generative" or review_backend.name == backend.name:
            raise ValueError("separately named generative review backend required")
    else:
        review_identity = None
        if review_confidence_threshold is not None:
            raise ValueError("review threshold requires a configured reviewer")
    # Freeze the guard and public packet before any installed callback is invoked.
    policy = replace(policy)
    features = prepare_routing(request, policy)
    packet = prepare_decision_packet(features, public)
    source = {
        "config": config.to_dict(),
        "packet": packet.to_dict(),
        "backend": identity,
        "review_backend": review_identity,
        "confidence_threshold": confidence_threshold,
        "review_confidence_threshold": review_confidence_threshold,
        "policy": {field.name: getattr(policy, field.name) for field in fields(policy)},
    }
    # Both retained decisions, public corrective feedback and final dispatch must fit.
    validate_input_envelope(
        {
            "source": source,
            "decisions": [source] * 7,
            "feedback": source,
            "reserved_metadata": "x" * 32768,
        }
    )
    _probability(confidence_threshold)
    if review_backend is not None:
        _probability(review_confidence_threshold)
    if not authenticate_payload(source, authenticate):
        raise ValueError("decision source authentication withheld")
    first = evaluate_decision_backend(
        features, packet.context, backend, confidence_threshold=confidence_threshold
    )
    decisions = [first.to_dict()]
    final = first
    review_skipped = None
    if review_backend is not None and (first.error is not None or first.confidence_escalation):
        feedback = canonical_json(
            {"original_artifact": packet.context.artifact, "decision_feedback": first.to_dict()}
        )
        if len(feedback) > 65536:
            review_skipped = "feedback_capacity"
        else:
            reviewed_public = replace(packet.context, artifact=feedback)
            final = evaluate_decision_backend(
                features,
                reviewed_public,
                review_backend,
                confidence_threshold=review_confidence_threshold,
            )
            decisions.append(final.to_dict())
    route = final.routing.action
    report = {
        "schema_version": "decision-workflow-report-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "optimizer_input": False,
        "source": source,
        "decisions": decisions,
        "review_skipped": review_skipped,
        "route": route.value,
        "status": "authorization_pending",
    }
    validate_input_envelope(report)
    if route not in handlers:
        report["status"] = "handler_unavailable"
        return DecisionWorkflowResult(False, canonical_json(report))
    # A changing latency capture is diagnostic; bind the exact full decision packet nonetheless.
    scope = "decision-workflow:" + content_digest(report)
    report["action_scope"] = scope
    validate_input_envelope(report)
    encoded = canonical_json(report)
    detached = json.loads(encoded)
    try:
        authorized = authorize(detached)
        validate_input_envelope(detached)
        unchanged = canonical_json(detached) == encoded
    except Exception:
        raise ValueError("runtime authorization callback failed or mutated its input") from None
    if not unchanged or type(authorized) is not tuple or len(authorized) != 3:
        raise ValueError("runtime authorization callback failed or mutated its input")
    action, decision, registry = authorized
    if (
        type(action) is not ActionRecord
        or type(decision) is not AuthorizationDecision
        or type(registry) is not AuthorityGrantRegistry
    ):
        raise ValueError("existing owner-issued action, authorization and registry required")
    action = ActionRecord(**action.to_dict())
    if action.authorization_scope != scope or action.action_class != route.value:
        raise ValueError("runtime action/decision route binding mismatch")
    report.update(
        status="handler_invoked", action=action.to_dict(), authorization=decision.to_dict()
    )
    validate_input_envelope(report)
    consume_authorization(
        decision,
        action,
        grant_registry=registry,
        clock=clock,
        action_author_id=action_author_id,
        action_executor_id=action_executor_id,
    )
    return DecisionWorkflowResult(True, canonical_json(report), handlers[route](action))
