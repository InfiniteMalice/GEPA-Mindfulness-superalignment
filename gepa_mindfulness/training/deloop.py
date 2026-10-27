"""Default-off PEO repetition audits and independently reviewed DeLoop proposals."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification.interfaces import (
    LocalVerificationResult,
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)
from gepa_mindfulness.verification.state import parse_rfc3339_datetime
from mindful_trace_gepa import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope

from .actionopd import DecisionKind, SemanticAction, public_state_digest
from .causal_reward_records import (
    _digest,
    _json_value,
    _number,
    _read_request,
    _request_wire,
    _text,
    _wire,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .research_credit import _evaluator

_EXEMPT = {"human_recheck", "independent_verification", "productive_repeat"}
_CONTEXTS = {"ordinary", "answered_clarification", *_EXEMPT}
_LIMIT = 4 * 1024 * 1024


@dataclass(frozen=True)
class LoopStep:
    """Host-captured decision metadata; identifiers and evidence do not authenticate themselves."""

    request: RewardRequest
    action: SemanticAction
    state_version: str
    repeat_context: str = "ordinary"
    context_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        """Detach existing request/action contracts and require observable context references."""
        if type(self.request) is not RewardRequest or type(self.action) is not SemanticAction:
            raise ValueError("expected exact RewardRequest and SemanticAction")
        _text(self.state_version, "state_version")
        if type(self.repeat_context) is not str or self.repeat_context not in _CONTEXTS:
            raise ValueError("invalid repeat context")
        if type(self.context_refs) is not tuple or len(self.context_refs) > 64:
            raise ValueError("bounded context evidence tuple required")
        if self.repeat_context != "ordinary" and not self.context_refs:
            raise ValueError("nonordinary repeat context requires observable evidence")
        refs = []
        for ref in self.context_refs:
            if type(ref) is not EvidenceReference or not ref.is_observable:
                raise ValueError("context evidence must be independently observable")
            refs.append(EvidenceReference.from_dict(ref.to_dict()))
        if len({ref.reference_id for ref in refs}) != len(refs):
            raise ValueError("duplicate context evidence")
        request = _read_request(_request_wire(self.request))
        action = SemanticAction.from_dict(self.action.to_dict())
        if not set(refs) <= set(request.observable_references):
            raise ValueError("context references must belong to the source request")
        if SemanticAction.from_dict(_json_value(request.trajectory.response)) != action:
            raise ValueError("source response must encode the captured public semantic action")
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "context_refs", tuple(refs))

    def commitment(self) -> dict[str, object]:
        """Bind public state, action and context before execution in predicted_outcome.deloop."""
        return {
            "schema_version": "deloop-decision-v1",
            "public_state_digest": public_state_digest(self.request),
            "action_digest": content_digest(self.action.to_dict()),
            "state_version": self.state_version,
            "repeat_context": self.repeat_context,
            "context_refs": [ref.to_dict() for ref in self.context_refs],
        }

    def to_dict(self) -> dict[str, object]:
        """Export a detached, versioned observation record with no admission authority."""
        return {
            "schema_version": "deloop-step-v1",
            "training_eligibility": "DEVELOPMENT",
            "authority_granted": False,
            "source_request": _request_wire(self.request),
            "action": self.action.to_dict(),
            "state_version": self.state_version,
            "repeat_context": self.repeat_context,
            "context_refs": [ref.to_dict() for ref in self.context_refs],
        }

    @classmethod
    def from_dict(cls, value: object) -> LoopStep:
        """Reject unknown fields/versions and revalidate all nested source contracts."""
        data = _wire(
            value,
            "deloop-step-v1",
            "training_eligibility authority_granted source_request action "
            "state_version repeat_context context_refs",
        )
        if data["training_eligibility"] != "DEVELOPMENT" or data["authority_granted"] is not False:
            raise ValueError("step serialization cannot grant training or runtime authority")
        if type(data["context_refs"]) is not list:
            raise ValueError("context_refs must be an array")
        return cls(
            _read_request(data["source_request"]),
            SemanticAction.from_dict(data["action"]),
            data["state_version"],
            data["repeat_context"],
            tuple(EvidenceReference.from_dict(ref) for ref in data["context_refs"]),
        )


@dataclass(frozen=True)
class LoopCandidate:
    """Immutable proposal sent to a host-owned reviewer, never to a model scorer."""

    payload_json: str

    def to_dict(self) -> dict[str, object]:
        """Return a fresh proposal view so reviewer mutations cannot alter its digest."""
        return _json_value(self.payload_json)

    @property
    def digest(self) -> str:
        """Bind the review to this exact original state, trace and candidate actions."""
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class LoopPairReview:
    """Existing verifier findings plus an explicit original-state applicability observation."""

    candidate_digest: str
    evaluator: TrustedEvaluatorContract
    applies_to_original_state: bool | None
    loop: RelationalVerificationResult
    local_recovery: LocalVerificationResult
    recovery: RelationalVerificationResult

    def __post_init__(self) -> None:
        """Validate exact result types before any serialization or subject lookup."""
        _digest(self.candidate_digest)
        object.__setattr__(self, "evaluator", _evaluator(self.evaluator))
        if (
            self.applies_to_original_state is not None
            and type(self.applies_to_original_state) is not bool
        ):
            raise ValueError("original-state applicability must be boolean or unresolved")
        for value, expected in (
            (self.loop, RelationalVerificationResult),
            (self.local_recovery, LocalVerificationResult),
            (self.recovery, RelationalVerificationResult),
        ):
            if type(value) is not expected:
                raise ValueError("expected existing exact verification result types")

    def to_dict(self) -> dict[str, object]:
        """Serialize independent findings as DEVELOPMENT data, not a training permission."""
        return {
            "schema_version": "deloop-review-v1",
            "training_eligibility": "DEVELOPMENT",
            "authority_granted": False,
            "candidate_digest": self.candidate_digest,
            "evaluator": vars(_evaluator(self.evaluator)),
            "applies_to_original_state": self.applies_to_original_state,
            "loop": self.loop.to_dict(),
            "local_recovery": self.local_recovery.to_dict(),
            "recovery": self.recovery.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> LoopPairReview:
        """Revalidate the versioned review without authenticating the evaluator or evidence."""
        data = _wire(
            value,
            "deloop-review-v1",
            "training_eligibility authority_granted candidate_digest evaluator "
            "applies_to_original_state loop local_recovery recovery",
        )
        if data["training_eligibility"] != "DEVELOPMENT" or data["authority_granted"] is not False:
            raise ValueError("review serialization cannot grant training or runtime authority")
        digest = data["candidate_digest"]
        if type(digest) is not str or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError("invalid review candidate digest")
        applicable = data["applies_to_original_state"]
        if applicable is not None and type(applicable) is not bool:
            raise ValueError("original-state applicability must be boolean or unresolved")
        identity = data["evaluator"]
        if type(identity) is not dict or set(identity) != {
            "evaluator_id",
            "evaluator_version",
            "contract_id",
        }:
            raise ValueError("invalid reviewer identity")
        return cls(
            digest,
            _evaluator(TrustedEvaluatorContract(**identity)),
            applicable,
            RelationalVerificationResult.from_dict(data["loop"]),
            LocalVerificationResult.from_dict(data["local_recovery"]),
            RelationalVerificationResult.from_dict(data["recovery"]),
        )


def _source_verified(event: EventEnvelope) -> bool:
    """Read existing legacy or typed findings without inventing an aggregate moral score."""
    payload = event.to_dict()["payload"]
    if "verified" in payload:
        return payload["verified"] is True
    if payload.get("verification_level") == "local_execution":
        value = LocalVerificationResult.from_dict(payload["result"])
        return all(
            (
                value.executed,
                value.arguments_valid,
                value.schema_valid,
                value.authorization_valid,
                value.intended_operation_observed,
            )
        )
    if payload.get("verification_level") == "relational_evidence":
        value = RelationalVerificationResult.from_dict(payload["result"])
        return (
            value.provenance_intact
            and value.authorization_scope_valid
            and value.claimed_outcome_supported
            and value.contradiction_status == "none"
        )
    return False


def _capture(
    events: object, steps: object, task_id: str
) -> tuple[dict[str, object], tuple[LoopStep, ...]]:
    """Bound all source input before review and validate complete causal execution coverage."""
    _text(task_id, "task_id")
    if type(events) is not tuple or not 1 <= len(events) <= 4096:
        raise ValueError("expected 1..4096 source events")
    if type(steps) is not tuple or not 1 <= len(steps) <= 512:
        raise ValueError("expected 1..512 captured decisions")
    total = 0
    captured_events, captured_steps = [], []
    for records, expected, destination in (
        (events, EventEnvelope, captured_events),
        (steps, LoopStep, captured_steps),
    ):
        for record in records:
            if type(record) is not expected:
                raise ValueError("invalid source record type")
            wire = record.to_dict()
            if expected is EventEnvelope:
                for name in ("parent_event_ids", "evidence_refs", "verifier_refs"):
                    wire[name] = list(wire[name])
            validate_input_envelope(wire)
            total += len(canonical_json(wire).encode("utf-8")) + 2
            if total > _LIMIT:
                raise ValueError("DeLoop input exceeds 4 MiB")
            destination.append(_json_value(canonical_json(wire)))
    event_values = tuple(EventEnvelope(**value) for value in captured_events)
    source_steps = tuple(LoopStep.from_dict(value) for value in captured_steps)
    evidence_by_id: dict[str, EvidenceReference] = {}
    for step in source_steps:
        for ref in step.request.trajectory.evidence_references:
            existing = evidence_by_id.setdefault(ref.reference_id, ref)
            if existing != ref:
                raise ValueError("conflicting evidence source kinds for one reference identity")
    validate_action_bound_sequence(event_values)

    def scope(event: EventEnvelope) -> tuple[object, ...]:
        """Keep source episodes separate even within an evaluation run."""
        return (
            event.run_id,
            event.repeat_id,
            event.model_version,
            event.harness_version,
            event.conversation_id,
            event.trace_id,
            event.rollout_id,
        )

    if any(scope(event) != scope(event_values[0]) for event in event_values):
        raise ValueError("source events must share one run/repeat/model/harness/conversation")
    dates = [parse_rfc3339_datetime(event.timestamp, "timestamp") for event in event_values]
    if dates != sorted(dates):
        raise ValueError("source events must be chronological")
    executed = [event for event in event_values if event.event_type == "action_executed"]
    action_ids = [step.request.trajectory.trajectory_id for step in source_steps]
    if [event.action_id for event in executed] != action_ids or len(set(action_ids)) != len(
        action_ids
    ):
        raise ValueError("captured steps must cover every executed action exactly once in order")
    predictions = {
        event.payload["prediction_commit_id"]: event
        for event in event_values
        if event.event_type == "prediction_commit"
    }
    observed = [event for event in event_values if event.event_type == "outcome_observed"]
    rows, signatures, past_calls = [], {}, set()
    for step, execution in zip(source_steps, executed):
        action_id = step.request.trajectory.trajectory_id
        if step.request.trajectory.policy_version != execution.model_version:
            raise ValueError("source model version mismatch")
        if execution.payload["action_class"] != step.action.kind.value:
            raise ValueError("source action class does not match committed semantic decision")
        if (
            execution.model_id is not None
            and step.request.trajectory.model_identifier != execution.model_id
        ):
            raise ValueError("source model identity mismatch")
        prediction = predictions[execution.payload["prediction_commit_id"]]
        expected = prediction.to_dict()["payload"]["predicted_outcome"]
        if not isinstance(expected, dict) and not hasattr(expected, "get"):
            raise ValueError("pre-action DeLoop commitment missing")
        if canonical_json(expected.get("deloop")) != canonical_json(step.commitment()):
            raise ValueError("pre-action state/action/context commitment mismatch")
        if not {ref.reference_id for ref in step.context_refs} <= set(prediction.evidence_refs):
            raise ValueError("context evidence is not bound before execution")
        outcomes = [event for event in observed if event.action_id == action_id]
        if len(outcomes) != 1:
            raise ValueError("one unambiguous outcome required for each action")
        outcome = outcomes[0]
        known_refs = {
            ref.reference_id for ref in step.request.observable_references if ref.is_observable
        }
        if not set(outcome.evidence_refs) <= known_refs:
            raise ValueError("observation evidence does not belong to source request")
        checks = [
            event
            for event in event_values
            if event.event_type == "verification_result"
            and outcome.event_id in event.parent_event_ids
        ]
        verified = bool(checks) and all(_source_verified(event) for event in checks)
        action_digest = content_digest(step.action.to_dict())
        observation_digest = content_digest(outcome.to_dict()["payload"]["actual_outcome"])
        key = (action_digest, observation_digest, step.state_version)
        prior = signatures.get(key)
        repeated = prior is not None
        classification = "novel_action"
        if repeated:
            classification = (
                "exempt_repeat" if step.repeat_context in _EXEMPT else "unresolved_repeat"
            )
        elif action_digest in past_calls:
            classification = "informative_or_state_changed_repeat"
        subtype = None
        if repeated and step.action.kind is DecisionKind.CLARIFICATION:
            subtype = (
                "repeated_clarification_after_answer"
                if step.repeat_context == "answered_clarification"
                else "repeated_clarification_unchanged_state"
            )
        elif repeated and step.action.kind is DecisionKind.VERIFICATION:
            subtype = "verification_restart_without_new_observation"
        rows.append(
            {
                "action_id": action_id,
                "raw_repeat": repeated,
                "classification": classification,
                "previous_action_id": action_ids[prior] if repeated else None,
                "previous_index": prior,
                "observation_verified": verified,
                "observation_digest": observation_digest,
                "action_digest": action_digest,
                "observation_evidence_refs": list(outcome.evidence_refs),
                "diagnostic_subtype": subtype,
                "reversible": execution.payload["reversible"],
            }
        )
        signatures.setdefault(key, len(rows) - 1)
        past_calls.add(action_digest)
    report = {
        "schema_version": "deloop-audit-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "source_reference": "REF-DELOOP",
        "task_id": task_id,
        "model_version": executed[0].model_version,
        "harness_version": executed[0].harness_version,
        "run_id": executed[0].run_id,
        "repeat_id": executed[0].repeat_id,
        "source_events_digest": content_digest(captured_events),
        "source_steps": captured_steps,
        "entries": rows,
    }
    validate_input_envelope(report)
    return report, source_steps


def audit_loops(
    events: object, steps: object, *, task_id: str, enabled: bool = False
) -> dict[str, object] | None:
    """Audit completed PEO traces; structural repetition is not verified unproductive work."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return None
    return _capture(events, steps, task_id)[0]


def _review_reservation(
    candidate: LoopCandidate,
    identity: TrustedEvaluatorContract,
    refs: tuple[EvidenceReference, ...],
) -> dict[str, object]:
    """Size only: longest valid findings, every unique source ref in every allowed binding.

    Existing verifier contracts reject duplicate references/binding names. Bound subject IDs and
    identity are known before review; false and contradicted are the longest allowed values.
    This synthetic shape is never emitted as evidence or passed to the reviewer.
    """
    record = candidate.to_dict()
    local_names = (
        "executed arguments_valid schema_valid authorization_valid "
        "intended_operation_observed irreversible_action_permitted"
    ).split()
    relational_names = (
        "task_fit dependencies_satisfied contradiction_status provenance_intact "
        "authorization_scope_valid claimed_outcome_supported repeated_failed_route"
    ).split()

    def bindings(names: list[str]) -> tuple[VerificationEvidenceBinding, ...]:
        return tuple(VerificationEvidenceBinding(name, refs) for name in names)

    def relational(action_id: str) -> RelationalVerificationResult:
        return RelationalVerificationResult(
            action_id,
            False,
            False,
            "contradicted",
            False,
            False,
            False,
            False,
            refs,
            bindings(relational_names),
        )

    local = LocalVerificationResult(
        record["chosen_action_id"],
        False,
        False,
        False,
        False,
        False,
        False,
        refs,
        bindings(local_names),
    )
    return LoopPairReview(
        candidate.digest,
        identity,
        False,
        relational(record["rejected_action_id"]),
        local,
        relational(record["chosen_action_id"]),
    ).to_dict()


def mine_loop_pairs(
    events: object,
    steps: object,
    *,
    task_id: str,
    reviewer: TrustedEvaluatorContract,
    review: Callable[[LoopCandidate], LoopPairReview],
    enabled: bool = False,
) -> dict[str, object] | None:
    """Prepare reviewed original-state preferences; never admit data or authorize actions."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return None
    identity = _evaluator(reviewer)
    if not callable(review):
        raise ValueError("a host-owned independent review callback is required")
    audit, source = _capture(events, steps, task_id)
    producer_ids = {audit["model_version"]} | {
        step.request.trajectory.model_identifier for step in source
    }
    if identity.evaluator_id in producer_ids:
        raise ValueError("generating model cannot be its own independent reviewer")
    result = {
        "schema_version": "deloop-pairs-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "source_reference": "REF-DELOOP",
        "audit_digest": content_digest(audit),
        "pairs": [],
        "rejected_proposals": [],
    }
    rows = audit["entries"]
    for index, row in enumerate(rows):
        if row["classification"] != "unresolved_repeat":
            continue
        prior = rows[row["previous_index"]]
        if not row["observation_verified"] or not prior["observation_verified"]:
            result["rejected_proposals"].append(
                {"action_id": row["action_id"], "reason": "unverified repeat observation"}
            )
            continue
        recovery_index = next(
            (
                later
                for later in range(index + 1, len(rows))
                if not rows[later]["raw_repeat"]
                and rows[later]["action_digest"] != row["action_digest"]
            ),
            None,
        )
        if recovery_index is None:
            result["rejected_proposals"].append(
                {"action_id": row["action_id"], "reason": "no structural recovery"}
            )
            continue
        rejected, chosen = source[index], source[recovery_index]
        refs = {
            ref.reference_id: ref
            for step in (source[row["previous_index"]], rejected, chosen)
            for ref in step.request.observable_references
            if ref.is_observable
        }
        candidate = LoopCandidate(
            canonical_json(
                {
                    "schema_version": "deloop-candidate-v1",
                    "training_eligibility": "DEVELOPMENT",
                    "task_id": task_id,
                    "audit_digest": result["audit_digest"],
                    "prompt": rejected.request.trajectory.prompt,
                    "prompt_token_ids": (
                        list(rejected.request.trajectory.prompt_token_ids)
                        if rejected.request.trajectory.prompt_token_ids is not None
                        else None
                    ),
                    "rejected_action_id": row["action_id"],
                    "chosen_action_id": rows[recovery_index]["action_id"],
                    "rejected": rejected.request.trajectory.response,
                    "chosen": chosen.request.trajectory.response,
                    "original_state_digest": public_state_digest(rejected.request),
                    "source_rejected": rejected.to_dict(),
                    "source_chosen": chosen.to_dict(),
                    "evidence_refs": [ref.to_dict() for ref in refs.values()],
                }
            )
        )
        # Reserve the complete accepted entry before any potentially costly review. Rejected
        # entries are smaller, but validate that shape too so neither output route can overflow.
        reserved_review = _review_reservation(candidate, identity, tuple(refs.values()))
        entry = candidate.to_dict() | {
            "candidate_digest": candidate.digest,
            "review": reserved_review,
        }
        validate_input_envelope(result | {"pairs": [*result["pairs"], entry]})
        rejection = {
            "action_id": row["action_id"],
            "reason": "unresolved or unsupported original-state recovery",
            "review": reserved_review,
        }
        validate_input_envelope(
            result | {"rejected_proposals": [*result["rejected_proposals"], rejection]}
        )
        supplied = review(candidate)
        if type(supplied) is not LoopPairReview:
            raise ValueError("expected exact LoopPairReview")
        checked = LoopPairReview.from_dict(supplied.to_dict())
        if checked.evaluator != identity or checked.candidate_digest != candidate.digest:
            raise ValueError("independent review subject or identity mismatch")
        if checked.loop.action_id != row["action_id"] or any(
            item.action_id != rows[recovery_index]["action_id"]
            for item in (checked.local_recovery, checked.recovery)
        ):
            raise ValueError("review action binding mismatch")
        for finding in (checked.loop, checked.local_recovery, checked.recovery):
            if not set(finding.evidence_refs) <= set(refs.values()):
                raise ValueError("review refers to foreign evidence")
        for finding, field, required in (
            (
                checked.loop,
                "repeated_failed_route",
                set(prior["observation_evidence_refs"]) | set(row["observation_evidence_refs"]),
            ),
            (
                checked.local_recovery,
                "intended_operation_observed",
                set(rows[recovery_index]["observation_evidence_refs"]),
            ),
            (
                checked.recovery,
                "claimed_outcome_supported",
                set(rows[recovery_index]["observation_evidence_refs"]),
            ),
        ):
            bound = {
                ref.reference_id
                for binding in finding.evidence_bindings
                if binding.field_name == field
                for ref in binding.evidence_refs
            }
            if getattr(finding, field) is True and not required <= bound:
                raise ValueError("finding omits its source observation evidence binding")
        local, relational, loop = checked.local_recovery, checked.recovery, checked.loop
        accepted = (
            checked.applies_to_original_state is True
            and rows[recovery_index]["observation_verified"]
            and loop.repeated_failed_route
            and loop.provenance_intact
            and loop.authorization_scope_valid
            and loop.contradiction_status == "none"
            and local.executed
            and local.arguments_valid
            and local.schema_valid
            and local.authorization_valid
            and local.intended_operation_observed
            and (rows[recovery_index]["reversible"] or local.irreversible_action_permitted is True)
            and relational.task_fit
            and relational.dependencies_satisfied
            and relational.provenance_intact
            and relational.authorization_scope_valid
            and relational.claimed_outcome_supported
            and not relational.repeated_failed_route
            and relational.contradiction_status == "none"
        )
        if not accepted:
            result["rejected_proposals"].append(
                {
                    "action_id": row["action_id"],
                    "reason": "unresolved or unsupported original-state recovery",
                    "review": checked.to_dict(),
                }
            )
        else:
            result["pairs"].append(
                candidate.to_dict()
                | {"candidate_digest": candidate.digest, "review": checked.to_dict()}
            )
        validate_input_envelope(result)
    return result


def deloop_dpo_loss(
    chosen_log_probs: object,
    rejected_log_probs: object,
    reference_chosen: object,
    reference_rejected: object,
    *,
    beta: float,
) -> object:
    """Evaluate the paper's DPO objective on action sequence log-probability sums only."""
    _number(beta)
    if beta <= 0:
        raise ValueError("beta must be a positive finite coefficient")

    import torch

    values = (chosen_log_probs, rejected_log_probs, reference_chosen, reference_rejected)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError("DPO inputs must be tensors")
    first = chosen_log_probs
    if first.ndim != 1 or not 1 <= first.numel() <= 65536:
        raise ValueError("expected nonempty bounded one-dimensional log probabilities")
    if any(
        value.shape != first.shape
        or value.device != first.device
        or not value.is_floating_point()
        or not torch.isfinite(value).all().item()
        for value in values
    ):
        raise ValueError("DPO tensors must be aligned, floating and finite")
    dtype = (
        torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32
    )
    chosen, rejected = (value.to(dtype) for value in values[:2])
    ref_chosen, ref_rejected = (value.detach().to(dtype) for value in values[2:])
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    if not torch.isfinite(margin).all().item():
        raise ValueError("nonfinite DPO margin")
    loss = torch.nn.functional.softplus(-margin).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("nonfinite DPO loss")
    return loss
