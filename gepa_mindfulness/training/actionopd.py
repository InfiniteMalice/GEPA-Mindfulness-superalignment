"""Source-checked ActionOPD numerical helpers and default-off comparison adapter."""

# Standard library
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification.interfaces import RelationalVerificationResult

from .causal_reward_records import (
    _digest,
    _json_value,
    _number,
    _read_evaluator,
    _read_request,
    _request_wire,
    _text,
    validate_input_envelope,
)
from .contracts import RewardRequest


class DecisionKind(str, Enum):
    """Semantic decision types; this is not another canonical case taxonomy."""

    GROUNDED_ANSWER = "grounded_answer"
    QUALIFIED_ANSWER = "qualified_answer"
    IDK = "idk_abstention"
    CLARIFICATION = "necessary_clarification"
    REFUSAL = "refusal_or_redirection"
    VERIFICATION = "verification"
    TOOL_USE = "authorized_tool_use"
    EVIDENCE_UPDATE = "evidence_update_proposal"
    ESCALATION = "escalation"
    RESUMPTION = "resumption_after_clarification"


@dataclass(frozen=True)
class SemanticAction:
    """Host-normalized action content; type labels alone never establish equivalence."""

    kind: DecisionKind
    payload_json: str

    def __post_init__(self) -> None:
        """Canonicalize object keys without inventing paraphrase equivalence."""
        if type(self.kind) is not DecisionKind:
            raise ValueError("expected exact DecisionKind")
        value = _json_value(self.payload_json)
        if type(value) is not dict:
            raise ValueError("semantic action payload must be an object")
        object.__setattr__(self, "payload_json", canonical_json(value))

    def to_dict(self) -> dict[str, object]:
        """Export exact v1 action data with no authorization implication."""
        snapshot = SemanticAction(self.kind, self.payload_json)
        return {
            "schema_version": "semantic-action-v1",
            "kind": snapshot.kind.value,
            "payload": _json_value(snapshot.payload_json),
        }

    @classmethod
    def from_dict(cls, value: object) -> SemanticAction:
        """Restore bounded exact action fields, rejecting unknown schema versions."""
        validate_input_envelope(value)
        if type(value) is not dict or set(value) != {"schema_version", "kind", "payload"}:
            raise ValueError("invalid semantic action fields")
        if value["schema_version"] != "semantic-action-v1" or type(value["kind"]) is not str:
            raise ValueError("invalid action schema or kind")
        return cls(DecisionKind(value["kind"]), canonical_json(value["payload"]))


@dataclass(frozen=True)
class TeacherActionSample:
    """One black-box generation with its public state and sampling provenance."""

    sample_id: str
    teacher: TrustedEvaluatorContract
    state_digest: str
    sampling_config_json: str
    action: SemanticAction | None

    def __post_init__(self) -> None:
        """Validate generation metadata without treating the teacher as a verifier."""
        _text(self.sample_id, "sample_id")
        _digest(self.state_digest)
        if type(self.teacher) is not TrustedEvaluatorContract:
            raise ValueError("expected generation identity contract")
        for value in (
            self.teacher.evaluator_id,
            self.teacher.evaluator_version,
            self.teacher.contract_id,
        ):
            _text(value, "teacher identity")
        object.__setattr__(
            self,
            "teacher",
            TrustedEvaluatorContract(
                self.teacher.evaluator_id, self.teacher.evaluator_version, self.teacher.contract_id
            ),
        )
        config = _json_value(self.sampling_config_json)
        if type(config) is not dict:
            raise ValueError("sampling configuration must be an object")
        object.__setattr__(self, "sampling_config_json", canonical_json(config))
        if self.action is not None:
            if type(self.action) is not SemanticAction:
                raise ValueError("expected semantic action or invalid None")
            object.__setattr__(self, "action", SemanticAction.from_dict(self.action.to_dict()))

    def to_dict(self) -> dict[str, object]:
        """Retain one sample, including invalid outputs and generation configuration."""
        snapshot = TeacherActionSample(
            self.sample_id, self.teacher, self.state_digest, self.sampling_config_json, self.action
        )
        return {
            "schema_version": "teacher-action-sample-v1",
            "sample_id": snapshot.sample_id,
            "teacher": {
                "evaluator_id": snapshot.teacher.evaluator_id,
                "evaluator_version": snapshot.teacher.evaluator_version,
                "contract_id": snapshot.teacher.contract_id,
            },
            "state_digest": snapshot.state_digest,
            "sampling_config": _json_value(snapshot.sampling_config_json),
            "action": None if snapshot.action is None else snapshot.action.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> TeacherActionSample:
        """Restore exact sample metadata without restoring any reward authority."""
        validate_input_envelope(value)
        if type(value) is not dict or set(value) != {
            "schema_version",
            "sample_id",
            "teacher",
            "state_digest",
            "sampling_config",
            "action",
        }:
            raise ValueError("invalid teacher sample fields")
        if value["schema_version"] != "teacher-action-sample-v1":
            raise ValueError("unsupported teacher sample schema")
        return cls(
            value["sample_id"],
            _read_evaluator(value["teacher"]),
            value["state_digest"],
            canonical_json(value["sampling_config"]),
            None if value["action"] is None else SemanticAction.from_dict(value["action"]),
        )


def public_state_digest(request: RewardRequest) -> str:
    """Fingerprint retained public prompt and student version, excluding sampled response."""
    if type(request) is not RewardRequest:
        raise ValueError("expected exact RewardRequest")
    snapshot = _read_request(_request_wire(request))
    return _state_digest(snapshot)


def _state_digest(snapshot: RewardRequest) -> str:
    """Fingerprint an already validated request without restoring it a second time."""
    if not snapshot.trajectory.policy_version:
        raise ValueError("student policy version must be recorded")
    return content_digest(
        {
            "prompt": snapshot.trajectory.prompt,
            "prompt_token_ids": snapshot.trajectory.prompt_token_ids,
            "sampling_parameters": dict(snapshot.trajectory.sampling_parameters),
            "model_identifier": snapshot.trajectory.model_identifier,
            "policy_version": snapshot.trajectory.policy_version,
        }
    )


def agreement_reward(
    match_count: int, teacher_count: int, *, valid: bool, frequency_bonus: float
) -> float | None:
    """Evaluate paper Eq. 6, including its K=0/no-supervision and K=1 conventions."""
    if (
        type(match_count) is not int
        or type(teacher_count) is not int
        or not 0 <= match_count <= teacher_count
    ):
        raise ValueError("invalid teacher match counts")
    if type(valid) is not bool:
        raise ValueError("expected strict valid flag and finite nonnegative bonus")
    _number(frequency_bonus, nonnegative=True)
    if teacher_count == 0:
        return None
    if not valid:
        if match_count:
            raise ValueError("invalid action cannot match")
        return -1.0
    result = (
        0.0
        if match_count == 0
        else (
            1.0
            if teacher_count == 1
            else 1.0 + frequency_bonus * ((match_count - 1) / (teacher_count - 1))
        )
    )
    if not math.isfinite(result):
        raise ValueError("agreement reward overflow")
    return result


def actionopd_advantages(
    groups: tuple[tuple[float | None, ...], ...], *, epsilon: float
) -> tuple[tuple[float, ...] | None, ...]:
    """Center retained groups and whiten across the entire batch as in paper Eq. 7."""
    _number(epsilon, nonnegative=True)
    if epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if type(groups) is not tuple or not 1 <= len(groups) <= 1024:
        raise ValueError("expected 1..1024 groups")
    centered = []
    for group in groups:
        if type(group) is not tuple or not 1 <= len(group) <= 256:
            raise ValueError("expected 1..256 candidates per group")
        for value in group:
            if value is not None:
                _number(value)
        if any(value is None for value in group) or len(set(group)) == 1:
            centered.append(None)
            continue
        mean = math.fsum(value / len(group) for value in group)
        values = tuple(value - mean for value in group)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("centered reward overflow")
        centered.append(values)
    active = [value for group in centered if group is not None for value in group]
    if not active:
        return tuple(None for _ in groups)
    scale = math.hypot(*active) / math.sqrt(len(active)) + epsilon
    if not math.isfinite(scale):
        raise ValueError("advantage scale overflow")
    return tuple(
        None if group is None else tuple(value / scale for value in group) for group in centered
    )


def evaluate_action_group(
    requests: tuple[RewardRequest, ...],
    actions: tuple[SemanticAction | None, ...],
    teacher_samples: tuple[TeacherActionSample, ...],
    verifications: tuple[RelationalVerificationResult | None, ...],
    *,
    frequency_bonus: float = 0.5,
    nominal_teacher_count: int | None = None,
    enabled: bool = False,
) -> dict[str, object] | None:
    """Compare caller-collected same-state actions; report agreement separately from outcomes.

    This bounded matcher uses exact typed canonical content. A teacher sample is not
    an evaluator. Hosts authenticate source rollouts, parse actions and verify the
    semantic meaning of evidence independently. No action is selected or executed.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if type(requests) is not tuple or not 1 <= len(requests) <= 256:
        raise ValueError("expected 1..256 student requests")
    if (
        type(actions) is not tuple
        or type(verifications) is not tuple
        or not (len(requests) == len(actions) == len(verifications))
    ):
        raise ValueError("student/action/verification counts differ")
    if type(teacher_samples) is not tuple or len(teacher_samples) > 256:
        raise ValueError("expected at most 256 teacher samples")
    nominal = len(teacher_samples) if nominal_teacher_count is None else nominal_teacher_count
    if type(nominal) is not int or not len(teacher_samples) <= nominal <= 256:
        raise ValueError("nominal teacher count must cover retained samples")
    input_bytes = 0
    students = []
    for value in requests:
        if type(value) is not RewardRequest:
            raise ValueError("expected exact student request")
        wire = _request_wire(value)
        input_bytes = _input_budget(input_bytes, wire)
        students.append(_read_request(wire))
    state = _state_digest(students[0])
    if any(_state_digest(value) != state for value in students):
        raise ValueError("student state or policy mismatch")
    if len({value.trajectory.trajectory_id for value in students}) != len(students):
        raise ValueError("duplicate student sample identity")
    teachers = []
    for value in teacher_samples:
        if type(value) is not TeacherActionSample:
            raise ValueError("expected exact teacher sample")
        wire = value.to_dict()
        input_bytes = _input_budget(input_bytes, wire)
        teachers.append(TeacherActionSample.from_dict(wire))
    if any(value.state_digest != state for value in teachers):
        raise ValueError("teacher sampled at a different public state")
    if len({value.sample_id for value in teachers}) != len(teachers):
        raise ValueError("duplicate teacher sample identity")
    supervised = any(value.action is not None for value in teachers)
    candidates = []
    for request, value, verification in zip(students, actions, verifications):
        if value is not None and type(value) is not SemanticAction:
            raise ValueError("expected exact semantic action or invalid None")
        decision = None if value is None else SemanticAction.from_dict(value.to_dict())
        if decision is not None:
            input_bytes = _input_budget(input_bytes, decision.to_dict())
        matches = sum(
            decision is not None and other.action is not None and decision == other.action
            for other in teachers
        )
        if verification is not None:
            if type(verification) is not RelationalVerificationResult:
                raise ValueError("expected exact independent outcome record")
            verification = RelationalVerificationResult.from_dict(verification.to_dict())
            input_bytes = _input_budget(input_bytes, verification.to_dict())
            if verification.action_id != request.trajectory.trajectory_id:
                raise ValueError("verification candidate identity mismatch")
            if not set(verification.evidence_refs).issubset(request.observable_references):
                raise ValueError("verification evidence outside request")
        candidates.append(
            {
                "candidate_id": request.trajectory.trajectory_id,
                "request": _request_wire(request),
                "action": None if decision is None else decision.to_dict(),
                "match_count": matches,
                "agreement_reward": agreement_reward(
                    matches,
                    nominal if supervised else 0,
                    valid=decision is not None,
                    frequency_bonus=frequency_bonus,
                ),
                "verification": None if verification is None else verification.to_dict(),
            }
        )
    report = {
        "schema_version": "actionopd-comparison-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "state_digest": state,
        "matcher": "typed-canonical-json-v1",
        "frequency_bonus": frequency_bonus,
        "source_reference": "REF-ACTIONOPD",
        "nominal_teacher_count": nominal,
        "teacher_samples": [value.to_dict() for value in teachers],
        "candidates": candidates,
    }
    validate_input_envelope(report)
    return report


def _input_budget(current: int, value: object) -> int:
    """Stop snapshot ingestion before an aggregate can grow past its 4 MiB envelope."""
    validate_input_envelope(value)
    size = current + len(canonical_json(value).encode("utf-8"))
    if size > 4 * 1024 * 1024:
        raise ValueError("aggregate ActionOPD input exceeds 4 MiB")
    return size
