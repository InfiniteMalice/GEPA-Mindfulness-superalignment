"""Default-off, host-reviewed knowledge grouping and milestone potential proposals."""

# Standard library
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Callable

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)
from gepa_mindfulness.verification.state import EvidenceState

from .causal_reward import _accepted
from .causal_reward_records import (
    _digest,
    _number,
    _read_request,
    _request_wire,
    _text,
    _wire,
    request_digest,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .eligibility import require_training_eligible
from .research_credit import ResearchCreditClaim, _evaluator, audit_research_credit

_TAGS = {"training_eligibility": "DEVELOPMENT", "authority_granted": False}


def _restore(value: object, schema: str, names: str) -> dict[str, object]:
    data = _wire(value, schema, "training_eligibility authority_granted " + names)
    if data["training_eligibility"] != "DEVELOPMENT" or data["authority_granted"] is not False:
        raise ValueError("learning records cannot grant training or runtime authority")
    return {key: item for key, item in data.items() if key not in {*_TAGS, "schema_version"}}


def _identity(value: object) -> TrustedEvaluatorContract:
    if type(value) is not dict or set(value) != {
        "evaluator_id",
        "evaluator_version",
        "contract_id",
    }:
        raise ValueError("invalid evaluator identity")
    return _evaluator(TrustedEvaluatorContract(**value))


@dataclass(frozen=True)
class KnowledgeCondition:
    """One task-local factual condition or milestone; no generated label is self-verifying."""

    condition_id: str
    task_id: str
    version: str
    proposition: str
    kind: str
    verifier: TrustedEvaluatorContract
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("condition_id", "task_id", "version"):
            _text(getattr(self, name), name)
        _text(self.proposition, "proposition", 8192)
        if type(self.kind) is not str or self.kind not in {"knowledge", "milestone"}:
            raise ValueError("condition kind must be knowledge or milestone")
        object.__setattr__(self, "verifier", _evaluator(self.verifier))
        if type(self.dependencies) is not tuple or len(self.dependencies) > 64:
            raise ValueError("bounded dependency tuple required")
        for dependency in self.dependencies:
            _text(dependency, "dependency")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("duplicate dependency")
        if self.kind == "knowledge" and self.dependencies:
            raise ValueError("atomic knowledge conditions cannot have milestone dependencies")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": "knowledge-condition-v1",
            **_TAGS,
            "condition_id": self.condition_id,
            "task_id": self.task_id,
            "version": self.version,
            "proposition": self.proposition,
            "kind": self.kind,
            "verifier": dict(vars(self.verifier)),
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, value: object) -> KnowledgeCondition:
        data = _restore(
            value,
            "knowledge-condition-v1",
            "condition_id task_id version proposition kind verifier dependencies",
        )
        if type(data["dependencies"]) is not list:
            raise ValueError("dependency wire must be an array")
        data["dependencies"] = tuple(data["dependencies"])
        data["verifier"] = _identity(data["verifier"])
        return cls(**data)


@dataclass(frozen=True)
class LearningContext:
    """Host-declared run identities and fixed experimental discount; not authorization."""

    authority_domain: str
    run_id: str
    task_id: str
    task_version: str
    model_version: str
    harness_version: str
    producer_id: str
    evaluator: TrustedEvaluatorContract
    dataset_digest: str
    gamma: float

    def __post_init__(self) -> None:
        for name in (
            "authority_domain",
            "run_id",
            "task_id",
            "task_version",
            "model_version",
            "harness_version",
            "producer_id",
        ):
            _text(getattr(self, name), name)
        object.__setattr__(self, "evaluator", _evaluator(self.evaluator))
        _digest(self.dataset_digest)
        _number(self.gamma)
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {"schema_version": "learning-context-v1", **_TAGS} | {
            item.name: (
                dict(vars(self.evaluator)) if item.name == "evaluator" else getattr(self, item.name)
            )
            for item in fields(self)
        }

    @classmethod
    def from_dict(cls, value: object) -> LearningContext:
        data = _restore(value, "learning-context-v1", " ".join(item.name for item in fields(cls)))
        data["evaluator"] = _identity(data["evaluator"])
        return cls(**data)


@dataclass(frozen=True)
class LearningStep:
    """Complete declared evidence transition; beliefs remain a distinct, optional observation."""

    episode_id: str
    step_index: int
    event_id: str
    request: RewardRequest
    before: EvidenceState
    after: EvidenceState
    termination: str = "continuing"
    is_answer: bool = False
    belief_before: EvidenceState | None = None
    belief_after: EvidenceState | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "event_id"):
            _text(getattr(self, name), name)
        if type(self.step_index) is not int or not 0 <= self.step_index < 128:
            raise ValueError("step index must be a bounded nonnegative integer")
        if type(self.request) is not RewardRequest:
            raise ValueError("expected exact RewardRequest")
        object.__setattr__(self, "request", _read_request(_request_wire(self.request)))
        if type(self.termination) is not str or self.termination not in {
            "continuing",
            "terminated",
            "truncated",
        }:
            raise ValueError("invalid termination kind")
        if type(self.is_answer) is not bool:
            raise ValueError("is_answer must be a boolean")
        for name in ("before", "after", "belief_before", "belief_after"):
            state = getattr(self, name)
            if state is None and name.startswith("belief_"):
                continue
            if type(state) is not EvidenceState:
                raise ValueError("expected exact EvidenceState")
            wire = state.to_dict()
            validate_input_envelope(wire)
            object.__setattr__(self, name, EvidenceState.from_dict(wire))

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        result = {"schema_version": "learning-step-v1", **_TAGS}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "request":
                value = _request_wire(value)
            elif type(value) is EvidenceState:
                value = value.to_dict()
            result[item.name] = value
        validate_input_envelope(result)
        return result

    @classmethod
    def from_dict(cls, value: object) -> LearningStep:
        data = _restore(value, "learning-step-v1", " ".join(item.name for item in fields(cls)))
        data["request"] = _read_request(data["request"])
        for name in ("before", "after", "belief_before", "belief_after"):
            if data[name] is not None:
                data[name] = EvidenceState.from_dict(data[name])
        return cls(**data)


def _state(state: EvidenceState, specs: dict[str, KnowledgeCondition], refs: set) -> dict[str, str]:
    """Require exact condition coverage, proposition identity and observable source membership."""
    claims = {claim.claim_id: claim for claim in state.claims}
    if set(claims) != set(specs):
        raise ValueError("evidence state must cover exactly the declared conditions")
    for key, claim in claims.items():
        if (
            claim.proposition != specs[key].proposition
            or claim.status not in {"supported", "contradicted", "unverified"}
            or claim.superseded_by is not None
        ):
            raise ValueError("noncanonical condition proposition or status")
        if any(not ref.is_observable or ref not in refs for ref in claim.evidence_refs):
            raise ValueError("condition evidence must belong to the observable request")
    return {key: claim.status for key, claim in claims.items()}


def _catalog(context: LearningContext, conditions: tuple) -> dict[str, KnowledgeCondition]:
    specs = {item.condition_id: item for item in conditions}
    if len(specs) != len(conditions):
        raise ValueError("duplicate condition identity")
    for spec in specs.values():
        if (spec.task_id, spec.version, spec.verifier) != (
            context.task_id,
            context.task_version,
            context.evaluator,
        ):
            raise ValueError("condition task/version/verifier mismatch")
        if not set(spec.dependencies) <= set(specs):
            raise ValueError("unknown milestone dependency")
    pending = set(specs)
    while pending:
        ready = {key for key in pending if not set(specs[key].dependencies) & pending}
        if not ready:
            raise ValueError("cyclic milestone dependencies")
        pending -= ready
    return specs


def _completed(status: dict[str, str], specs: dict[str, KnowledgeCondition]) -> list[str]:
    valid = {key for key, value in status.items() if value == "supported"}
    while True:
        closed = {key for key in valid if set(specs[key].dependencies) <= valid}
        if closed == valid:
            return sorted(key for key in valid if specs[key].kind == "milestone")
        valid = closed


def prepare_knowledge_progress(
    context: LearningContext,
    conditions: tuple[KnowledgeCondition, ...],
    steps: tuple[LearningStep, ...],
    *,
    authorize_evidence: Callable[[dict[str, object]], bool] | None = None,
    epsilon: float = 1e-6,
    enabled: bool = False,
) -> dict[str, object] | None:
    """Prepare reviewed DEVELOPMENT components, never pay or combine optimizer reward.

    The configured host must authenticate evidence, issuer independence, semantic event identity,
    fixed Markov potential and discount/termination assumptions. Label checks cannot prove them.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return None
    _number(epsilon)
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1]")
    if type(context) is not LearningContext:
        raise ValueError("expected exact LearningContext")
    ctx = LearningContext.from_dict(context.to_dict())
    if ctx.evaluator.evaluator_id in {ctx.producer_id, ctx.model_version}:
        raise ValueError("producer cannot review its own evidence")
    if type(conditions) is not tuple or not 1 <= len(conditions) <= 64:
        raise ValueError("expected 1..64 conditions")
    if type(steps) is not tuple or not 1 <= len(steps) <= 128:
        raise ValueError("expected 1..128 learning steps")
    source = {
        "context": ctx.to_dict(),
        "conditions": [],
        "steps": [],
        "normalization_epsilon": epsilon,
    }
    # Validate the combined envelope after each addition, before restoring or reviewing more data.
    for values, expected, key in (
        (conditions, KnowledgeCondition, "conditions"),
        (steps, LearningStep, "steps"),
    ):
        for value in values:
            if type(value) is not expected:
                raise ValueError("unexpected learning record type")
            source[key].append(value.to_dict())
            validate_input_envelope(source)
    specs = _catalog(ctx, tuple(KnowledgeCondition.from_dict(v) for v in source["conditions"]))
    catalog_digest = content_digest(source["conditions"])
    captured = tuple(LearningStep.from_dict(value) for value in source["steps"])
    milestones = {key for key, spec in specs.items() if spec.kind == "milestone"}
    knowledge = set(specs) - milestones
    previous, seen, rows, groups, evidence_kinds = {}, {}, [], defaultdict(list), {}
    action_ids, event_ids = set(), set()
    for item in captured:
        target = _request_wire(item.request)
        require_training_eligible(target)
        if item.request.trajectory.sampling_parameters.get("training_eligibility") != "TRAIN":
            raise ValueError("explicit TRAIN source lineage is required")
        if (item.request.trajectory.policy_version, item.request.trajectory.model_identifier) != (
            ctx.model_version,
            ctx.producer_id,
        ):
            raise ValueError("source model/version mismatch")
        action_id = item.request.trajectory.trajectory_id
        _text(action_id, "action_id")
        if action_id in action_ids or item.event_id in event_ids:
            raise ValueError("duplicate action or semantic event identity")
        action_ids.add(action_id)
        event_ids.add(item.event_id)
        refs = set(item.request.observable_references)
        for ref in item.request.trajectory.evidence_references:
            _text(ref.reference_id, "reference_id")
            old = evidence_kinds.setdefault(ref.reference_id, ref.source_kind)
            if old != ref.source_kind:
                raise ValueError("conflicting evidence source kinds")
        before, after = _state(item.before, specs, refs), _state(item.after, specs, refs)
        for belief in (item.belief_before, item.belief_after):
            if belief is not None:
                _state(belief, specs, refs)
        earlier = previous.get(item.episode_id)
        if earlier is None:
            if item.step_index != 0:
                raise ValueError("episode must start at step zero")
            seen[item.episode_id] = {key for key in knowledge if before[key] == "supported"}
        elif (
            item.step_index != earlier.step_index + 1
            or earlier.termination != "continuing"
            or item.before.to_dict() != earlier.after.to_dict()
        ):
            raise ValueError("discontinuous or already terminated episode")
        if (
            earlier is not None
            and earlier.belief_after is not None
            and (
                item.belief_before is None
                or item.belief_before.to_dict() != earlier.belief_after.to_dict()
            )
        ):
            raise ValueError("discontinuous declared belief history")
        previous[item.episode_id] = item
        acquired = {
            key for key in knowledge if after[key] == "supported" and before[key] != "supported"
        }
        fresh = acquired - seen[item.episode_id]
        before_done, after_done = _completed(before, specs), _completed(after, specs)
        phi_before = len(before_done) / len(milestones) if milestones else 0.0
        phi_after = len(after_done) / len(milestones) if milestones else 0.0
        if item.termination == "terminated":
            phi_after = 0.0
        shaping = None if item.termination == "truncated" else ctx.gamma * phi_after - phi_before
        correction = []
        if item.belief_before is not None and item.belief_after is not None:
            b0 = _state(item.belief_before, specs, refs)
            b1 = _state(item.belief_after, specs, refs)
            correction = sorted(
                key
                for key in knowledge
                if b0[key] == "contradicted"
                and b1[key] == "supported"
                and after[key] == "supported"
            )
        group_id = None
        if not item.is_answer and item.termination == "continuing":
            group_id = content_digest(
                {
                    "catalog_digest": catalog_digest,
                    "prior_state": {key: before[key] for key in sorted(knowledge)},
                    "ever_supported": sorted(seen[item.episode_id]),
                }
            )
            groups[group_id].append(len(rows))
        rows.append(
            {
                "episode_id": item.episode_id,
                "step_index": item.step_index,
                "event_id": item.event_id,
                "action_id": action_id,
                "knowledge_gain": len(fresh),
                "acquired_condition_ids": sorted(fresh),
                "reacquired_condition_ids": sorted(acquired - fresh),
                "revoked_condition_ids": sorted(
                    key
                    for key in knowledge
                    if before[key] == "supported" and after[key] != "supported"
                ),
                "belief_correction_ids": correction,
                "completed_milestone_ids": after_done,
                "blocked_milestone_ids": sorted(
                    key for key in milestones if after[key] == "supported" and key not in after_done
                ),
                "milestone_gain": len(after_done) - len(before_done),
                "phi_before": phi_before,
                "phi_after": phi_after,
                "shaping": shaping,
                "termination": item.termination,
                "group_id": group_id,
                "knowledge_advantage": None,
            }
        )
        seen[item.episode_id].update(acquired)
    for indices in groups.values():
        gains = [rows[index]["knowledge_gain"] for index in indices]
        mean = math.fsum(gains) / len(gains)
        std = math.sqrt(math.fsum((value - mean) ** 2 for value in gains) / len(gains))
        for index in indices:
            rows[index]["knowledge_advantage"] = (rows[index]["knowledge_gain"] - mean) / (
                std + epsilon
            )
    claims = []
    source_digest = content_digest(source)
    for item, row in zip(captured, rows):
        for kind, value, reference in (
            ("knowledge", row["knowledge_gain"], "REF-KNOWGPO"),
            ("progress", row["shaping"], "REF-PRM4RL"),
        ):
            if value is None or value == 0:
                continue
            component = (
                "verified_knowledge_gain" if kind == "knowledge" else "verified_milestone_progress"
            )
            refs = tuple(
                dict.fromkeys(
                    ref
                    for frame in (item.before, item.after)
                    for claim in frame.claims
                    for ref in claim.evidence_refs
                )
            )
            if not refs:
                raise ValueError("a proposed contribution requires observable source evidence")
            claims.append(
                ResearchCreditClaim(
                    claim_id=content_digest([ctx.run_id, ctx.task_id, item.event_id, kind]),
                    method_id="knowgpo" if kind == "knowledge" else "prm4rl",
                    component_name=component,
                    credit_kind=kind,
                    source_reference=reference,
                    authority_domain=ctx.authority_domain,
                    run_id=ctx.run_id,
                    task_id=ctx.task_id,
                    event_id=item.event_id,
                    action_id=row["action_id"],
                    model_version=ctx.model_version,
                    harness_version=ctx.harness_version,
                    producer_id=ctx.producer_id,
                    evaluator=ctx.evaluator,
                    dataset_digest=ctx.dataset_digest,
                    state_digest=content_digest(item.before.to_dict()),
                    evidence_refs=refs,
                    score=value,
                    admissibility="unresolved",
                    provenance=RewardProvenance(
                        component,
                        "host-reviewed observable condition transition",
                        VerificationRoute.TRUSTED_EVALUATOR,
                        evaluator=ctx.evaluator,
                    ),
                    source_metadata_json=canonical_json(
                        {
                            "training_eligibility": "TRAIN",
                            "source_request": _request_wire(item.request),
                            "target_request_digest": request_digest(item.request),
                            "source_digest": source_digest,
                            "catalog_digest": catalog_digest,
                            "task_version": ctx.task_version,
                            "raw_metric": row,
                        }
                    ),
                )
            )
    audit = audit_research_credit(tuple(claims), enabled=True) if claims else None
    envelope = {
        "schema_version": "knowledge-progress-evidence-v1",
        **_TAGS,
        **source,
        "metrics": rows,
        "proposed_claims": [claim.to_dict() for claim in claims],
        "shaping_assumptions": [
            "fixed_bounded_markov_potential",
            "same_discount_as_objective",
            "zero_terminated_potential",
            "truncations_not_scored",
        ],
    }
    result = {
        "schema_version": "knowledge-progress-report-v1",
        **_TAGS,
        "source": envelope,
        "evidence_review_accepted": False,
        "steps": rows,
        "claims": envelope["proposed_claims"],
        "credit_audit": audit,
        "source_digest": source_digest,
        "catalog_digest": catalog_digest,
        "requires_host_admission": True,
    }
    result["review_decision"] = {
        "status": "evidence_accepted_admission_unresolved",
        "route": "independent_host_admission",
        "reason": "evidence review grants no admissibility or action authority",
        "authorized_follow_up": [],
    }
    validate_input_envelope(result)
    accepted = _accepted(authorize_evidence, envelope)
    result["evidence_review_accepted"] = accepted
    result["review_decision"] = {
        "status": "evidence_accepted_admission_unresolved" if accepted else "evidence_unresolved",
        "route": "independent_host_admission" if accepted else "further_verification",
        "reason": (
            "evidence review grants no admissibility or action authority"
            if accepted
            else "independent evidence review unavailable or rejected"
        ),
        "authorized_follow_up": [],
    }
    if not accepted:
        result["claims"] = []
        result["credit_audit"] = None
    validate_input_envelope(result)
    return result
