"""Default-off, independently reviewed VPO set-score preparation, never reward authority."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields
from typing import Callable

from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)
from gepa_mindfulness.verification.interfaces import RelationalVerificationResult

from .actionopd import SemanticAction, public_state_digest
from .algorithms.grpo import compute_group_advantages
from .causal_reward import _accepted
from .causal_reward_records import (
    _digest,
    _evaluator_wire,
    _json_value,
    _number,
    _provenance_wire,
    _read_evaluator,
    _read_provenance,
    _read_request,
    _request_wire,
    _text,
    _wire,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .eligibility import require_training_eligible

_TAGS = {"training_eligibility": "DEVELOPMENT", "authority_granted": False}
_DIMENSIONS = frozenset(
    {
        "verified_task_quality",
        "reasoning_efficiency",
        "explanatory_clarity",
        "robustness",
        "generalization",
        "resource_efficiency",
        "evidence_grounded_completeness",
    }
)


def _restore(value: object, schema: str, names: str) -> dict:
    data = _wire(value, schema, "training_eligibility authority_granted " + names)
    if data["training_eligibility"] != "DEVELOPMENT" or data["authority_granted"] is not False:
        raise ValueError("vector records cannot grant authority")
    return {key: item for key, item in data.items() if key not in {*_TAGS, "schema_version"}}


def _bounded_tuple(value: object, maximum: int, *, minimum: int = 1) -> None:
    if type(value) is not tuple or not minimum <= len(value) <= maximum:
        raise ValueError("expected bounded exact tuple")


def _weights(value: object, dimension_count: int) -> None:
    _bounded_tuple(value, 256)
    for row in value:
        _bounded_tuple(row, dimension_count)
        if len(row) != dimension_count:
            raise ValueError("weight dimension mismatch")
        for number in row:
            _number(number, nonnegative=True)
            if number > 1:
                raise ValueError("simplex coordinates cannot exceed one")
        if not math.isclose(math.fsum(row), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("weights must sum to one (binary64 absolute tolerance 1e-12)")


@dataclass(frozen=True)
class VectorScore:
    """A declared task metric and provenance; host review still authenticates its truth."""

    dimension: str
    value: float
    provenance: RewardProvenance

    def __post_init__(self) -> None:
        if type(self.dimension) is not str or self.dimension not in _DIMENSIONS:
            raise ValueError("ineligible vector dimension")
        _number(self.value)
        if type(self.provenance) is not RewardProvenance:
            raise ValueError("expected exact RewardProvenance")
        validated = RewardProvenance(
            self.provenance.component_name,
            self.provenance.verification_method,
            self.provenance.route,
            self.provenance.evidence_refs,
            self.provenance.evaluator,
            self.provenance.public_rationale_comparison,
        )
        snapshot = _read_provenance(_provenance_wire(validated))
        if snapshot.component_name != self.dimension:
            raise ValueError("score provenance component mismatch")
        _text(snapshot.verification_method, "verification_method", 8192)
        object.__setattr__(self, "provenance", snapshot)

    def to_dict(self) -> dict:
        self.__post_init__()
        result = {
            "schema_version": "vector-score-v1",
            **_TAGS,
            "dimension": self.dimension,
            "value": self.value,
            "provenance": _provenance_wire(self.provenance),
        }
        validate_input_envelope(result)
        return result

    @classmethod
    def from_dict(cls, value: object) -> VectorScore:
        data = _restore(value, "vector-score-v1", "dimension value provenance")
        data["provenance"] = _read_provenance(data["provenance"])
        return cls(**data)


@dataclass(frozen=True)
class VectorCandidate:
    """One public candidate with independently reviewable evidence and declared admissibility."""

    candidate_id: str
    action: SemanticAction
    scores: tuple[VectorScore, ...]
    admissibility: str
    verification: RelationalVerificationResult

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        if type(self.admissibility) is not str or self.admissibility not in {
            "admitted",
            "rejected",
            "unresolved",
        }:
            raise ValueError("invalid admissibility")
        if type(self.action) is not SemanticAction:
            raise ValueError("expected exact SemanticAction")
        object.__setattr__(self, "action", SemanticAction.from_dict(self.action.to_dict()))
        _bounded_tuple(self.scores, 7)
        if any(type(score) is not VectorScore for score in self.scores):
            raise ValueError("expected exact VectorScore")
        object.__setattr__(
            self, "scores", tuple(VectorScore.from_dict(score.to_dict()) for score in self.scores)
        )
        if len({score.dimension for score in self.scores}) != len(self.scores):
            raise ValueError("duplicate score dimension")
        if type(self.verification) is not RelationalVerificationResult:
            raise ValueError("expected exact relational verification")
        verification = RelationalVerificationResult.from_dict(self.verification.to_dict())
        if verification.action_id != self.candidate_id:
            raise ValueError("verification action mismatch")
        object.__setattr__(self, "verification", verification)

    def to_dict(self) -> dict:
        self.__post_init__()
        result = {
            "schema_version": "vector-candidate-v1",
            **_TAGS,
            "candidate_id": self.candidate_id,
            "action": self.action.to_dict(),
            "scores": [score.to_dict() for score in self.scores],
            "admissibility": self.admissibility,
            "verification": self.verification.to_dict(),
        }
        validate_input_envelope(result)
        return result

    @classmethod
    def from_dict(cls, value: object) -> VectorCandidate:
        data = _restore(
            value, "vector-candidate-v1", "candidate_id action scores admissibility verification"
        )
        if type(data["scores"]) is not list:
            raise ValueError("scores wire must be an array")
        data["scores"] = tuple(VectorScore.from_dict(score) for score in data["scores"])
        data["action"] = SemanticAction.from_dict(data["action"])
        data["verification"] = RelationalVerificationResult.from_dict(data["verification"])
        return cls(**data)


@dataclass(frozen=True)
class VectorRollout:
    """An entire student-generated candidate set, not independently relabeled token spans."""

    rollout_id: str
    request: RewardRequest
    candidates: tuple[VectorCandidate, ...]

    def __post_init__(self) -> None:
        _text(self.rollout_id, "rollout_id")
        if type(self.request) is not RewardRequest:
            raise ValueError("expected exact RewardRequest")
        object.__setattr__(self, "request", _read_request(_request_wire(self.request)))
        if self.request.trajectory.trajectory_id != self.rollout_id:
            raise ValueError("rollout source identity mismatch")
        _bounded_tuple(self.candidates, 16, minimum=0)
        if any(type(c) is not VectorCandidate for c in self.candidates):
            raise ValueError("expected exact VectorCandidate")
        object.__setattr__(
            self,
            "candidates",
            tuple(VectorCandidate.from_dict(c.to_dict()) for c in self.candidates),
        )
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate candidate identity")
        expected = [
            {"candidate_id": c.candidate_id, "action": c.action.to_dict()} for c in self.candidates
        ]
        if canonical_json(_json_value(self.request.trajectory.response)) != canonical_json(
            expected
        ):
            raise ValueError("public source response must bind exact candidate actions")

    def to_dict(self) -> dict:
        self.__post_init__()
        result = {
            "schema_version": "vector-rollout-v1",
            **_TAGS,
            "rollout_id": self.rollout_id,
            "request": _request_wire(self.request),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        validate_input_envelope(result)
        return result

    @classmethod
    def from_dict(cls, value: object) -> VectorRollout:
        data = _restore(value, "vector-rollout-v1", "rollout_id request candidates")
        if type(data["candidates"]) is not list:
            raise ValueError("candidate wire must be an array")
        return cls(
            data["rollout_id"],
            _read_request(data["request"]),
            tuple(VectorCandidate.from_dict(candidate) for candidate in data["candidates"]),
        )


@dataclass(frozen=True)
class VectorPolicyConfig:
    """Explicit run identities and shared scalarizations; declarations are not admission."""

    authority_domain: str
    run_id: str
    task_id: str
    model_version: str
    harness_version: str
    producer_id: str
    evaluator: TrustedEvaluatorContract
    dataset_digest: str
    constitution_digest: str
    dimensions: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    epsilon: float

    def __post_init__(self) -> None:
        for name in (
            "authority_domain",
            "run_id",
            "task_id",
            "model_version",
            "harness_version",
            "producer_id",
        ):
            _text(getattr(self, name), name)
        object.__setattr__(self, "evaluator", _read_evaluator(_evaluator_wire(self.evaluator)))
        if self.producer_id == self.evaluator.evaluator_id:
            raise ValueError("producer cannot certify its own experiment")
        _digest(self.dataset_digest)
        _digest(self.constitution_digest)
        _bounded_tuple(self.dimensions, 7)
        if any(type(d) is not str or d not in _DIMENSIONS for d in self.dimensions):
            raise ValueError("ineligible vector dimension")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("duplicate dimension")
        _weights(self.weights, len(self.dimensions))
        _number(self.epsilon)
        if self.epsilon <= 0:
            raise ValueError("positive normalization epsilon required")

    def to_dict(self) -> dict:
        self.__post_init__()
        result = {"schema_version": "vector-policy-config-v1", **_TAGS} | {
            item.name: getattr(self, item.name) for item in fields(self)
        }
        result["evaluator"] = _evaluator_wire(self.evaluator)
        result["dimensions"] = list(self.dimensions)
        result["weights"] = [list(row) for row in self.weights]
        validate_input_envelope(result)
        return result

    @classmethod
    def from_dict(cls, value: object) -> VectorPolicyConfig:
        data = _restore(value, "vector-policy-config-v1", " ".join(f.name for f in fields(cls)))
        if type(data["dimensions"]) is not list or type(data["weights"]) is not list:
            raise ValueError("dimensions and weights wire must be arrays")
        if any(type(row) is not list for row in data["weights"]):
            raise ValueError("weight rows must be arrays")
        data["dimensions"] = tuple(data["dimensions"])
        data["weights"] = tuple(tuple(row) for row in data["weights"])
        data["evaluator"] = _read_evaluator(data["evaluator"])
        return cls(**data)


def common_dirichlet_weights(
    dimensions: int, draws: int, *, seed: int
) -> tuple[tuple[float, ...], ...]:
    """Sample Dir(1) locally; retain the returned matrix for actual comparisons."""
    if type(dimensions) is not int or not 1 <= dimensions <= 7:
        raise ValueError("expected 1..7 dimensions")
    if type(draws) is not int or not 1 <= draws <= 256:
        raise ValueError("expected 1..256 draws")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("expected unsigned 64-bit seed")
    rng = random.Random(seed)
    rows = []
    for _ in range(draws):
        values = [rng.expovariate(1.0) for _ in range(dimensions)]
        total = math.fsum(values)
        if total <= 0:
            raise ValueError("degenerate random draw")
        rows.append(tuple(value / total for value in values))
    result = tuple(rows)
    _weights(result, dimensions)
    return result


def vector_set_score(
    vectors: tuple[tuple[float, ...], ...], weights: tuple[tuple[float, ...], ...]
) -> float:
    """Paper mean-of-weighted-maxima kernel; pure diagnostics, with no admissibility claim."""
    _bounded_tuple(vectors, 16)
    _bounded_tuple(vectors[0], 7)
    dimension_count = len(vectors[0])
    _weights(weights, dimension_count)
    for vector in vectors:
        _bounded_tuple(vector, dimension_count)
        if len(vector) != dimension_count:
            raise ValueError("score dimension mismatch")
        for value in vector:
            _number(value)
    try:
        result = math.fsum(
            max(math.fsum(w * r for w, r in zip(row, vector)) for vector in vectors)
            for row in weights
        ) / len(weights)
        if not math.isfinite(result):
            raise ValueError("nonfinite vector arithmetic")
        return result
    except OverflowError:
        raise ValueError("nonfinite vector arithmetic") from None


def _structurally_eligible(candidate: VectorCandidate) -> bool:
    """Check declared records; actual admissibility still requires independent host review."""
    review = candidate.verification
    return (
        candidate.admissibility == "admitted"
        and review.task_fit
        and review.dependencies_satisfied
        and review.contradiction_status == "none"
        and review.provenance_intact
        and review.authorization_scope_valid
        and review.claimed_outcome_supported
        and not review.repeated_failed_route
    )


def prepare_vector_group(
    config: VectorPolicyConfig,
    rollouts: tuple[VectorRollout, ...],
    *,
    authorize_evidence: Callable[[dict], bool] | None = None,
    enabled: bool = False,
) -> dict | None:
    """Quarantine unsafe whole rollouts before independent, mutation-safe host review."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be a strict boolean")
    if not enabled:
        return None
    if type(config) is not VectorPolicyConfig:
        raise ValueError("expected exact VectorPolicyConfig")
    _bounded_tuple(rollouts, 32)
    source = {"config": config.to_dict(), "rollouts": []}
    for rollout in rollouts:
        if type(rollout) is not VectorRollout:
            raise ValueError("expected exact VectorRollout")
        source["rollouts"].append(rollout.to_dict())
        validate_input_envelope(source)
    ctx = VectorPolicyConfig.from_dict(source["config"])
    captured = tuple(VectorRollout.from_dict(row) for row in source["rollouts"])
    state_digest, identities, evidence_kinds = None, set(), {}
    metrics, eligible = [], []
    for item in captured:
        target = item.request
        require_training_eligible(_request_wire(target))
        if target.trajectory.sampling_parameters.get("training_eligibility") != "TRAIN":
            raise ValueError("explicit TRAIN source lineage required")
        if (target.trajectory.model_identifier, target.trajectory.policy_version) != (
            ctx.producer_id,
            ctx.model_version,
        ):
            raise ValueError("source model/version mismatch")
        digest = public_state_digest(target)
        if state_digest is not None and state_digest != digest:
            raise ValueError("rollouts must share a public prompt and sampling configuration")
        state_digest = digest
        if item.rollout_id in identities:
            raise ValueError("duplicate rollout identity")
        identities.add(item.rollout_id)
        refs = set(target.observable_references)
        all_refs = list(target.trajectory.evidence_references)
        for candidate in item.candidates:
            if tuple(score.dimension for score in candidate.scores) != ctx.dimensions:
                raise ValueError("ordered score dimensions do not match configuration")
            if not set(candidate.verification.evidence_refs).issubset(refs):
                raise ValueError("verification evidence outside observable source")
            all_refs.extend(candidate.verification.evidence_refs)
            for score in candidate.scores:
                provenance = score.provenance
                if provenance.route == VerificationRoute.TRUSTED_EVALUATOR:
                    if provenance.evaluator != ctx.evaluator:
                        raise ValueError("score evaluator mismatch")
                elif not set(provenance.evidence_refs).issubset(refs):
                    raise ValueError("score evidence outside observable source")
                all_refs.extend(provenance.evidence_refs)
        for ref in all_refs:
            _text(ref.reference_id, "reference_id")
            if (
                ref.reference_id in evidence_kinds
                and evidence_kinds[ref.reference_id] != ref.source_kind
            ):
                raise ValueError("conflicting evidence source kinds")
            evidence_kinds[ref.reference_id] = ref.source_kind
        admitted = [candidate for candidate in item.candidates if _structurally_eligible(candidate)]
        safe = bool(admitted) and len(admitted) == len(item.candidates)
        score = (
            vector_set_score(tuple(tuple(s.value for s in c.scores) for c in admitted), ctx.weights)
            if safe
            else None
        )
        if safe:
            eligible.append(len(metrics))
        metrics.append(
            {
                "rollout_id": item.rollout_id,
                "candidate_count": len(item.candidates),
                "candidate_ids_passing_declared_checks": [c.candidate_id for c in admitted],
                "disposition": "review_pending" if safe else "quarantined",
                "reason": (
                    "independent_review_required" if safe else "empty_or_incomplete_admission"
                ),
                "raw_set_score": score,
                "reviewed_set_score": None,
                "advantage": None,
            }
        )
    advantages = []
    if eligible:
        rewards = [metrics[index]["raw_set_score"] for index in eligible]
        try:
            mean = math.fsum(rewards) / len(rewards)
            variance_sum = math.fsum((reward - mean) ** 2 for reward in rewards)
            if not math.isfinite(variance_sum):
                raise ValueError("nonfinite group normalization")
            advantages = compute_group_advantages(rewards, epsilon=ctx.epsilon)
            if not all(math.isfinite(value) for value in advantages):
                raise ValueError("nonfinite group normalization")
        except OverflowError:
            raise ValueError("nonfinite group normalization") from None
    proposed = [dict(row) for row in metrics]
    for index, advantage in zip(eligible, advantages):
        proposed[index].update(
            disposition="reviewed",
            reason="evidence_accepted_no_optimizer_or_runtime_authority",
            reviewed_set_score=proposed[index]["raw_set_score"],
            advantage=advantage,
        )
    envelope = {
        "schema_version": "vector-policy-evidence-v1",
        **_TAGS,
        **source,
        "source_reference": "REF-VPO",
        "public_state_digest": state_digest,
        "metrics": metrics,
        "proposed_metrics": proposed,
        "normalization": "population_std_plus_epsilon_no_low_variance_threshold",
        "simplex_sum_absolute_tolerance": 1e-12,
        "whole_rollout_quarantine": True,
    }
    result = {
        "schema_version": "vector-policy-report-v1",
        **_TAGS,
        "source": envelope,
        "source_digest": content_digest(source),
        "evidence_review_accepted": False,
        "requires_host_admission": True,
        "rollouts": metrics,
        "fallback_required": True,
        "decision_route": "independent_host_fallback",
    }
    reviewed = result | {
        "evidence_review_accepted": True,
        "rollouts": proposed,
        "fallback_required": not bool(eligible),
        "decision_route": (
            "independent_host_fallback" if not eligible else "independent_host_admission"
        ),
    }
    # Reserve both complete serialized outcomes before invoking independent review.
    validate_input_envelope(result)
    validate_input_envelope(reviewed)
    before = canonical_json(source)
    accepted = _accepted(authorize_evidence, envelope)
    try:
        after = {"config": config.to_dict(), "rollouts": [r.to_dict() for r in rollouts]}
        accepted = accepted and canonical_json(after) == before
    except Exception:
        accepted = False
    return reviewed if accepted else result
