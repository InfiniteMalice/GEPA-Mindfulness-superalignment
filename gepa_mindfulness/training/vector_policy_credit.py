"""Bounded verified-set proposals for the existing independently governed credit owner."""

# Standard library
from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute
from gepa_mindfulness.verification.interfaces import RelationalVerificationResult

from .actionopd import SemanticAction, public_state_digest
from .causal_reward_records import (
    _evaluator_wire,
    _provenance_wire,
    _read_evaluator,
    _read_provenance,
    _read_request,
    _request_wire,
    _text,
    request_digest,
    validate_input_envelope,
)
from .vector_policy import VectorCandidate, VectorPolicyConfig, VectorRollout, VectorScore

if TYPE_CHECKING:
    from .research_credit import ResearchCreditClaim


def _event_id(rollout: VectorRollout) -> str:
    """Read host-captured trace identity without changing the public sampling state."""
    events = [
        ref.removeprefix("vpo-event:")
        for ref in rollout.request.trajectory.trace_references
        if ref.startswith("vpo-event:")
    ]
    if len(events) != 1:
        raise ValueError("one host-captured VPO semantic event reference required")
    _text(events[0], "semantic_event_id")
    return events[0]


def _raw_proof(config: VectorPolicyConfig, rollout: VectorRollout) -> dict[str, object]:
    """Retain raw source and independently reviewable facts, never relabel audit records."""
    scope = {field.name: getattr(config, field.name) for field in fields(config)}
    scope["evaluator"] = _evaluator_wire(config.evaluator)
    scope["dimensions"] = list(config.dimensions)
    scope["weights"] = [list(row) for row in config.weights]
    return {
        "schema_version": "verified-vector-set-v1",
        "config": scope,
        "members": [
            {
                "candidate_id": candidate.candidate_id,
                "action": candidate.action.to_dict(),
                "admissibility": candidate.admissibility,
                "verification": candidate.verification.to_dict(),
                "scores": [
                    {
                        "dimension": score.dimension,
                        "value": score.value,
                        "provenance": _provenance_wire(score.provenance),
                    }
                    for score in candidate.scores
                ],
            }
            for candidate in rollout.candidates
        ],
    }


def _decode(metadata: dict[str, object]) -> tuple[VectorPolicyConfig, VectorRollout, float]:
    """Recompute exact source/action/metric bindings; this does not authenticate evidence."""
    from .vector_policy import prepare_vector_group

    validate_input_envelope(metadata)
    if (
        set(metadata)
        != {"training_eligibility", "source_request", "target_request_digest", "verified_set"}
        or metadata["training_eligibility"] != "TRAIN"
    ):
        raise ValueError("invalid verified-set source metadata")
    proof = metadata["verified_set"]
    if (
        type(proof) is not dict
        or set(proof) != {"schema_version", "config", "members"}
        or proof["schema_version"] != "verified-vector-set-v1"
        or type(proof["config"]) is not dict
        or type(proof["members"]) is not list
        or not 1 <= len(proof["members"]) <= 16
    ):
        raise ValueError("invalid verified-set proof")
    scope = dict(proof["config"])
    if set(scope) != {field.name for field in fields(VectorPolicyConfig)}:
        raise ValueError("invalid vector scope")
    if type(scope["dimensions"]) not in (list, tuple) or type(scope["weights"]) not in (
        list,
        tuple,
    ):
        raise ValueError("invalid vector dimensions or weights")
    if any(type(row) not in (list, tuple) for row in scope["weights"]):
        raise ValueError("invalid vector weight rows")
    scope["dimensions"] = tuple(scope["dimensions"])
    scope["weights"] = tuple(tuple(row) for row in scope["weights"])
    scope["evaluator"] = _read_evaluator(scope["evaluator"])
    config = VectorPolicyConfig(**scope)
    request = _read_request(metadata["source_request"])
    if request_digest(request) != metadata["target_request_digest"]:
        raise ValueError("verified-set request digest mismatch")
    members = []
    for raw in proof["members"]:
        if (
            type(raw) is not dict
            or set(raw) != {"candidate_id", "action", "admissibility", "verification", "scores"}
            or type(raw["scores"]) is not list
            or not 1 <= len(raw["scores"]) <= 7
        ):
            raise ValueError("invalid raw set member")
        scores = []
        for score in raw["scores"]:
            if type(score) is not dict or set(score) != {"dimension", "value", "provenance"}:
                raise ValueError("invalid raw vector score")
            scores.append(
                VectorScore(
                    score["dimension"], score["value"], _read_provenance(score["provenance"])
                )
            )
        members.append(
            VectorCandidate(
                raw["candidate_id"],
                SemanticAction.from_dict(raw["action"]),
                tuple(scores),
                raw["admissibility"],
                RelationalVerificationResult.from_dict(raw["verification"]),
            )
        )
    rollout = VectorRollout(request.trajectory.trajectory_id, request, tuple(members))
    result = prepare_vector_group(config, (rollout,), enabled=True)
    row = result["rollouts"][0]
    if row["disposition"] == "quarantined" or row["raw_set_score"] is None:
        raise ValueError("empty or inadmissible set cannot propose optimizer credit")
    return config, rollout, row["raw_set_score"]


def validate_vector_credit(claim: ResearchCreditClaim, metadata: dict[str, object]) -> None:
    """Validate reproducible proposal facts before the existing fresh host admission."""
    config, rollout, score = _decode(metadata)
    request = rollout.request
    event_id = _event_id(rollout)
    if (
        claim.method_id != "vpo"
        or claim.source_reference != "REF-VPO"
        or claim.component_name != "verified_vector_set_quality"
        or claim.score != score
        or claim.event_id != event_id
        or claim.action_id != rollout.rollout_id
        or claim.state_digest != public_state_digest(request)
        or (
            claim.authority_domain,
            claim.run_id,
            claim.task_id,
            claim.model_version,
            claim.harness_version,
            claim.producer_id,
            claim.dataset_digest,
            claim.evaluator,
        )
        != (
            config.authority_domain,
            config.run_id,
            config.task_id,
            config.model_version,
            config.harness_version,
            config.producer_id,
            config.dataset_digest,
            config.evaluator,
        )
        or set(claim.evidence_refs) != set(request.observable_references)
        or claim.provenance is None
        or claim.provenance.route != VerificationRoute.TRUSTED_EVALUATOR
        or claim.provenance.evaluator != config.evaluator
    ):
        raise ValueError("verified-set scope, score, event or provenance mismatch")


def make_vector_credit(
    config: VectorPolicyConfig, rollout: VectorRollout, *, enabled: bool = False
) -> ResearchCreditClaim | None:
    """Propose one reproducible set score; fresh existing host review must still admit it."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return None
    from .research_credit import ResearchCreditClaim

    if type(config) is not VectorPolicyConfig or type(rollout) is not VectorRollout:
        raise ValueError("exact vector scope and rollout required")
    config = VectorPolicyConfig.from_dict(config.to_dict())
    rollout = VectorRollout.from_dict(rollout.to_dict())
    metadata = {
        "training_eligibility": "TRAIN",
        "source_request": _request_wire(rollout.request),
        "target_request_digest": request_digest(rollout.request),
        "verified_set": _raw_proof(config, rollout),
    }
    config, rollout, score = _decode(metadata)
    event_id = _event_id(rollout)
    component = "verified_vector_set_quality"
    return ResearchCreditClaim(
        claim_id=content_digest([config.run_id, config.task_id, event_id, "verified_set"]),
        method_id="vpo",
        component_name=component,
        credit_kind="verified_set",
        source_reference="REF-VPO",
        authority_domain=config.authority_domain,
        run_id=config.run_id,
        task_id=config.task_id,
        event_id=event_id,
        action_id=rollout.rollout_id,
        model_version=config.model_version,
        harness_version=config.harness_version,
        producer_id=config.producer_id,
        evaluator=config.evaluator,
        dataset_digest=config.dataset_digest,
        state_digest=public_state_digest(rollout.request),
        evidence_refs=rollout.request.observable_references,
        score=score,
        admissibility="unresolved",
        provenance=RewardProvenance(
            component,
            "fresh host verification of admissible set metrics",
            VerificationRoute.TRUSTED_EVALUATOR,
            evaluator=config.evaluator,
        ),
        source_metadata_json=canonical_json(metadata),
    )
