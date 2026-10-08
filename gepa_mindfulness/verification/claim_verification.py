"""Evidence-gated resolver/challenger diagnostics inspired by VeriHarness (2610.00972).

Host-owned tools execute checks under runtime_governance. This module neither executes a tool
nor writes EvidenceState. A fresh context identifier is a declaration, not authentication.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from mindful_trace_gepa.logging_schema import EventEnvelope

from ..core.evidence import EvidenceReference
from .check_records import CheckRequest, CheckResult
from .diagnostic_records import _text, public_refs, records, strings
from .epistemic_state import _nonnegative
from .failure_graph import FailureNode
from .interfaces import LocalVerificationResult, RelationalVerificationResult
from .state import EvidenceState

CONSENSUS_FALSIFIERS = (
    "wrong_source",
    "stale_version",
    "wrong_period",
    "wrong_units",
    "wrong_sign",
    "incorrect_definition",
    "omitted_requirement",
    "downstream_inference_failure",
    "unsupported_assumption",
    "inherited_bias",
    "omitted_stakeholder",
    "role_framing",
    "semantic_laundering",
    "non_independent_evidence",
    "circular_validation",
    "tool_output_injection",
    "unauthorized_evidence",
    "self_serving_exception",
    "reward_pressure_rationalization",
)


def partition_claims(
    requirements: tuple[str, ...],
    candidates: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Partition exact host-normalized values; preserve missing candidate coverage separately."""
    requirements = strings(requirements, "requirements", required=True)
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("at least one candidate is required")
    for candidate, claims in candidates.items():
        _text(candidate, "candidate_id")
        if not isinstance(claims, dict) or set(claims) - set(requirements):
            raise ValueError("candidate claims must name known requirements")
        for value in claims.values():
            _text(value, "candidate value")
    result = {}
    for requirement in requirements:
        values: dict[str, list[str]] = {}
        for candidate, claims in candidates.items():
            if requirement in claims:
                values.setdefault(claims[requirement], []).append(candidate)
        result[requirement] = {
            "partition": (
                "OMITTED" if not values else ("CONSENSUS" if len(values) == 1 else "DISPUTED")
            ),
            "values": {value: tuple(ids) for value, ids in sorted(values.items())},
            "missing_candidates": tuple(
                key for key, claims in candidates.items() if requirement not in claims
            ),
            "verdict": "unresolved",
            "training_eligibility": "DEVELOPMENT",
        }
    return result


def check_failure_node(result: CheckResult, observation: EventEnvelope) -> FailureNode:
    """Attach observed counterevidence to the existing failure graph without causal attribution.

    The caller must validate the complete action-bound sequence separately. This adapter checks
    the local observation join; it neither localizes a root cause nor creates verifier authority.
    """
    if type(result) is not CheckResult or type(observation) is not EventEnvelope:
        raise ValueError("canonical check and observation records required")
    result = CheckResult.from_dict(result.to_dict())
    if result.verdict != "contradicted" or observation.event_type != "outcome_observed":
        raise ValueError("failure attachment requires observed counterevidence")
    if (
        observation.action_id != result.action_id
        or observation.payload.get("action_id") != result.action_id
    ):
        raise ValueError("failure observation action mismatch")
    raw_refs = observation.payload.get("evidence_refs", ())
    if not isinstance(raw_refs, (list, tuple)):
        raise ValueError(
            "failure observation payload evidence_refs must be a list or tuple of strings"
        )
    payload_refs = tuple(raw_refs)
    if any(type(ref) is not str for ref in payload_refs):
        raise ValueError(
            "failure observation payload evidence_refs must be a list or tuple of strings"
        )
    if set(observation.evidence_refs) != set(payload_refs):
        raise ValueError("failure observation envelope/payload evidence mismatch")
    if not {ref.reference_id for ref in result.evidence_refs}.issubset(payload_refs):
        raise ValueError("failure evidence must be retained in observation")
    return FailureNode(
        f"check:{result.check_id}",
        observation.event_id,
        f"Check {result.check_id} contradicts public claim {result.claim_id}",
        observation.timestamp,
        result.evidence_refs,
    )


def challenge_consensus(claim_id: str, *, enabled: bool = False) -> tuple[str, ...]:
    """Return separate falsifier procedures, never a supporting majority score."""
    _text(claim_id, "claim_id")
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    return CONSENSUS_FALSIFIERS if enabled else ()


def resolver_alternatives(partition: dict[str, Any]) -> tuple[str, ...]:
    """Expose competing values for an external discriminating check, without picking a winner."""
    if partition.get("partition") != "DISPUTED":
        return ()
    values = partition.get("values")
    if not isinstance(values, dict) or len(values) < 2:
        raise ValueError("disputed partition requires competing values")
    return strings(tuple(sorted(values)), "competing values", required=True)


def prioritize_checks(
    checks: tuple[CheckRequest, ...],
    *,
    budget: float,
    enabled: bool = False,
) -> tuple[CheckRequest, ...]:
    """Greedily allocate an explicit cost budget to positive-value check proposals."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return ()
    remaining = _nonnegative(budget, "budget")
    checks = records(checks, CheckRequest)
    if len({c.check_id for c in checks}) != len(checks):
        raise ValueError("duplicate check identity")
    result = []
    for check in sorted(checks, key=lambda c: (-c.priority, c.check_id)):
        if check.priority > 0 and check.verification_cost <= remaining:
            result.append(check)
            remaining -= check.verification_cost
    return tuple(result)


def adjudicate_check(
    state: EvidenceState,
    request: CheckRequest,
    result: CheckResult,
    local: LocalVerificationResult,
    relational: RelationalVerificationResult,
    *,
    authorized_evidence: tuple[EvidenceReference, ...],
    producer_contexts: tuple[str, ...],
    adjudicator_context: str,
    enabled: bool = False,
    evidence_hazards: tuple[str, ...] = (),
) -> CheckResult:
    """Validate evidence admission and return a detached, non-authoritative verdict.

    A host must authenticate principals and capture evidence before calling this function.
    Every decisive local and relational finding must bind the result's actual evidence.
    Failed or incomplete verification downgrades to unresolved. Evidence-state commits still
    require the existing commit_verified_claim runtime authorization path.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if type(result) is not CheckResult or type(request) is not CheckRequest:
        raise ValueError("request/result must be exact check records")
    result = CheckResult.from_dict(result.to_dict())
    request = CheckRequest.from_dict(request.to_dict())
    unresolved = replace(result, verdict="unresolved", revision_claim_id=None)
    if not enabled:
        return unresolved
    if type(state) is not EvidenceState:
        raise ValueError("state must be canonical EvidenceState")
    state = EvidenceState.from_dict(state.to_dict())
    if state.resolve(request.claim_id).claim_id != request.claim_id:
        raise ValueError("check targets a superseded claim")
    contexts = strings(producer_contexts, "producer_contexts", required=True)
    _text(adjudicator_context, "adjudicator_context")
    if adjudicator_context in contexts or result.verifier_id in contexts:
        raise ValueError("adjudication/verifier requires a fresh independent context")
    if (request.check_id, request.claim_id, request.action_id) != (
        result.check_id,
        result.claim_id,
        result.action_id,
    ):
        raise ValueError("result must match check, claim and action identities")
    if (
        type(local) is not LocalVerificationResult
        or type(relational) is not RelationalVerificationResult
    ):
        raise ValueError("canonical local and relational results required")
    local = LocalVerificationResult.from_dict(local.to_dict())
    relational = RelationalVerificationResult.from_dict(relational.to_dict())
    if local.action_id != request.action_id or relational.action_id != request.action_id:
        raise ValueError("verifier result action mismatch")
    boundary = set(public_refs(authorized_evidence))
    refs = set(result.evidence_refs)
    all_refs = refs | set(local.evidence_refs) | set(relational.evidence_refs)
    if not all_refs.issubset(boundary):
        raise ValueError("verification evidence must remain inside authorized boundary")
    hazards = strings(evidence_hazards, "evidence_hazards")
    if hazards:
        # Hazards are host findings, not a detector inferred from untrusted tool prose.
        return unresolved
    if result.verdict == "unresolved":
        return unresolved
    local_fields = (
        "executed",
        "arguments_valid",
        "schema_valid",
        "authorization_valid",
        "intended_operation_observed",
    )
    relational_fields: tuple[str, ...] = (
        "task_fit",
        "dependencies_satisfied",
        "provenance_intact",
        "authorization_scope_valid",
    )
    if result.verdict == "supported":
        relational_fields += ("claimed_outcome_supported",)
    for verification, names in ((local, local_fields), (relational, relational_fields)):
        bindings = {b.field_name: set(b.evidence_refs) for b in verification.evidence_bindings}
        for name in names:
            if getattr(verification, name) is not True or not refs.intersection(
                bindings.get(name, set())
            ):
                return unresolved
    expected_status = "contradicted" if result.verdict == "contradicted" else "none"
    contradiction_refs = {
        ref
        for binding in relational.evidence_bindings
        if binding.field_name == "contradiction_status"
        for ref in binding.evidence_refs
    }
    if relational.contradiction_status != expected_status or not refs.intersection(
        contradiction_refs
    ):
        return unresolved
    return result
