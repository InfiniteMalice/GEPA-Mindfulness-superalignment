"""Optional independent finite CVA checking with explicit host authentication boundaries.

Only trusted host code configures reviewers/clocks and owns task/authorization formalizations.
No callback may be loaded from model/scenario text. A True fixture callback is not an identity
provider or factual verifier; arbitrary Python callers are outside this library's threat model.
Results are diagnostic records, not execution, evidence-store admission or optimizer credentials.
"""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone

# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit

from .cva_records import (
    ActionBinding,
    CandidatePlan,
    CandidateVerdict,
    CandidateVerification,
    CertificateType,
    ConstraintSet,
    FeasibilityScope,
    FeasibilityVerdict,
    FeasibilityVerification,
    GuaranteeKind,
    OutcomeCertificate,
    PredicateKind,
    Relation,
    VerificationInput,
    VerificationRecord,
    _record,
    evaluator,
    text,
)
from .runtime_governance import TrustedClock, action_record_digest

CHECKER_ID = "gepa-finite-integer-cva"
CHECKER_VERSION = "1"


@dataclass(frozen=True)
class HostVerificationContext:
    """Host-owned configuration; labels alone do not authenticate reviewer independence."""

    evaluator: TrustedEvaluatorContract
    actor_id: str
    clock: TrustedClock
    authenticate: Callable[[VerificationInput], bool] | None
    checker_available: bool = True

    def __post_init__(self) -> None:
        evaluator(self.evaluator)
        text(self.actor_id, "actor_id")
        if self.actor_id == self.evaluator.evaluator_id:
            raise ValueError("self-review cannot establish independent verification")
        if not callable(getattr(self.clock, "now", None)):
            raise ValueError("host clock must implement now()")
        if self.authenticate is not None and not callable(self.authenticate):
            raise ValueError("authentication must be a host callable or None")
        if type(self.checker_available) is not bool:
            raise ValueError("checker_available requires exact bool")


def _context(context: HostVerificationContext | None) -> HostVerificationContext | None:
    if context is None:
        return None
    if type(context) is not HostVerificationContext:
        raise ValueError("exact host verification context required")
    return HostVerificationContext(
        evaluator(context.evaluator),
        context.actor_id,
        context.clock,
        context.authenticate,
        context.checker_available,
    )


def _now(context: HostVerificationContext | None) -> str:
    observed = datetime.now(timezone.utc) if context is None else context.clock.now()
    if type(observed) is not datetime or observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("trusted host clock must return an aware datetime")
    return observed.astimezone(timezone.utc).isoformat()


def _input(
    constraints: ConstraintSet,
    candidate: CandidatePlan | None,
    certificate: OutcomeCertificate | None,
) -> VerificationInput:
    task = _record(constraints, ConstraintSet)
    plan = None if candidate is None else _record(candidate, CandidatePlan)
    proof = None if certificate is None else _record(certificate, OutcomeCertificate)
    packet = VerificationInput(task, plan, proof)
    _evidence(packet)
    return packet


def _evidence(packet: VerificationInput) -> tuple[EvidenceReference, ...]:
    indexed: dict[str, EvidenceReference] = {}
    proof_refs = () if packet.certificate is None else packet.certificate.evidence_refs
    for ref in packet.constraints.evidence_refs + proof_refs:
        if ref.reference_id in indexed and indexed[ref.reference_id] != ref:
            raise ValueError("conflicting evidence source kinds")
        indexed[ref.reference_id] = ref
    return tuple(indexed.values())


def _precheck(packet: VerificationInput, context: HostVerificationContext | None) -> str | None:
    if packet.constraints.authorization_scope is None:
        return "INSUFFICIENT_AUTHORIZATION"
    if context is None or not context.checker_available:
        return "VERIFIER_UNAVAILABLE"
    if context.authenticate is None:
        return "UNRESOLVED"
    candidate = VerificationInput.from_dict(packet.to_dict())
    before = candidate.to_dict()
    try:
        accepted = context.authenticate(candidate)
        if accepted is not True or candidate.to_dict() != before:
            return "UNRESOLVED"
    except Exception:
        return "UNRESOLVED"
    if packet.constraints.unresolved_normative_judgments:
        return "UNRESOLVED"
    return None


def _provenance(
    packet: VerificationInput,
    context: HostVerificationContext | None,
    *,
    method: str,
    exact: bool,
    uncertainty: tuple[str, ...],
    timestamp: str,
) -> VerificationRecord:
    task, plan, proof = packet.constraints, packet.candidate, packet.certificate
    return VerificationRecord(
        task_id=task.task_id,
        constraint_set_id=task.constraint_set_id,
        constraint_version=task.constraint_version,
        constraint_digest=task.digest,
        authority_hierarchy=task.authority_hierarchy,
        candidate_digest=None if plan is None else plan.digest,
        claimed_objective=task.claimed_objective,
        verification_method=method,
        checker_id=CHECKER_ID,
        checker_version=CHECKER_VERSION,
        host_evaluator=None if context is None else context.evaluator,
        certificate_type=CertificateType.NONE if proof is None else proof.certificate_type,
        certificate_digest=None if proof is None else proof.digest,
        evidence_refs=_evidence(packet),
        guarantee=GuaranteeKind.EXACT if exact else GuaranteeKind.UNKNOWN,
        completeness_assumptions=task.completeness_assumptions,
        remaining_uncertainty=tuple(
            dict.fromkeys(uncertainty + task.unresolved_normative_judgments)
        ),
        permitted_next_actions=(
            "Review this finding and pursue useful options through existing host authorization.",
        ),
        provenance=("Host-owned finite formalization; original evidence references retained.",),
        created_at=timestamp,
    )


def _domain(task: ConstraintSet, row: tuple[int, ...]) -> bool:
    return len(row) == len(task.variables) and all(
        row[index] in task.variables[index].values for index in range(len(row))
    )


def _failed(task: ConstraintSet, row: tuple[int, ...], scope: FeasibilityScope) -> tuple[str, ...]:
    """Independent checking-side arithmetic, separate from candidate construction."""
    failed = []
    for predicate in task.predicates:
        if scope is FeasibilityScope.TECHNICAL and predicate.kind in (
            PredicateKind.AUTHORIZATION,
            PredicateKind.ETHICAL,
        ):
            continue
        total = 0
        for index in range(len(task.variables)):
            total += row[index] * predicate.coefficients[index]
        relation, bound = predicate.relation, predicate.bound
        invalid = (
            (relation is Relation.EQ and total != bound)
            or (relation is Relation.NE and total == bound)
            or (relation is Relation.LE and total > bound)
            or (relation is Relation.GE and total < bound)
        )
        if invalid:
            failed.append(predicate.predicate_id)
    return tuple(failed)


def _subject(packet: VerificationInput) -> bool:
    task, plan, proof = packet.constraints, packet.candidate, packet.certificate
    if proof is not None and (
        proof.task_id != task.task_id
        or proof.constraint_digest != task.digest
        or proof.guarantee is not GuaranteeKind.EXACT
    ):
        return False
    if plan is not None:
        if plan.task_id != task.task_id or plan.constraint_digest != task.digest:
            return False
        if plan.action_binding is not None:
            unbound = replace(plan, action_binding=None)
            if plan.action_binding.source_candidate_digest != unbound.digest:
                return False
    return True


def _witness(
    packet: VerificationInput,
    scope: FeasibilityScope = FeasibilityScope.CONSTRAINT_VALID,
) -> bool:
    plan, proof = packet.candidate, packet.certificate
    return (
        _subject(packet)
        and plan is not None
        and proof is not None
        and proof.certificate_type is CertificateType.FINITE_WITNESS
        and proof.scope is scope
        and proof.candidate_digest == plan.digest
        and proof.witness == plan.values
        and _domain(packet.constraints, plan.values)
    )


def verify_candidate(
    constraints: ConstraintSet,
    candidate: CandidatePlan,
    certificate: OutcomeCertificate | None,
    *,
    context: HostVerificationContext | None = None,
    enabled: bool = False,
) -> CandidateVerification:
    """Check one modeled plan. Its rejection never establishes task infeasibility."""
    if enabled is not True:
        raise ValueError("CVA checking requires enabled=True")
    packet = _input(constraints, candidate, certificate)
    if packet.candidate is None:
        raise ValueError("candidate checking requires one exact candidate")
    host = _context(context)
    timestamp = _now(host)
    status = _precheck(packet, host)
    failed: tuple[str, ...] = ()
    if status is None:
        binding = packet.candidate.action_binding
        if (
            binding is not None
            and binding.authorization_scope != packet.constraints.authorization_scope
        ):
            status = "INSUFFICIENT_AUTHORIZATION"
        elif certificate is None:
            status = "UNRESOLVED"
        elif not _witness(packet):
            status = "INVALID_CERTIFICATE"
        else:
            failed = _failed(
                packet.constraints, packet.candidate.values, FeasibilityScope.CONSTRAINT_VALID
            )
            status = "CONSTRAINT_VIOLATION" if failed else "VERIFIED_SUCCESS"
    verdict = CandidateVerdict(status)
    exact = verdict in (CandidateVerdict.VERIFIED_SUCCESS, CandidateVerdict.CONSTRAINT_VIOLATION)
    uncertainty = (
        ("The finding applies only to the exact finite formalization, not real-world execution.",)
        if exact
        else ("No independently accepted candidate conclusion was established.",)
    )
    record = _provenance(
        packet,
        host,
        method="finite-witness-check-v1" if exact else "not_verified",
        exact=exact,
        uncertainty=uncertainty,
        timestamp=timestamp,
    )
    return CandidateVerification(record, verdict, failed)


def verify_feasibility(
    constraints: ConstraintSet,
    certificate: OutcomeCertificate | None,
    *,
    candidate: CandidatePlan | None = None,
    context: HostVerificationContext | None = None,
    enabled: bool = False,
) -> FeasibilityVerification:
    """Check a witness or finite exclusion coverage, never an unsupported solver label."""
    if enabled is not True:
        raise ValueError("CVA checking requires enabled=True")
    packet = _input(constraints, candidate, certificate)
    host = _context(context)
    timestamp = _now(host)
    status = _precheck(packet, host)
    task, proof = packet.constraints, packet.certificate
    scope = FeasibilityScope.CONSTRAINT_VALID if proof is None else proof.scope
    if status is None:
        plan = packet.candidate
        if (
            scope is FeasibilityScope.CONSTRAINT_VALID
            and plan is not None
            and plan.action_binding is not None
            and plan.action_binding.authorization_scope != task.authorization_scope
        ):
            status = "INSUFFICIENT_AUTHORIZATION"
        elif proof is None:
            status = "UNRESOLVED"
        elif proof.certificate_type is CertificateType.FINITE_WITNESS:
            valid = (
                _witness(packet, scope)
                and packet.candidate is not None
                and not _failed(task, packet.candidate.values, scope)
            )
            status = "VERIFIED_FEASIBLE" if valid else "INVALID_CERTIFICATE"
        elif (
            not _subject(packet)
            or proof.certificate_type is not CertificateType.FINITE_EXHAUSTION
            or packet.candidate is not None
        ):
            status = "INVALID_CERTIFICATE"
        else:
            # Every unique member is in the domain and cardinality equals the Cartesian product.
            # Thus coverage is complete without trusting construction-side enumeration.
            count = 1
            for variable in task.variables:
                count *= len(variable.values)
            rows = proof.excluded_assignments
            complete = len(rows) == count and len(set(rows)) == count
            valid = complete and all(
                _domain(task, row) and _failed(task, row, scope) for row in rows
            )
            status = "VERIFIED_INFEASIBLE" if valid else "INVALID_CERTIFICATE"
    verdict = FeasibilityVerdict(status)
    exact = verdict in (
        FeasibilityVerdict.VERIFIED_INFEASIBLE,
        FeasibilityVerdict.VERIFIED_FEASIBLE,
    )
    uncertainty = (
        ("The feasibility finding is limited to the exact finite domain, scope and assumptions.",)
        if exact
        else ("No independent proof of infeasibility was established; UNKNOWN is not UNSAT.",)
    )
    record = _provenance(
        packet,
        host,
        method=(
            "finite-witness-check-v1"
            if verdict is FeasibilityVerdict.VERIFIED_FEASIBLE
            else "finite-exhaustion-check-v1" if exact else "not_verified"
        ),
        exact=exact,
        uncertainty=uncertainty,
        timestamp=timestamp,
    )
    return FeasibilityVerification(record, verdict, scope)


def bind_candidate_action(
    candidate: CandidatePlan,
    action: ActionRecord,
    prediction: PredictionCommit,
) -> CandidatePlan:
    """Bind a proposal to a precommitted existing action, without asserting execution."""
    plan = _record(candidate, CandidatePlan)
    if type(action) is not ActionRecord or type(prediction) is not PredictionCommit:
        raise ValueError("exact existing action and prediction records required")
    if plan.action_binding is not None:
        raise ValueError("candidate action is already bound")
    action_hash = action_record_digest(action)
    checked = PredictionCommit(
        prediction.prediction_commit_id,
        prediction.predicted_outcome,
        prediction.confidence,
        prediction.evidence_refs,
    )
    if action.prediction_commit_id != checked.prediction_commit_id or checked.to_dict()[
        "predicted_outcome"
    ] != {"cva_candidate_digest": plan.digest}:
        raise ValueError("prediction commitment must identify this exact unbound candidate")
    binding = ActionBinding(
        action.action_id,
        action.prediction_commit_id,
        action_hash,
        content_digest(checked.to_dict()),
        plan.digest,
        action.authorization_scope,
    )
    return replace(plan, action_binding=binding)
