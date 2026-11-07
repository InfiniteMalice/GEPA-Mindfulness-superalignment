"""Bounded toy candidate/certificate construction, without independent verification authority."""

# Standard library
from __future__ import annotations

from itertools import product

# Local
from gepa_mindfulness.core.evidence import EvidenceReference

from .cva_records import (
    MAX_ASSIGNMENTS,
    CandidatePlan,
    CertificateType,
    ConstraintSet,
    FeasibilityScope,
    GuaranteeKind,
    NoncompletionAssessment,
    OutcomeCertificate,
    PredicateKind,
    Relation,
    SearchReport,
    _record,
    enum,
)


def _require_enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("CVA construction requires enabled=True")


def _satisfies(task: ConstraintSet, values: tuple[int, ...], scope: FeasibilityScope) -> bool:
    """Construction-side arithmetic; the checking module implements its own checker."""
    for predicate in task.predicates:
        if scope is FeasibilityScope.TECHNICAL and predicate.kind not in (
            PredicateKind.GOAL,
            PredicateKind.TECHNICAL,
        ):
            continue
        total = sum(c * v for c, v in zip(predicate.coefficients, values))
        comparisons = {
            Relation.EQ: total == predicate.bound,
            Relation.NE: total != predicate.bound,
            Relation.LE: total <= predicate.bound,
            Relation.GE: total >= predicate.bound,
        }
        if not comparisons[predicate.relation]:
            return False
    return True


def build_witness_certificate(
    constraints: ConstraintSet,
    candidate: CandidatePlan,
    *,
    evidence_refs: tuple[EvidenceReference, ...],
    scope: FeasibilityScope = FeasibilityScope.CONSTRAINT_VALID,
    enabled: bool = False,
) -> OutcomeCertificate:
    """Package an untrusted witness; this neither accepts the plan nor checks success."""
    _require_enabled(enabled)
    task = _record(constraints, ConstraintSet)
    plan = _record(candidate, CandidatePlan)
    enum(scope, FeasibilityScope)
    if (plan.task_id, plan.constraint_digest) != (task.task_id, task.digest):
        raise ValueError("candidate does not match the exact task")
    return OutcomeCertificate(
        task.task_id,
        task.digest,
        plan.digest,
        CertificateType.FINITE_WITNESS,
        scope,
        GuaranteeKind.EXACT,
        plan.values,
        (),
        evidence_refs,
    )


def build_exhaustion_certificate(
    constraints: ConstraintSet,
    *,
    scope: FeasibilityScope,
    evidence_refs: tuple[EvidenceReference, ...],
    enabled: bool = False,
) -> OutcomeCertificate:
    """Retain enumerated rejected rows; incomplete coverage cannot certify infeasibility."""
    _require_enabled(enabled)
    task = _record(constraints, ConstraintSet)
    enum(scope, FeasibilityScope)
    rows = tuple(
        row
        for row in product(*(v.values for v in task.variables))
        if not _satisfies(task, row, scope)
    )
    return OutcomeCertificate(
        task.task_id,
        task.digest,
        None,
        CertificateType.FINITE_EXHAUSTION,
        scope,
        GuaranteeKind.EXACT,
        None,
        rows,
        evidence_refs,
    )


def search_finite(
    constraints: ConstraintSet,
    *,
    budget: int,
    enabled: bool = False,
) -> SearchReport:
    """Search declared toy assignments only; even exhaustion is an unverified search report."""
    _require_enabled(enabled)
    if type(budget) is not int or not 1 <= budget <= MAX_ASSIGNMENTS:
        raise ValueError("search budget must be an integer in 1..4096")
    task = _record(constraints, ConstraintSet)
    total = 1
    for variable in task.variables:
        total *= len(variable.values)
    attempted: list[tuple[int, ...]] = []
    found = None
    if task.authorization_scope is not None and not task.unresolved_normative_judgments:
        for values in product(*(v.values for v in task.variables)):
            if len(attempted) == budget:
                break
            attempted.append(values)
            if _satisfies(task, values, FeasibilityScope.CONSTRAINT_VALID):
                found = CandidatePlan(task.task_id, task.digest, values, "finite-search proposal")
                break
    assessment = None
    if found is None:
        assessment = NoncompletionAssessment(
            task.task_id,
            task.digest,
            (f"Tested {len(attempted)} declared finite assignments without a validated witness.",),
            task.evidence_refs,
            tuple(p.predicate_id for p in task.predicates),
            None,
            tuple(
                dict.fromkeys(
                    (
                        "No independently checked certificate establishes feasibility "
                        "or infeasibility.",
                    )
                    + task.unresolved_normative_judgments
                )
            ),
            ("Confirm the formalization, applicable rules and authorization with the host.",),
            (
                "Continue useful authorized investigation or submit a scoped certificate "
                "for checking.",
            ),
        )
    return SearchReport(task.digest, tuple(attempted), len(attempted) == total, found, assessment)
