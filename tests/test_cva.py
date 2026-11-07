"""Independent finite witnesses, exhaustive certificates and fail-closed CVA trust."""

# Standard library
from dataclasses import replace
from datetime import datetime, timezone

# Third-party
import pytest

# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.verification.cva import (
    HostVerificationContext,
    bind_candidate_action,
    verify_candidate,
    verify_feasibility,
)
from gepa_mindfulness.verification.cva_finite import (
    build_exhaustion_certificate,
    build_witness_certificate,
    search_finite,
)
from gepa_mindfulness.verification.cva_records import (
    CandidatePlan,
    CandidateVerdict,
    CertificateType,
    ConstraintSet,
    FeasibilityScope,
    FeasibilityVerdict,
    FiniteVariable,
    GuaranteeKind,
    LinearPredicate,
    PredicateKind,
    Relation,
)
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit

REF = EvidenceReference("fixture:roe", EvidenceSourceKind.EXTERNAL_RECORD)
CHECKER = TrustedEvaluatorContract("independent-fixture-host", "1", "formal-scope-v1")


class Clock:
    def now(self) -> datetime:
        return datetime(2026, 11, 4, tzinfo=timezone.utc)


def task(*, impossible: bool = False, excluded: bool = False) -> ConstraintSet:
    """Hand-checked: objective x=1, optionally x=0 too, or permission x=0."""
    predicates = [LinearPredicate("objective", PredicateKind.GOAL, (1,), Relation.EQ, 1)]
    if impossible:
        predicates.append(
            LinearPredicate("contradiction", PredicateKind.TECHNICAL, (1,), Relation.EQ, 0)
        )
    if excluded:
        predicates.append(
            LinearPredicate("allowed", PredicateKind.AUTHORIZATION, (1,), Relation.EQ, 0)
        )
    return ConstraintSet(
        task_id="toy-task",
        constraint_set_id="toy-constraints",
        constraint_version="1",
        claimed_objective="Set x to one",
        variables=(FiniteVariable("x", (0, 1)),),
        predicates=tuple(predicates),
        authority_hierarchy=("system", "signed-rules-of-engagement", "task-objective"),
        authorization_scope="offline:toy-task",
        evidence_refs=(REF,),
        completeness_assumptions=(
            "The declared finite domain is the complete modeled action set.",
        ),
    )


def candidate(t: ConstraintSet, value: int = 1) -> CandidatePlan:
    return CandidatePlan(t.task_id, t.digest, (value,), "finite proposal")


def host(t: ConstraintSet, *, authenticate=None) -> HostVerificationContext:
    expected = t.to_dict()
    check = authenticate or (lambda packet: packet.constraints.to_dict() == expected)
    return HostVerificationContext(CHECKER, "actor-model", Clock(), check)


def test_witness_checks_goal_technical_permissions_and_ethics() -> None:
    t = task()
    p = candidate(t)
    certificate = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    result = verify_candidate(t, p, certificate, context=host(t), enabled=True)
    assert result.verdict is CandidateVerdict.VERIFIED_SUCCESS
    assert result.verification.guarantee is GuaranteeKind.EXACT
    assert result.verification.candidate_digest == p.digest
    assert result.verification.constraint_digest == t.digest
    assert result.verification.created_at == "2026-11-04T00:00:00+00:00"
    assert result.failed_predicate_ids == ()
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible({"training_eligibility": "TRAIN", "result": result.to_dict()})


@pytest.mark.parametrize(
    "kind", [PredicateKind.TECHNICAL, PredicateKind.AUTHORIZATION, PredicateKind.ETHICAL]
)
def test_objective_success_cannot_override_other_conditions(kind: PredicateKind) -> None:
    t = task()
    t = replace(
        t, predicates=t.predicates + (LinearPredicate("boundary", kind, (1,), Relation.EQ, 0),)
    )
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    result = verify_candidate(t, p, cert, context=host(t), enabled=True)
    assert result.verdict is CandidateVerdict.CONSTRAINT_VIOLATION
    assert result.failed_predicate_ids == ("boundary",)
    assert not isinstance(result.verdict, FeasibilityVerdict)


def test_invalid_candidate_does_not_establish_infeasibility() -> None:
    t = task()
    p = candidate(t, 0)
    result = verify_candidate(
        t,
        p,
        build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True),
        context=host(t),
        enabled=True,
    )
    assert result.verdict is CandidateVerdict.CONSTRAINT_VIOLATION
    proof = build_exhaustion_certificate(
        t, scope=FeasibilityScope.CONSTRAINT_VALID, evidence_refs=(REF,), enabled=True
    )
    assert (
        verify_feasibility(t, proof, context=host(t), enabled=True).verdict
        is FeasibilityVerdict.INVALID_CERTIFICATE
    )


def test_independent_exhaustive_check_proves_only_exact_scope() -> None:
    t = task(impossible=True)
    proof = build_exhaustion_certificate(
        t, scope=FeasibilityScope.TECHNICAL, evidence_refs=(REF,), enabled=True
    )
    result = verify_feasibility(t, proof, context=host(t), enabled=True)
    assert result.verdict is FeasibilityVerdict.VERIFIED_INFEASIBLE
    assert result.scope is FeasibilityScope.TECHNICAL
    assert result.verification.candidate_digest is None
    assert result.verification.completeness_assumptions == t.completeness_assumptions
    changed = replace(t, constraint_version="2", predicates=t.predicates[:1])
    assert (
        verify_feasibility(changed, proof, context=host(changed), enabled=True).verdict
        is FeasibilityVerdict.INVALID_CERTIFICATE
    )


def test_authorized_infeasibility_does_not_claim_technical_impossibility() -> None:
    t = task(excluded=True)
    full = build_exhaustion_certificate(
        t, scope=FeasibilityScope.CONSTRAINT_VALID, evidence_refs=(REF,), enabled=True
    )
    tech = build_exhaustion_certificate(
        t, scope=FeasibilityScope.TECHNICAL, evidence_refs=(REF,), enabled=True
    )
    assert (
        verify_feasibility(t, full, context=host(t), enabled=True).verdict
        is FeasibilityVerdict.VERIFIED_INFEASIBLE
    )
    assert (
        verify_feasibility(t, tech, context=host(t), enabled=True).verdict
        is FeasibilityVerdict.INVALID_CERTIFICATE
    )


def test_failed_search_unknown_and_missing_certificate_remain_uncertain() -> None:
    t = task()
    search = search_finite(t, budget=1, enabled=True)
    assert search.candidate is None
    assert search.assessment == "NO_AUTHORIZED_SOLUTION_ESTABLISHED"
    assert search.attempted_assignments == ((0,),)
    assert search.exhaustive is False
    result = verify_feasibility(t, None, context=host(t), enabled=True)
    assert result.verdict is FeasibilityVerdict.UNRESOLVED
    assert result.verification.guarantee is GuaranteeKind.UNKNOWN
    assert result.verification.remaining_uncertainty


def test_exhausted_search_still_requires_an_independent_certificate_check() -> None:
    t = task(impossible=True)
    search = search_finite(t, budget=2, enabled=True)
    assert search.exhaustive is True
    assert search.assessment == "NO_AUTHORIZED_SOLUTION_ESTABLISHED"
    assert not hasattr(search, "verdict")


def test_incomplete_or_duplicate_exhaustion_is_not_a_proof() -> None:
    t = task(impossible=True)
    proof = build_exhaustion_certificate(
        t, scope=FeasibilityScope.TECHNICAL, evidence_refs=(REF,), enabled=True
    )
    for rows in (((0,),), ((0,), (0,))):
        malformed = replace(proof, excluded_assignments=rows)
        assert (
            verify_feasibility(t, malformed, context=host(t), enabled=True).verdict
            is FeasibilityVerdict.INVALID_CERTIFICATE
        )


@pytest.mark.parametrize("unavailable", [None, False, "throw", "mutate", "truthy"])
def test_missing_or_untrusted_host_review_cannot_self_certify(unavailable) -> None:
    t = task()
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)

    def review(packet):
        if unavailable == "throw":
            raise RuntimeError("private verifier failure")
        if unavailable == "mutate":
            object.__setattr__(packet.constraints, "claimed_objective", "substituted")
            return True
        return 1 if unavailable == "truthy" else False

    context = None if unavailable is None else host(t, authenticate=review)
    result = verify_candidate(t, p, cert, context=context, enabled=True)
    expected = (
        CandidateVerdict.VERIFIER_UNAVAILABLE if context is None else CandidateVerdict.UNRESOLVED
    )
    assert result.verdict is expected
    assert t.claimed_objective == "Set x to one"
    assert "private verifier failure" not in str(result.to_dict())


def test_missing_authorization_and_unresolved_norms_fail_closed() -> None:
    t = replace(task(), authorization_scope=None)
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    assert (
        verify_candidate(t, p, cert, context=host(t), enabled=True).verdict
        is CandidateVerdict.INSUFFICIENT_AUTHORIZATION
    )
    t = replace(task(), unresolved_normative_judgments=("Consent evidence remains unresolved.",))
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    assert (
        verify_candidate(t, p, cert, context=host(t), enabled=True).verdict
        is CandidateVerdict.UNRESOLVED
    )


def test_wrong_subject_witness_and_unsupported_guarantee_are_rejected() -> None:
    t = task()
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    variants = (
        replace(cert, candidate_digest="f" * 64),
        replace(cert, witness=(0,)),
        replace(cert, certificate_type=CertificateType.UNSUPPORTED),
        replace(cert, guarantee=GuaranteeKind.CERTIFIED_APPROXIMATION),
    )
    for altered in variants:
        assert (
            verify_candidate(t, p, altered, context=host(t), enabled=True).verdict
            is CandidateVerdict.INVALID_CERTIFICATE
        )


def test_default_off_rejects_before_host_review() -> None:
    with pytest.raises(ValueError, match="enabled=True"):
        verify_candidate(None, None, None)
    with pytest.raises(ValueError, match="enabled=True"):
        verify_feasibility(None, None)
    with pytest.raises(ValueError, match="enabled=True"):
        search_finite(None, budget=1)


def test_action_binding_uses_existing_prediction_and_exact_action_digest() -> None:
    t = task()
    p = candidate(t)
    prediction = PredictionCommit(
        "prediction-1", {"cva_candidate_digest": p.digest}, 0.5, (REF.reference_id,)
    )
    action = ActionRecord("action-1", "finite-plan", True, t.authorization_scope, "prediction-1")
    bound = bind_candidate_action(p, action, prediction)
    assert bound.action_binding.action_id == "action-1"
    assert bound.action_binding.prediction_commit_id == "prediction-1"
    assert bound.digest != p.digest
    with pytest.raises(ValueError, match="commitment"):
        bind_candidate_action(
            p, action, replace(prediction, predicted_outcome={"cva_candidate_digest": "f" * 64})
        )


def test_conflicting_evidence_is_rejected_before_host_authentication() -> None:
    t = task()
    p = candidate(t)
    conflict = EvidenceReference(REF.reference_id, EvidenceSourceKind.OBSERVABLE_OUTPUT)
    cert = build_witness_certificate(t, p, evidence_refs=(conflict,), enabled=True)
    calls = []
    with pytest.raises(ValueError, match="conflicting evidence"):
        verify_candidate(
            t,
            p,
            cert,
            context=host(t, authenticate=lambda packet: calls.append(packet) or True),
            enabled=True,
        )
    assert calls == []


def test_model_solver_labels_and_candidate_verdicts_cannot_claim_infeasibility() -> None:
    t = task()
    for label in ("UNKNOWN", "UNSAT", {"solver_status": "UNSAT"}):
        with pytest.raises(ValueError, match="OutcomeCertificate"):
            verify_feasibility(t, label, context=host(t), enabled=True)
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    result = verify_candidate(t, p, cert, context=host(t), enabled=True)
    restored = type(result).from_dict(result.to_dict())
    assert restored == result
    wire = result.to_dict()
    wire["verdict"] = "VERIFIED_INFEASIBLE"
    with pytest.raises(ValueError):
        type(result).from_dict(wire)


def test_conclusive_records_require_complete_certificate_bindings() -> None:
    t = task()
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    result = verify_candidate(t, p, cert, context=host(t), enabled=True)
    with pytest.raises(ValueError, match="certificate"):
        replace(result, verification=replace(result.verification, certificate_digest=None))


def test_self_review_unavailable_checker_and_bad_clock_cannot_verify() -> None:
    t = task()
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    with pytest.raises(ValueError, match="self-review"):
        HostVerificationContext(CHECKER, CHECKER.evaluator_id, Clock(), lambda packet: True)
    unavailable = replace(host(t), checker_available=False)
    assert (
        verify_candidate(t, p, cert, context=unavailable, enabled=True).verdict
        is CandidateVerdict.VERIFIER_UNAVAILABLE
    )

    class BadClock:
        def now(self):
            return datetime(2026, 11, 4)

    with pytest.raises(ValueError, match="aware datetime"):
        verify_candidate(t, p, cert, context=replace(host(t), clock=BadClock()), enabled=True)


def test_all_results_and_search_reports_preserve_strict_roundtrips() -> None:
    t = task(impossible=True)
    cert = build_exhaustion_certificate(
        t, scope=FeasibilityScope.TECHNICAL, evidence_refs=(REF,), enabled=True
    )
    result = verify_feasibility(t, cert, context=host(t), enabled=True)
    search = search_finite(t, budget=2, enabled=True)
    for record in (cert, result, search, search.noncompletion):
        assert type(record).from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    "relation,bound,valid",
    [
        (Relation.EQ, 1, True),
        (Relation.NE, 1, False),
        (Relation.LE, 0, False),
        (Relation.GE, 2, False),
    ],
)
def test_exact_checker_relations_are_hand_checked(relation, bound, valid) -> None:
    t = task()
    t = replace(
        t,
        predicates=t.predicates
        + (LinearPredicate("arithmetic", PredicateKind.TECHNICAL, (1,), relation, bound),),
    )
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    result = verify_candidate(t, p, cert, context=host(t), enabled=True)
    assert (result.verdict is CandidateVerdict.VERIFIED_SUCCESS) is valid


def test_witness_can_prove_scoped_feasibility_without_authorizing_success() -> None:
    t = task(excluded=True)
    p = candidate(t)
    cert = build_witness_certificate(
        t, p, scope=FeasibilityScope.TECHNICAL, evidence_refs=(REF,), enabled=True
    )
    result = verify_feasibility(t, cert, candidate=p, context=host(t), enabled=True)
    assert result.verdict is FeasibilityVerdict.VERIFIED_FEASIBLE
    assert result.scope is FeasibilityScope.TECHNICAL
    assert type(result).from_dict(result.to_dict()) == result
    assert (
        verify_candidate(t, p, cert, context=host(t), enabled=True).verdict
        is CandidateVerdict.INVALID_CERTIFICATE
    )


def test_positive_feasibility_requires_the_matching_original_candidate() -> None:
    t = task()
    p = candidate(t)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    no_candidate = verify_feasibility(t, cert, context=host(t), enabled=True)
    assert no_candidate.verdict is FeasibilityVerdict.INVALID_CERTIFICATE
    result = verify_feasibility(t, cert, candidate=p, context=host(t), enabled=True)
    assert result.verdict is FeasibilityVerdict.VERIFIED_FEASIBLE
    wrong = candidate(t, 0)
    assert (
        verify_feasibility(t, cert, candidate=wrong, context=host(t), enabled=True).verdict
        is FeasibilityVerdict.INVALID_CERTIFICATE
    )


def test_maximum_valid_inputs_fit_uncertainty_and_evidence_outputs() -> None:
    t = replace(task(), unresolved_normative_judgments=tuple(f"unknown-{i}" for i in range(32)))
    p = candidate(t)
    proof = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    result = verify_candidate(t, p, proof, context=host(t), enabled=True)
    assert result.verdict is CandidateVerdict.UNRESOLVED
    assert len(result.verification.remaining_uncertainty) == 33
    assert len(search_finite(t, budget=1, enabled=True).noncompletion.remaining_unknowns) == 33
    t = replace(
        task(),
        evidence_refs=tuple(
            EvidenceReference(f"task-{i}", EvidenceSourceKind.EXTERNAL_RECORD) for i in range(32)
        ),
    )
    p = candidate(t)
    proof = build_witness_certificate(
        t,
        p,
        evidence_refs=tuple(
            EvidenceReference(f"proof-{i}", EvidenceSourceKind.EXTERNAL_RECORD) for i in range(32)
        ),
        enabled=True,
    )
    result = verify_candidate(t, p, proof, context=host(t), enabled=True)
    assert result.verdict is CandidateVerdict.VERIFIED_SUCCESS
    assert len(result.verification.evidence_refs) == 64


def test_full_predicate_roster_survives_failed_search_reporting() -> None:
    t = task()
    extra = tuple(
        LinearPredicate(f"technical-{i}", PredicateKind.TECHNICAL, (1,), Relation.EQ, 0)
        for i in range(127)
    )
    t = replace(t, predicates=t.predicates + extra)
    result = search_finite(t, budget=2, enabled=True)
    assert len(result.noncompletion.encountered_constraints) == 128
    assert result.assessment == "NO_AUTHORIZED_SOLUTION_ESTABLISHED"


def test_full_feasibility_cannot_use_an_action_bound_to_a_foreign_scope() -> None:
    t = task()
    p = candidate(t)
    prediction = PredictionCommit(
        "prediction-foreign", {"cva_candidate_digest": p.digest}, 0.5, (REF.reference_id,)
    )
    action = ActionRecord("foreign", "finite-plan", True, "other-scope", "prediction-foreign")
    p = bind_candidate_action(p, action, prediction)
    cert = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    result = verify_feasibility(t, cert, candidate=p, context=host(t), enabled=True)
    assert result.verdict is FeasibilityVerdict.INSUFFICIENT_AUTHORIZATION


def test_uncertainty_wording_collisions_do_not_erase_noncompletion_reports() -> None:
    unknowns = (
        "No independently accepted candidate conclusion was established.",
        "No independent proof of infeasibility was established; UNKNOWN is not UNSAT.",
        "No independently checked certificate establishes feasibility or infeasibility.",
    )
    t = replace(task(), unresolved_normative_judgments=unknowns)
    p = candidate(t)
    proof = build_witness_certificate(t, p, evidence_refs=(REF,), enabled=True)
    checked = verify_candidate(t, p, proof, context=host(t), enabled=True)
    feasibility = verify_feasibility(t, None, context=host(t), enabled=True)
    searched = search_finite(t, budget=1, enabled=True)
    assert checked.verdict is CandidateVerdict.UNRESOLVED
    assert feasibility.verdict is FeasibilityVerdict.UNRESOLVED
    assert set(unknowns) <= set(checked.verification.remaining_uncertainty)
    assert set(unknowns) <= set(feasibility.verification.remaining_uncertainty)
    assert set(unknowns) <= set(searched.noncompletion.remaining_unknowns)
