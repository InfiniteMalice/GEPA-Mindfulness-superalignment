"""Contracts for explicit public formal audits and bounded inference retries."""

from dataclasses import FrozenInstanceError, replace
from importlib import import_module

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind


def test_tautology_without_premises_cannot_supply_grounding(formal):
    result = audit(formal, claim(formal, "True", ()))
    assert result.status is formal.FormalAuditStatus.VALID
    assert result.premise_grounding_status is formal.PremiseGroundingStatus.UNKNOWN


@pytest.fixture
def formal():
    try:
        return import_module("gepa_mindfulness.verification.formal_reasoning")
    except ModuleNotFoundError:
        pytest.fail("The opt-in public formal reasoning audit is not implemented")


def ref(name, kind=EvidenceSourceKind.EXTERNAL_RECORD):
    return EvidenceReference(name, kind)


def claim(formal, conclusion="q", premises=("p", "implies(p, q)"), **changes):
    values = {
        "claim_id": "claim-1",
        "premises": premises,
        "conclusion": conclusion,
        "public_reasoning_ref": ref("public-1", EvidenceSourceKind.OBSERVABLE_OUTPUT),
        "evidence_refs": (ref("source-1"),),
        "provenance_refs": (ref("trace-1"),),
    }
    values.update(changes)
    return formal.FormalClaim(**values)


def audit(formal, item=None, **kwargs):
    return formal.audit_formal_claim(
        item or claim(formal), audit_id="audit-1", enabled=True, **kwargs
    )


def test_grounded_validity_retains_exact_public_formulas_and_independent_grounding(formal):
    item = claim(formal)
    groundings = tuple(
        formal.PremiseGrounding(
            premise, formal.PremiseGroundingStatus.GROUNDED, (ref("fact"),), (ref("verifier"),)
        )
        for premise in item.premises
    )
    result = audit(formal, item, premise_grounding=groundings)
    assert result.status is formal.FormalAuditStatus.VALID
    assert result.premise_grounding_status is formal.PremiseGroundingStatus.GROUNDED
    assert result.exact_formalization == item
    assert result.verifier_refs
    assert result.solver_backend == "bounded-propositional-v1"
    assert result.premise_grounding == groundings


@pytest.mark.parametrize(
    ("premises", "conclusion", "expected"),
    [
        (("p", "implies(p, q)"), "q", "VALID"),
        (("q", "implies(p, q)"), "p", "INVALID"),
        (("p or q", "not p"), "q", "VALID"),
        (("p and q",), "p", "VALID"),
        ((), "p or not p", "VALID"),
        (("False",), "q", "VALID"),
        (("p",), "forall(x, p)", "UNSUPPORTED_TRANSLATION"),
        (("p",), "p == True", "UNSUPPORTED_TRANSLATION"),
        (("p",), "p + q", "UNSUPPORTED_TRANSLATION"),
        (("p",), "implies(p)", "UNSUPPORTED_TRANSLATION"),
        (("p",), "p and", "UNSUPPORTED_TRANSLATION"),
    ],
)
def test_reference_solver_has_explicit_entailment_and_translation_outcomes(
    formal, premises, conclusion, expected
):
    assert audit(formal, claim(formal, conclusion, premises)).status.value == expected


def test_valid_inference_does_not_supply_missing_premise_grounding(formal):
    result = audit(formal)
    assert result.status is formal.FormalAuditStatus.VALID
    assert result.premise_grounding_status is formal.PremiseGroundingStatus.UNKNOWN
    unsupported = formal.PremiseGrounding("p", formal.PremiseGroundingStatus.UNSUPPORTED)
    result = audit(formal, premise_grounding=(unsupported,))
    assert result.premise_grounding_status is formal.PremiseGroundingStatus.UNSUPPORTED


def test_solver_budget_exhaustion_is_unknown_not_invalid(formal):
    result = audit(formal, solver=formal.BoundedPropositionalSolver(max_variables=1))
    assert result.status is formal.FormalAuditStatus.UNKNOWN
    result = audit(formal, solver=formal.BoundedPropositionalSolver(max_assignments=1))
    assert result.status is formal.FormalAuditStatus.UNKNOWN


def test_unsupported_fragment_does_not_run_custom_solver(formal):
    class MustNotRun:
        backend_id = "must-not-run"

        def solve(self, claim):
            raise AssertionError("Unsupported fragment reached solver")

    result = audit(formal, claim(formal, logic_fragment="first-order"), solver=MustNotRun())
    assert result.status is formal.FormalAuditStatus.UNSUPPORTED_TRANSLATION
    assert result.solver_backend == "formal-fragment-validator-v1"
    assert tuple(item.reference_id for item in result.verifier_refs) == (
        "verifier:formal-fragment-validator-v1",
    )


def test_disabled_default_never_runs_solver(formal):
    with pytest.raises(ValueError, match="enabled"):
        formal.audit_formal_claim(claim(formal), audit_id="disabled")


def test_public_references_reject_private_reasoning_without_fetching_text(formal):
    with pytest.raises(ValueError, match="public_reasoning_ref"):
        claim(formal, public_reasoning_ref=ref("private", EvidenceSourceKind.PRIVATE_REASONING))
    with pytest.raises(ValueError, match="observable"):
        claim(formal, evidence_refs=(ref("private", EvidenceSourceKind.PRIVATE_REASONING),))
    assert audit(formal).status is formal.FormalAuditStatus.VALID


def test_grounding_must_bind_the_exact_premise_and_have_separate_verification(formal):
    with pytest.raises(ValueError, match="grounded"):
        formal.PremiseGrounding("p", formal.PremiseGroundingStatus.GROUNDED)
    wrong = formal.PremiseGrounding("other", formal.PremiseGroundingStatus.UNKNOWN)
    with pytest.raises(ValueError, match="premise"):
        audit(formal, premise_grounding=(wrong,))


def test_input_lists_are_detached_and_audits_are_immutable(formal):
    premises = ["p"]
    evidence = [ref("source")]
    item = claim(formal, "p", premises, evidence_refs=evidence)
    premises.append("False")
    evidence.clear()
    result = audit(formal, item)
    assert result.exact_formalization.premises == ("p",)
    assert result.exact_formalization.evidence_refs == (ref("source"),)
    with pytest.raises(FrozenInstanceError):
        result.status = formal.FormalAuditStatus.INVALID


@pytest.mark.parametrize("budget", [True, -1, 0, float("nan"), float("inf"), "2"])
def test_solver_rejects_malformed_or_nonfinite_budgets(formal, budget):
    with pytest.raises(ValueError):
        formal.BoundedPropositionalSolver(max_assignments=budget)


@pytest.mark.parametrize("budget", [True, -1, float("nan"), float("inf"), "2"])
def test_retry_rejects_malformed_or_nonfinite_budgets(formal, budget):
    with pytest.raises(ValueError):
        formal.bounded_backtrack(
            (claim(formal),), lambda *_: claim(formal), max_retries=budget, enabled=True
        )


def test_backtracking_selects_latest_valid_public_decision_and_succeeds(formal):
    first = claim(formal, "p", ("p",), claim_id="first")
    latest = claim(formal, "q", claim_id="latest")
    invalid = claim(formal, "r", claim_id="invalid")

    def retry(anchor, failed, attempt):
        assert anchor == latest
        assert failed.status is formal.FormalAuditStatus.INVALID
        assert attempt == 1
        return replace(invalid, conclusion="q", claim_id="revised")

    result = formal.bounded_backtrack((first, latest, invalid), retry, max_retries=2, enabled=True)
    assert result.final_audit.status is formal.FormalAuditStatus.VALID
    assert result.retries_used == 1
    assert result.anchor_claim_id == "latest"
    assert [item.claim.claim_id for item in result.audits] == [
        "first",
        "latest",
        "invalid",
        "revised",
    ]


def test_backtracking_exhausts_and_unknown_never_triggers_retry(formal):
    invalid = claim(formal, "r")
    result = formal.bounded_backtrack((invalid,), lambda *_: invalid, max_retries=2, enabled=True)
    assert result.final_audit.status is formal.FormalAuditStatus.INVALID
    assert result.retries_used == 2
    assert result.exhausted is True
    assert result.anchor_claim_id is None
    result = formal.bounded_backtrack(
        (invalid,),
        lambda *_: pytest.fail("UNKNOWN cannot justify backtracking"),
        max_retries=2,
        solver=formal.BoundedPropositionalSolver(max_variables=1),
        enabled=True,
    )
    assert result.final_audit.status is formal.FormalAuditStatus.UNKNOWN
    assert result.retries_used == 0


def test_audit_cannot_be_consumed_as_runtime_authorization(formal):
    from gepa_mindfulness.verification.runtime_governance import AuthorityGrantRegistry

    with pytest.raises(ValueError):
        AuthorityGrantRegistry.enroll((audit(formal),))


@pytest.mark.parametrize(
    "changes",
    [
        {"premises": "p"},
        {"premises": [float("nan")]},
        {"conclusion": ""},
        {"evidence_refs": {"mutable": []}},
        {"provenance_refs": ["untyped"]},
    ],
)
def test_malformed_claim_inputs_are_rejected(formal, changes):
    with pytest.raises(ValueError):
        claim(formal, **changes)


def test_solver_timeout_is_unknown_and_malformed_adapter_results_are_rejected(formal):
    class TimesOut:
        backend_id = "timeout-backend"

        def solve(self, claim):
            raise TimeoutError("bounded timeout")

    result = audit(formal, solver=TimesOut())
    assert result.status is formal.FormalAuditStatus.UNKNOWN
    assert result.solver_backend == "timeout-backend"

    class Malformed:
        backend_id = "malformed-backend"

        def solve(self, claim):
            return {"status": "VALID"}

    with pytest.raises(ValueError, match="SolverFinding"):
        audit(formal, solver=Malformed())


@pytest.mark.parametrize(
    "conclusion",
    ["p" * 16385, "not " * 66 + "p", " and ".join(["p"] * 1100)],
)
def test_formula_resource_limits_fail_closed(formal, conclusion):
    assert audit(formal, claim(formal, conclusion)).status is formal.FormalAuditStatus.UNKNOWN


def test_retries_have_explicit_zero_budget_and_do_not_run_when_disabled(formal):
    def cannot_retry(*args):
        pytest.fail("No retry should execute")

    result = formal.bounded_backtrack(
        (claim(formal, "r"),), cannot_retry, max_retries=0, enabled=True
    )
    assert result.retries_used == 0
    assert result.exhausted is True
    with pytest.raises(ValueError, match="enabled"):
        formal.bounded_backtrack((claim(formal),), cannot_retry, max_retries=1)


def test_backtracking_result_detaches_audit_history_and_rejects_invalid_counters(formal):
    history = [audit(formal)]
    result = formal.BacktrackingResult(history, None, 0, False)
    history.clear()
    assert result.final_audit.status is formal.FormalAuditStatus.VALID
    with pytest.raises(ValueError):
        formal.BacktrackingResult([], None, 0, False)
    with pytest.raises(ValueError):
        formal.BacktrackingResult(result.audits, None, float("nan"), False)


def test_grounding_validation_precedes_external_solver(formal):
    class MustNotRun:
        backend_id = "must-not-run"

        def solve(self, claim):
            pytest.fail("Malformed grounding reached solver")

    with pytest.raises(ValueError, match="premise"):
        audit(formal, premise_grounding=("malformed",), solver=MustNotRun())
