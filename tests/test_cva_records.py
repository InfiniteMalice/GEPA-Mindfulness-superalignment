"""Strict diagnostic CVA records reject tampering, coercion and unbounded domains."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from test_cva import candidate, task

# Local
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.cva_records import (
    CandidatePlan,
    FiniteVariable,
    LinearPredicate,
    PredicateKind,
    Relation,
)


def test_task_candidate_and_certificate_roundtrips_are_exact() -> None:
    t = task()
    p = candidate(t)
    assert type(t).from_dict(t.to_dict()) == t
    assert CandidatePlan.from_dict(p.to_dict()) == p
    copy = t.to_dict()
    copy["variables"][0]["values"][0] = -1
    assert t.variables[0].values == (0, 1)
    assert t.digest != type(t).from_dict(copy).digest


@pytest.mark.parametrize("value", [True, 1.0, float("nan"), 2**64, "1"])
def test_domain_values_require_exact_json_safe_integers(value) -> None:
    with pytest.raises(ValueError):
        FiniteVariable("x", (value,))


def test_duplicates_unsupported_relations_and_dimension_mismatch_fail() -> None:
    with pytest.raises(ValueError):
        FiniteVariable("x", (1, 1))
    t = task()
    with pytest.raises(ValueError):
        replace(t, predicates=(LinearPredicate("bad", PredicateKind.GOAL, (1, 2), Relation.EQ, 1),))
    with pytest.raises(ValueError):
        LinearPredicate("bad", PredicateKind.GOAL, (1,), "eval", 1)
    with pytest.raises(ValueError):
        replace(t, variables=tuple(FiniteVariable(f"x{i}", tuple(range(16))) for i in range(8)))


def test_private_evidence_mutation_unknown_fields_and_train_relabel_fail() -> None:
    t = task()
    private = EvidenceReference("hidden", EvidenceSourceKind.PRIVATE_REASONING)
    with pytest.raises(ValueError, match="observable"):
        replace(t, evidence_refs=(private,))
    for key, value in (
        ("training_eligibility", "TRAIN"),
        ("schema_version", "forged"),
        ("authority_granted", True),
    ):
        data = t.to_dict()
        data[key] = value
        with pytest.raises(ValueError):
            type(t).from_dict(data)
    mutated = candidate(t)
    object.__setattr__(mutated, "values", (True,))
    with pytest.raises(ValueError):
        mutated.to_dict()


def test_blank_authority_and_oversized_fields_fail_before_work() -> None:
    t = task()
    with pytest.raises(ValueError):
        replace(t, authority_hierarchy=())
    with pytest.raises(ValueError):
        replace(t, claimed_objective="x" * 10000)
    with pytest.raises(ValueError):
        replace(t, variables=())


def test_all_cva_exports_explicitly_deny_optimizer_and_authority_effects() -> None:
    t = task()
    wire = t.to_dict()
    assert wire["maturity"] == "experimental"
    assert wire["diagnostic_status"] == "diagnostic"
    assert wire["optimizer_input"] is False
    assert wire["authority_granted"] is False
    for key in ("optimizer_input", "authority_granted"):
        altered = dict(wire, **{key: 1})
        with pytest.raises(ValueError):
            type(t).from_dict(altered)
