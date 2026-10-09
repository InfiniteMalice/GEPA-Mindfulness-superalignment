"""Explicit role permutations preserve legitimate asymmetries and original public prose."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest
from test_causal_diagnostics import captures, judgment, run
from test_causal_interventions import cells
from test_perspective_protocol import public_plan

# Local
from gepa_mindfulness.training.eligibility import TrainingEligibility
from synthetic_data.perspective_interventions import make_role_reversal_pair


def reversal(**kwargs):
    """Supply independently classified cells and explicit expected action sets."""
    source = public_plan().source
    source = replace(source, source_text="A database mentions a and b; do not replace substrings.")
    options = dict(
        after_roles=source.roles,
        actor_mapping=(("a", "b"), ("b", "a")),
        identity_only=True,
        pair_id="roles",
        family_id="roles",
        before_cell=cells()[0],
        after_cell=cells()[0],
        before_expected_actions=("answer", "clarify"),
        after_expected_actions=("answer", "clarify"),
        enabled=True,
    )
    options.update(kwargs)
    return make_role_reversal_pair(source, **options)


def test_equal_roles_preserve_prose_and_complete_changes():
    """Identity-only swaps change the explicit actor mapping, not source text."""
    p = reversal()
    assert p.claimed_equivalence and p.intervention_kind == "single_variable"
    assert p.changes[0].factor == "actor_mapping"
    assert p.before.turns[0] == p.after.turns[0]
    assert "database mentions a and b" in p.after.turns[0].content
    assert json.loads(dict(p.after.factors)["actor_mapping"]) == {"a": "b", "b": "a"}
    assert p.source_refs == public_plan().source.source_refs
    assert p.training_eligibility is TrainingEligibility.DEVELOPMENT


@pytest.mark.parametrize(
    "mapping",
    [(("a", "a"), ("b", "b")), (("a", "b"),), (("a", "b"), ("a", "a")), (("a", "b"), ("b", "c"))],
)
def test_invalid_mappings_rejected(mapping):
    """The complete actor permutation must be nonidentity and bijective."""
    with pytest.raises(ValueError):
        reversal(actor_mapping=mapping)


def test_authority_changes_cannot_claim_identity_only():
    """Unequal rights remain a compound contextual intervention."""
    roles = public_plan().source.roles
    roles = (replace(roles[0], authority=("approve",), duties=("authorize",)), roles[1])
    with pytest.raises(ValueError, match="identity"):
        reversal(after_roles=roles)
    p = reversal(after_roles=roles, identity_only=False, after_expected_actions=("defer",))
    assert not p.claimed_equivalence and p.intervention_kind == "compound"
    assert {c.factor for c in p.changes} == {"actor_mapping", "role:b:authority", "role:b:duties"}
    with pytest.raises(ValueError):
        reversal(after_roles=(roles[0], roles[0]))
    with pytest.raises(ValueError):
        reversal(enabled=False)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_causal_adjudication_distinguishes_invariance_from_required_update(asymmetric):
    """Independent acceptable outcomes, not matching phrases, determine classification."""
    roles = public_plan().source.roles
    if asymmetric:
        roles = (replace(roles[0], authority=("approve",)), roles[1])
    p = reversal(
        after_roles=roles,
        identity_only=not asymmetric,
        after_expected_actions=("defer",) if asymmetric else ("answer", "clarify"),
    )
    cs = captures(p)
    cs = (
        replace(cs[0], actions=("answer",)),
        replace(cs[1], actions=("defer",) if asymmetric else ("clarify",)),
    )
    j = judgment(
        p,
        (p,),
        cs,
        relevance="relevant" if asymmetric else "irrelevant",
        required_update=asymmetric,
        update_satisfied=True if asymmetric else None,
        change_justified=True,
    )
    report = run((p,), cs, (j,))
    assert report["pairs"][0]["classification"] == (
        "correct_sensitivity" if asymmetric else "correct_invariance"
    )
