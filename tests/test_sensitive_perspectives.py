"""Ablations locate scrutiny; narrator changes cannot silently change constraints."""

from dataclasses import replace

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.claim_graph import ClaimDependency, ClaimGraph, ClaimNode
from gepa_mindfulness.verification.perspective_records import Perspective
from gepa_mindfulness.verification.state import EvidenceClaim


def test_empirical_ablation_and_bounded_recursive_selection() -> None:
    from gepa_mindfulness.verification.sensitive_debate import (
        PremiseAblation,
        decision_sensitivity,
        select_challenges,
    )

    ref = EvidenceReference("public", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    nodes = tuple(
        ClaimNode(
            EvidenceClaim(c, c, (), "unverified"), "actor", None, "LEGACY_UNSPECIFIED", 1.0, 1.0
        )
        for c in ("c", "p", "q")
    )
    graph = ClaimGraph(
        nodes,
        (
            ClaimDependency("c", "p", "requires", (ref,)),
            ClaimDependency("p", "q", "requires", (ref,)),
        ),
    )
    probes = (
        PremiseAblation(("p",), "ANSWER", "IDK", (ref,)),
        PremiseAblation(("p",), "ANSWER", "ANSWER", (ref,)),
        PremiseAblation(("q",), "ANSWER", "IDK", (ref,)),
    )
    diagnostics = decision_sensitivity(graph, probes)
    assert diagnostics["p"] == 0.5
    assert diagnostics["q"] == 1.0
    assert select_challenges(graph, "c", diagnostics, max_checks=2, max_depth=2, enabled=True) == (
        "q",
        "p",
    )
    assert select_challenges(graph, "c", diagnostics) == ()
    assert graph.unresolved_claim_ids == ("c", "p", "q")
    with pytest.raises(ValueError):
        decision_sensitivity(graph, (replace(probes[0], premise_ids=("missing",)),))


def test_perspective_families_and_hard_constraint_drift() -> None:
    from gepa_mindfulness.verification.perspective_robustness import (
        Judgment,
        compare_perspectives,
        perspective_variants,
    )

    seed = Perspective("v", "core", "facts", "actor", ("s",), False, "seed")
    variants = perspective_variants(seed, enabled=True)
    assert len(variants) == 5
    assert all(v.material_facts_id == "facts" and not v.material_facts_changed for v in variants)
    assert variants[-1].perspective == "role_reversal"
    judgment = Judgment("true", "independent source", ("consent",), "CLARIFY", "wait", 0.5)
    other = replace(judgment, hard_constraints=(), response_mode="ANSWER", confidence=0.9)
    finding = compare_perspectives(
        seed,
        variants[-1],
        judgment,
        other,
        expected_stakeholders=("s",),
        observed_stakeholders=("invented",),
    )
    assert finding["hard_constraint_instability"] is True
    assert finding["spurious_framing_transition"] is True
    assert finding["stakeholder_omissions"] == ("s",)
    assert finding["stakeholder_hallucinations"] == ("invented",)
    material = replace(variants[-1], material_facts_id="new", material_facts_changed=True)
    assert compare_perspectives(seed, material, judgment, other)["transition"] == "UNRESOLVED"
    with pytest.raises(ValueError):
        compare_perspectives(seed, replace(material, material_facts_changed=False), judgment, other)


@pytest.mark.parametrize(
    "status,score", [("supported", 1.0), ("unverified", None), ("unverified", 0)]
)
def test_challenge_traversal_crosses_ineligible_intermediate_claims(status, score) -> None:
    from gepa_mindfulness.verification.sensitive_debate import select_challenges

    ref = EvidenceReference("public", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    nodes = tuple(
        ClaimNode(
            EvidenceClaim(key, key, (ref,), status if key == "middle" else "unverified"),
            "actor",
            None,
            "LEGACY_UNSPECIFIED",
            1,
            1,
        )
        for key in ("root", "middle", "leaf")
    )
    graph = ClaimGraph(
        nodes,
        (
            ClaimDependency("root", "middle", "requires", (ref,)),
            ClaimDependency("middle", "leaf", "requires", (ref,)),
        ),
    )
    sensitivity = {"middle": score, "leaf": 1.0}
    assert select_challenges(graph, "root", sensitivity, max_depth=2, enabled=True) == ("leaf",)
    assert select_challenges(graph, "root", sensitivity, max_depth=1, enabled=True) == ()


@pytest.mark.parametrize(
    "metadata",
    [
        {"source_actor": "other"},
        {"confidence": 0.8},
        {"task_relevance": 0.1},
        {"decision_importance": 0.5},
        {"confidence_source": "MODEL_SELF_REPORT"},
    ],
)
def test_perspective_metadata_does_not_change_public_claims(metadata) -> None:
    from gepa_mindfulness.verification.perspective_robustness import perspective_claim_divergence

    node = ClaimNode(
        EvidenceClaim("p", "premise", (), "unverified"), "actor", None, "LEGACY_UNSPECIFIED", 1, 1
    )
    before = ClaimGraph((node,))
    after = ClaimGraph((replace(node, **metadata),))
    finding = perspective_claim_divergence(before, after, material_facts_changed=False)
    assert finding["changed_claim_ids"] == ()
    assert finding["semantic_laundering_scrutiny"] is False


def test_perspective_divergence_locates_changed_premise_below_conclusion() -> None:
    from gepa_mindfulness.verification.perspective_robustness import perspective_claim_divergence

    ref = EvidenceReference("public", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    nodes = tuple(
        ClaimNode(
            EvidenceClaim(key, key, (), "unverified"),
            "actor",
            None,
            "LEGACY_UNSPECIFIED",
            1,
            1,
        )
        for key in ("conclusion", "premise")
    )
    before = ClaimGraph(nodes, (ClaimDependency("conclusion", "premise", "requires", (ref,)),))
    after = replace(
        before,
        nodes=tuple(
            replace(
                node, claim=replace(node.claim, proposition="changed " + node.claim.proposition)
            )
            for node in nodes
        ),
    )
    finding = perspective_claim_divergence(before, after, material_facts_changed=False)
    assert finding["earliest_divergence_claim_ids"] == ("premise",)
    assert finding["semantic_laundering_scrutiny"] is True


def test_rich_schema_accepts_optional_family_and_rejects_forged_target() -> None:
    import json
    from pathlib import Path

    from gepa_mindfulness.synthetic_dataset_validation import validate_rich_record

    row = json.loads(
        Path("data/synthetic/gold/superalignment_gold_v1.jsonl").read_text().splitlines()[0]
    )
    assert not validate_rich_record(row)
    row["argument_family"] = {
        "scenario_family_id": "f",
        "semantic_core_id": "core",
        "variant_id": "v0",
        "source_variant_id": "seed",
        "changed_parameter": "evidence_strength",
        "parameter_value": 0.0,
        "material_facts_changed": False,
        "canonical_case_target": 12,
        "response_mode": "IDK",
        "confidence": 0.1,
        "evidence_sufficiency": "insufficient",
        "unresolved_claims": ["c"],
        "critical_premises": ["p"],
        "decision_sensitivity": 0.8,
        "perspective": "actor",
        "expected_transition": "NO_TRANSITION_EXPECTED",
        "distance_band": "NEAR_BOUNDARY",
        "training_eligibility": "DEVELOPMENT",
    }
    assert not validate_rich_record(row)
    row["argument_family"]["canonical_case_target"] = 18
    assert validate_rich_record(row)
