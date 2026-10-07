"""Public graph records preserve evidence and reject authority-shaped inputs."""

import json
from dataclasses import replace

import pytest

from evaluation.cases.registry import load_case_manifest
from gepa_mindfulness.core.epistemic_process import EpistemicProcessAssessment
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.state import EvidenceClaim


def test_graph_round_trip_and_legacy_claim_identity() -> None:
    from gepa_mindfulness.verification.claim_graph import (
        ClaimDecomposition,
        ClaimDependency,
        ClaimGraph,
        ClaimNode,
    )

    ref = EvidenceReference("public-commit", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    claim = EvidenceClaim("c", "Conclusion", (), "unverified")
    before = claim.to_dict()
    nodes = (
        ClaimNode(claim, "actor", 0.8, "MODEL_SELF_REPORT", 1.0, 1.0),
        ClaimNode(
            EvidenceClaim("p", "Premise", (), "unverified"),
            "actor",
            None,
            "LEGACY_UNSPECIFIED",
            1.0,
            0.5,
        ),
    )
    edge = ClaimDependency("c", "p", "requires", (ref,))
    decomposition = ClaimDecomposition("c", ("p",), "actor", (ref,), None)
    graph = ClaimGraph(nodes, (edge,), (decomposition,))
    restored = ClaimGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert restored == graph
    assert restored.unresolved_claim_ids == ("c", "p")
    assert claim.to_dict() == before
    assert len(load_case_manifest().cases) == 17
    with pytest.raises(ValueError):
        EpistemicProcessAssessment((graph,))
    with pytest.raises(ValueError, match="unknown"):
        replace(graph, dependencies=(replace(edge, child_claim_id="missing"),))
    with pytest.raises(ValueError, match="cycle"):
        replace(graph, dependencies=(edge, replace(edge, parent_claim_id="p", child_claim_id="c")))
    with pytest.raises(ValueError):
        replace(graph, nodes=(nodes[0], nodes[0]))
    raw = graph.to_dict()
    raw["execute"] = True
    with pytest.raises(ValueError):
        ClaimGraph.from_dict(raw)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_graph_rejects_invalid_confidence(bad: object) -> None:
    from gepa_mindfulness.verification.claim_graph import ClaimNode

    with pytest.raises(ValueError):
        ClaimNode(
            EvidenceClaim("c", "Claim", (), "unverified"),
            "actor",
            bad,
            "MODEL_SELF_REPORT",
            1.0,
            1.0,
        )


def test_private_decomposition_is_not_public_provenance() -> None:
    from gepa_mindfulness.verification.claim_graph import ClaimDependency

    ref = EvidenceReference("hidden", EvidenceSourceKind.PRIVATE_REASONING)
    with pytest.raises(ValueError, match="observable"):
        ClaimDependency("c", "p", "requires", (ref,))


def test_graph_validates_supersession_closure_and_combined_cycles() -> None:
    from gepa_mindfulness.verification.claim_graph import ClaimDependency, ClaimGraph, ClaimNode

    ref = EvidenceReference("public", EvidenceSourceKind.OBSERVABLE_OUTPUT)

    def node(key: str, successor: str | None = None) -> ClaimNode:
        claim = EvidenceClaim(key, key, (), "superseded" if successor else "unverified", successor)
        return ClaimNode(claim, "actor", None, "LEGACY_UNSPECIFIED", 1, 1)

    with pytest.raises(ValueError, match="unknown"):
        ClaimGraph((node("old", "missing"),))
    with pytest.raises(ValueError, match="cycle"):
        ClaimGraph((node("old", "new"), node("new", "old")))
    with pytest.raises(ValueError, match="cycle"):
        ClaimGraph(
            (node("old", "new"), node("new")),
            (ClaimDependency("new", "old", "requires", (ref,)),),
        )
    graph = ClaimGraph((node("old", "new"), node("new")))
    assert ClaimGraph.from_dict(graph.to_dict()) == graph
    assert graph.unresolved_claim_ids == ("new",)


def test_stakeholder_inference_preserves_uncertainty_and_constraints() -> None:
    from gepa_mindfulness.verification.perspective_records import Perspective, Stakeholder

    ref = EvidenceReference("task", EvidenceSourceKind.EXTERNAL_RECORD)
    stakeholder = Stakeholder(
        "s",
        "affected party",
        "direct",
        "inferred",
        (ref,),
        ("privacy",),
        ("consent",),
        ("quiet",),
        (ref,),
        0.5,
    )
    assert Stakeholder.from_dict(stakeholder.to_dict()) == stakeholder
    perspective = Perspective("v", "core", "facts", "affected_party", ("s",), False, "seed")
    assert Perspective.from_dict(perspective.to_dict()) == perspective
    with pytest.raises(ValueError):
        replace(stakeholder, preference_uncertainty=None)
    with pytest.raises(ValueError):
        replace(perspective, material_facts_changed=1)
    with pytest.raises(ValueError):
        Stakeholder.from_dict(stakeholder.to_dict() | {"authority": True})


def test_checks_preserve_factors_and_evidence_links() -> None:
    from gepa_mindfulness.verification.check_records import CheckRequest, CheckResult

    ref = EvidenceReference("source", EvidenceSourceKind.EXTERNAL_RECORD)
    request = CheckRequest(
        "check", "claim", "falsifier", "source_version", 0.5, 0.8, 1.0, 2.0, (ref,), "action"
    )
    assert request.priority == pytest.approx(0.2)
    assert CheckRequest.from_dict(request.to_dict()) == request
    result = CheckResult("check", "claim", "action", "unresolved", (), "verifier", None)
    assert CheckResult.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError):
        replace(result, verdict="supported")
    with pytest.raises(ValueError):
        replace(request, verification_cost=0)
