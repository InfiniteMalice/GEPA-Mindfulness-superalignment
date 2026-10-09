"""Final review regressions: current conclusions and independently sufficient support routes."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from artifact_fixtures import make_assessment, make_capture, make_derived_snapshot, make_protocol
from test_evidence_topology import analyze

# Local
from evaluation.evidence_topology_records import ClaimSupportVerdict
from gepa_mindfulness.verification.artifact_records import payload_digest
from gepa_mindfulness.verification.state import EvidenceState


@pytest.mark.parametrize("status", ["stale", "contradicted", "superseded", "unavailable"])
def test_positive_receipt_cannot_clear_conclusion_status(status):
    p = make_protocol()
    goal = replace(
        p.snapshot.state.claims[-1],
        status=status,
        evidence_refs=p.snapshot.sources[0].claim.evidence_refs,
        superseded_by="b" if status == "superseded" else None,
    )
    s = replace(p.snapshot, state=EvidenceState(p.snapshot.state.claims[:-1] + (goal,)))
    p = make_protocol(snapshot=s)
    c = make_capture(p)
    out = analyze(p, c, make_assessment(p, c))
    assert out["supported_claims"]["goal"]["status"] == "blocked"
    assert out["source_claims"] == s.state.to_dict()


def test_positive_receipt_cannot_clear_derived_ancestry_restriction():
    p = make_protocol(snapshot=make_derived_snapshot())
    c = make_capture(p)
    a = make_assessment(p, c)
    out = analyze(p, c, a, authorize=lambda request: request.artifact_key == ("B", "1"))
    assert out["structure"]["available_route_ids"] == ["via-b"]
    assert out["retrieval"]["selected_item_ids"] == ["b"]
    assert out["supported_claims"]["goal"]["status"] == "blocked"
    assert analyze(p, c, a)["supported_claims"]["goal"]["status"] == "supported"


def test_full_context_receipt_does_not_certify_surviving_route():
    p = make_protocol()
    source = p.snapshot.sources[1]
    text = "The inspection occurred on Tuesday."
    claim = replace(source.claim, proposition=text)
    b = replace(
        source, quotation=text, claim=claim, memory=replace(source.memory, content_summary=text)
    )
    s = replace(
        p.snapshot,
        sources=(p.snapshot.sources[0], b),
        state=EvidenceState((p.snapshot.sources[0].claim, claim, p.snapshot.state.claims[-1])),
    )
    p = make_protocol(snapshot=s)
    c = make_capture(p)
    # A whole-context judgment without explicit independent route sufficiency.
    verdict = ClaimSupportVerdict(
        "goal", True, True, s.sources[0].claim.evidence_refs, "A proves risk"
    )
    a = replace(make_assessment(p, c), claim_verdicts=(verdict,))
    allowed = payload_digest(a)
    out = analyze(
        p,
        c,
        a,
        authenticate=lambda r: payload_digest(r) == allowed,
        authorize=lambda r: r.artifact_key == ("B", "1"),
    )
    assert out["structure"]["available_route_ids"] == ["via-b"]
    assert out["supported_claims"]["goal"]["status"] == "blocked"


def test_only_independently_adjudicated_alternative_can_survive():
    p = make_protocol()
    c = make_capture(p)
    a = make_assessment(p, c)
    first = replace(a.claim_verdicts[0], sufficient_route_ids=("via-a",))
    one_route = replace(a, claim_verdicts=(first,))
    assert analyze(p, c, one_route)["supported_claims"]["goal"]["status"] == "supported"

    def deny_a(request):
        return request.artifact_key == ("B", "1")

    assert (
        analyze(p, c, one_route, authorize=deny_a)["supported_claims"]["goal"]["status"]
        == "blocked"
    )
    both = replace(first, sufficient_route_ids=("via-a", "via-b"))
    two_routes = replace(a, claim_verdicts=(both,))
    assert (
        analyze(p, c, two_routes, authorize=deny_a)["supported_claims"]["goal"]["status"]
        == "supported"
    )


def test_sufficiency_route_ids_are_exact_and_bound():
    p = make_protocol()
    c = make_capture(p)
    a = make_assessment(p, c)
    with pytest.raises(ValueError):
        replace(a.claim_verdicts[0], sufficient_route_ids=("via-a", "via-a"))
    for claim_id, ids in (("goal", ("foreign",)), ("a", ("via-a",))):
        verdict = replace(a.claim_verdicts[0], claim_id=claim_id, sufficient_route_ids=ids)
        with pytest.raises(ValueError):
            analyze(p, c, replace(a, claim_verdicts=(verdict,)))
    bound = replace(a.claim_verdicts[0], sufficient_route_ids=("via-a",))
    assert ClaimSupportVerdict.from_dict(bound.to_dict()) == bound
    allowed = payload_digest(replace(a, claim_verdicts=(bound,)))
    forged = replace(a, claim_verdicts=(replace(bound, sufficient_route_ids=("via-b",)),))
    assert (
        analyze(p, c, forged, authenticate=lambda r: payload_digest(r) == allowed)[
            "supported_claims"
        ]["goal"]["status"]
        == "unresolved"
    )


def test_unavailable_bridge_claim_cannot_be_skipped_by_leaf_only_items():
    from artifact_fixtures import make_topology

    p = make_protocol("bridge")
    b = replace(
        p.snapshot.sources[1], claim=replace(p.snapshot.sources[1].claim, status="unavailable")
    )
    s = replace(
        p.snapshot,
        sources=(p.snapshot.sources[0], b),
        state=EvidenceState((p.snapshot.sources[0].claim, b.claim, p.snapshot.state.claims[-1])),
    )
    graph = make_topology(s, "bridge")
    graph = replace(graph, routes=(replace(graph.routes[0], item_ids=("a",)),))
    p = replace(p, snapshot=s, topology=graph)
    c = make_capture(p)
    out = analyze(p, c, make_assessment(p, c))
    assert out["structure"]["available_route_ids"] == ["bridge"]
    assert out["supported_claims"]["goal"]["status"] == "blocked"
