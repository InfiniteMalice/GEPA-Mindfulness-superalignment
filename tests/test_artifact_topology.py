"""Structural evidence routes preserve conjunctions, alternatives and ordered dependencies."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from artifact_fixtures import make_snapshot, make_topology

# Local
from gepa_mindfulness.verification.artifact_topology import (
    EvidenceTopology,
    SupportRoute,
    assess_support_routes,
    validate_topology,
)
from gepa_mindfulness.verification.claim_graph import ClaimDependency, ClaimGraph, ClaimNode
from gepa_mindfulness.verification.state import EvidenceClaim


@pytest.mark.parametrize(
    "layout,available,expected",
    [
        ("single", ("a",), ["via-a"]),
        ("redundant", ("a",), ["via-a"]),
        ("redundant", ("b",), ["via-b"]),
        ("synthesis", ("a",), []),
        ("synthesis", ("a", "b"), ["joint"]),
        ("bridge", ("b",), []),
        ("bridge", ("a", "b"), ["bridge"]),
    ],
)
def test_route_layouts(layout, available, expected):
    """A selected route needs every named input; different routes are alternatives."""
    s = make_snapshot()
    topology = make_topology(s, layout)
    report = assess_support_routes(s, topology, available_item_ids=available, enabled=True)
    assert report["available_route_ids"] == expected
    assert report["structural_only"] is True and report["confers_authority"] is False
    assert EvidenceTopology.from_dict(topology.to_dict()) == topology
    with pytest.raises(ValueError, match="enabled"):
        assess_support_routes(s, topology, available_item_ids=available)


def test_missing_required_support():
    """Omitted mandatory graph dependencies are invalid, not silently optional."""
    s = make_snapshot()
    t = make_topology(s, "synthesis")
    with pytest.raises(ValueError):
        validate_topology(s, replace(t, routes=(SupportRoute("skip", "goal", ("a",), ("a",)),)))
    missing = assess_support_routes(s, t, available_item_ids=(), enabled=True)
    assert missing["routes"][0]["missing_item_ids"] == ["a", "b"]
    with pytest.raises(ValueError):
        assess_support_routes(s, t, available_item_ids=("not-planned",), enabled=True)


def test_ordered_bridge():
    """A bridge cannot declare a dependent premise before its own prerequisite."""
    s = make_snapshot()
    t = make_topology(s, "bridge")
    with pytest.raises(ValueError):
        validate_topology(
            s, replace(t, routes=(replace(t.routes[0], prerequisite_claim_ids=("b", "a")),))
        )
    with pytest.raises(ValueError):
        validate_topology(s, replace(t, routes=(replace(t.routes[0], item_ids=("b",)),)))


@pytest.mark.parametrize("change", ["snapshot", "claim", "foreign", "cycle"])
def test_graph_binding(change):
    """Graph changes cannot reuse the original snapshot binding."""
    s = make_snapshot()
    t = make_topology(s, "single")
    with pytest.raises(ValueError):
        if change == "snapshot":
            validate_topology(s, replace(t, snapshot_digest="0" * 64))
        elif change == "claim":
            n = t.graph.nodes[0]
            graph = replace(
                t.graph,
                nodes=(replace(n, claim=replace(n.claim, proposition="Other")),)
                + t.graph.nodes[1:],
            )
            validate_topology(s, replace(t, graph=graph))
        elif change == "foreign":
            validate_topology(s, replace(t, routes=(replace(t.routes[0], item_ids=("absent",)),)))
        else:
            edge = ClaimDependency("a", "goal", "requires", s.sources[0].claim.evidence_refs)
            replace(t.graph, dependencies=t.graph.dependencies + (edge,))


def test_route_limits():
    """Inclusive route, prerequisite, graph and edge caps reject overflow."""
    s = make_snapshot()
    t = make_topology(s, "single")
    routes = tuple(replace(t.routes[0], route_id=str(i)) for i in range(64))
    assert len(replace(t, routes=routes).routes) == 64
    with pytest.raises(ValueError):
        replace(t, routes=routes + (replace(t.routes[0], route_id="extra"),))
    ids = tuple(str(i) for i in range(16))
    assert SupportRoute("r", "goal", ids, ("a",)).prerequisite_claim_ids == ids
    with pytest.raises(ValueError):
        SupportRoute("r", "goal", ids + ("extra",), ("a",))
    extra = tuple(
        ClaimNode(
            EvidenceClaim(str(i), "Claim", (), "unverified"),
            "host",
            None,
            "LEGACY_UNSPECIFIED",
            1,
            1,
        )
        for i in range(254)
    )
    assert len(replace(t, graph=ClaimGraph(t.graph.nodes + extra[:253])).graph.nodes) == 256
    with pytest.raises(ValueError):
        replace(t, graph=ClaimGraph(t.graph.nodes + extra))
    nodes = extra[:33]
    refs = s.sources[0].claim.evidence_refs
    edges = tuple(
        ClaimDependency(str(i), str(j), "supports", refs)
        for i in range(33)
        for j in range(i + 1, 33)
    )
    assert len(replace(t, graph=ClaimGraph(nodes, edges[:512])).graph.dependencies) == 512
    with pytest.raises(ValueError):
        replace(t, graph=ClaimGraph(nodes, edges[:513]))


def test_contradiction_is_not_support():
    """Contradictions remain diagnostic and cannot satisfy a support route."""
    s = make_snapshot()
    t = make_topology(s, "single")
    refs = s.sources[0].claim.evidence_refs
    conflict = ClaimDependency("goal", "b", "contradicts", refs)
    t = replace(t, graph=replace(t.graph, dependencies=t.graph.dependencies + (conflict,)))
    report = assess_support_routes(s, t, available_item_ids=("a", "b"), enabled=True)
    assert len(report["contradictions"]) == 1
    with pytest.raises(ValueError):
        validate_topology(s, replace(t, routes=(SupportRoute("bad", "goal", ("b",), ("b",)),)))
