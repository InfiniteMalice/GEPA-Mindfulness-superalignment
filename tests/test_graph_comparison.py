"""Tests for attribution graph comparison utilities."""

from __future__ import annotations

import importlib.util

import pytest

from gepa_mindfulness.interpret.attribution_graphs import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
)
from gepa_mindfulness.interpret.graph_comparison import (
    compare_graphs,
    compute_attribution_similarity,
    compute_metric_similarity,
    compute_structural_similarity,
    find_distinctive_subgraphs,
)

if importlib.util.find_spec("networkx") is None:
    pytest.skip("networkx is required for graph comparison tests.", allow_module_level=True)


def _graph_with_scale(scale: float) -> AttributionGraph:
    nodes = [
        AttributionNode(0, "attention", None, 0, 1.0, scale),
        AttributionNode(1, "mlp", None, 0, 1.0, scale / 2.0),
    ]
    edges = [AttributionEdge(nodes[0], nodes[1], scale / 3.0)]
    return AttributionGraph(
        prompt="hello",
        response="response",
        nodes=nodes,
        edges=edges,
        method="gradient_x_activation",
        metadata={},
    )


def test_identical_graphs_have_high_similarity() -> None:
    graph = _graph_with_scale(1.0)
    result = compare_graphs(graph, graph)
    for value in result.values():
        assert 0.0 <= value <= 1.0
    assert result["overall_similarity"] > 0.9


def test_structural_similarity_lower_for_different_graphs() -> None:
    g1 = _graph_with_scale(1.0)
    varied_nodes = [
        AttributionNode(0, "attention", None, 0, 1.0, 0.1),
        AttributionNode(1, "mlp", None, 0, 1.0, 0.9),
    ]
    g2 = AttributionGraph(
        prompt="hello",
        response="response",
        nodes=varied_nodes,
        edges=[AttributionEdge(varied_nodes[0], varied_nodes[1], 0.3)],
        method="gradient_x_activation",
        metadata={},
    )
    sim = compute_structural_similarity(g1.to_networkx(), g2.to_networkx())
    attr = compute_attribution_similarity(g1.to_networkx(), g2.to_networkx())
    metrics = compute_metric_similarity(
        {"path_coherence": 1.0, "entropy": 0.1},
        {"path_coherence": 0.1, "entropy": 1.0},
    )
    assert 0.0 <= sim <= 1.0
    assert 0.0 <= attr <= 1.0
    assert 0.0 <= metrics <= 1.0
    assert attr < 1.0


def test_attribution_similarity_is_scale_invariant() -> None:
    base = _graph_with_scale(1.0)
    scaled = _graph_with_scale(10.0)

    assert compute_attribution_similarity(
        base.to_networkx(), scaled.to_networkx()
    ) == pytest.approx(1.0)


def test_attribution_similarity_handles_negative_values() -> None:
    base = _graph_with_scale(-1.0)
    varied_nodes = [
        AttributionNode(0, "attention", None, 0, 1.0, -0.2),
        AttributionNode(1, "mlp", None, 0, 1.0, -0.8),
    ]
    varied = AttributionGraph(
        prompt="hello",
        response="response",
        nodes=varied_nodes,
        edges=[AttributionEdge(varied_nodes[0], varied_nodes[1], -0.3)],
        method="gradient_x_activation",
        metadata={},
    )
    similarity = compute_attribution_similarity(
        base.to_networkx(),
        varied.to_networkx(),
    )
    assert 0.0 <= similarity <= 1.0
    assert similarity < 1.0


def test_structural_similarity_accepts_fallback_two_tuple_edges() -> None:
    class TwoTupleEdgeGraph:
        nodes = ["a", "b"]

        def number_of_nodes(self) -> int:
            return 2

        def edges(self, data: bool = False):
            if data:
                raise TypeError("data keyword is unsupported")
            return [("a", "b")]

    graph = TwoTupleEdgeGraph()

    similarity = compute_structural_similarity(graph, graph)

    assert 0.0 <= similarity <= 1.0


def test_distinctive_subgraphs_is_explicitly_unsupported() -> None:
    graph = _graph_with_scale(1.0)

    with pytest.raises(NotImplementedError):
        find_distinctive_subgraphs([graph], [graph])


def test_internal_graph_scc_handles_large_graph_without_recursion() -> None:
    from gepa_mindfulness import internal_graph

    graph = internal_graph.DiGraph()
    for index in range(1200):
        graph.add_edge(index, index + 1)

    components = list(internal_graph.strongly_connected_components(graph))

    assert len(components) == 1201
