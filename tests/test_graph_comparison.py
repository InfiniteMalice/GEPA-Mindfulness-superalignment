"""Tests for attribution graph comparison utilities."""

from __future__ import annotations

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
)


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
    g2 = _graph_with_scale(0.2)
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
