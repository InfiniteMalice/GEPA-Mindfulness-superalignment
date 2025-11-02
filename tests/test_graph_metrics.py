"""Tests for attribution graph metrics."""

from __future__ import annotations

from gepa_mindfulness.interpret.attribution_graphs import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
)
from gepa_mindfulness.interpret.graph_metrics import (
    compute_all_metrics,
    compute_average_path_length,
    compute_centrality_concentration,
    compute_graph_entropy,
    compute_path_coherence,
)


def _simple_graph(high: bool = True) -> AttributionGraph:
    node_scores = (
        [0.9, 0.05, 0.05]
        if high
        else [0.34, 0.33, 0.33]
    )
    nodes = [
        AttributionNode(0, "attention", None, 0, 1.0, node_scores[0]),
        AttributionNode(1, "mlp", None, 0, 0.8, node_scores[1]),
        AttributionNode(2, "mlp", None, 0, 0.7, node_scores[2]),
    ]
    edges = [
        AttributionEdge(nodes[0], nodes[1], 0.5),
        AttributionEdge(nodes[1], nodes[2], 0.4),
    ]
    return AttributionGraph(
        prompt="hello",
        response="response",
        nodes=nodes,
        edges=edges,
        method="gradient_x_activation",
        metadata={},
    )


def test_path_coherence_reflects_concentration() -> None:
    concentrated = _simple_graph(high=True)
    diffuse = _simple_graph(high=False)
    assert compute_path_coherence(concentrated) > compute_path_coherence(diffuse)


def test_entropy_decreases_with_concentration() -> None:
    concentrated = _simple_graph(high=True)
    diffuse = _simple_graph(high=False)
    assert compute_graph_entropy(concentrated) < compute_graph_entropy(diffuse)


def test_centrality_concentration_bounds() -> None:
    graph = _simple_graph()
    score = compute_centrality_concentration(graph)
    assert 0.0 <= score <= 1.0


def test_average_path_length_handles_components() -> None:
    graph = _simple_graph()
    length = compute_average_path_length(graph)
    assert length == 0.0


def test_average_path_length_large_chain_returns_zero() -> None:
    size = 1500
    nodes = [
        AttributionNode(idx, "mlp", None, idx, 1.0, 0.1)
        for idx in range(size)
    ]
    edges = [
        AttributionEdge(nodes[idx], nodes[idx + 1], 0.1)
        for idx in range(size - 1)
    ]
    graph = AttributionGraph(
        prompt="chain",
        response="response",
        nodes=nodes,
        edges=edges,
        method="gradient_x_activation",
        metadata={},
    )
    assert compute_average_path_length(graph) == 0.0


def test_compute_all_metrics_contains_expected_keys() -> None:
    graph = _simple_graph()
    metrics = compute_all_metrics(graph)
    for key in {
        "path_coherence",
        "entropy",
        "centrality_concentration",
        "average_path_length",
        "num_nodes",
        "num_edges",
    }:
        assert key in metrics
