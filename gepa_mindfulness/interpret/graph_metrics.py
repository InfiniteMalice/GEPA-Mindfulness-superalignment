"""Graph metrics computed on attribution graphs."""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np

from gepa_mindfulness.interpret.attribution_graphs import AttributionGraph

_NETWORKX_SPEC = importlib.util.find_spec("networkx")
if _NETWORKX_SPEC is not None:
    nx = importlib.import_module("networkx")
else:  # pragma: no cover - optional dependency missing
    nx = None  # type: ignore[assignment]


def _require_networkx() -> None:
    if nx is None:
        raise ImportError("Graph metrics require networkx to be installed.")


def compute_path_coherence(graph: AttributionGraph) -> float:
    """Return a coherence score describing attribution concentration."""

    _require_networkx()
    network = graph.to_networkx()
    if network.number_of_nodes() == 0:
        return 0.0

    attributions = np.array(
        [network.nodes[node]["attribution"] for node in network.nodes], dtype=float
    )

    # Normalize attributions to handle negative values
    if attributions.min() < 0:
        attributions = attributions - attributions.min()

    total = attributions.sum()
    if np.isclose(total, 0.0):
        return 0.0

    # Compute Gini coefficient using pairwise differences
    n = len(attributions)
    if n <= 1:
        return 0.0

    # Sort attributions for Gini calculation
    sorted_attrs = np.sort(attributions)

    # Standard Gini formula: G = (2 * sum(i * x[i])) / (n * sum(x)) - (n + 1) / n
    # where i is the rank (1-indexed)
    ranks = np.arange(1, n + 1)
    gini = (2.0 * np.sum(ranks * sorted_attrs)) / (n * total) - (n + 1.0) / n

    return float(np.clip(gini, 0.0, 1.0))


def compute_graph_entropy(graph: AttributionGraph) -> float:
    """Return the Shannon entropy of node attributions."""

    _require_networkx()
    network = graph.to_networkx()
    if network.number_of_nodes() == 0:
        return 0.0

    attrs = np.array([network.nodes[node]["attribution"] for node in network.nodes], dtype=float)
    if np.allclose(attrs, 0.0):
        return 0.0

    safe = np.clip(attrs, 1e-12, None)
    entropy = -np.sum(safe * np.log(safe))
    return float(entropy)


def compute_centrality_concentration(graph: AttributionGraph) -> float:
    """Return the relative dispersion of degree centrality values."""

    _require_networkx()
    network = graph.to_networkx()
    if network.number_of_nodes() == 0:
        return 0.0

    centrality = nx.degree_centrality(network)
    values = np.array(list(centrality.values()))
    if values.size == 0:
        return 0.0

    mean = values.mean()
    if np.isclose(mean, 0.0):
        return 0.0

    variation = values.std() / mean
    return float(np.clip(variation / 2.0, 0.0, 1.0))


def compute_average_path_length(graph: AttributionGraph) -> float:
    """Return the average shortest path length for the largest component."""

    _require_networkx()
    network = graph.to_networkx()
    if network.number_of_nodes() < 2:
        return 0.0

    try:
        # For directed graphs, check strong connectivity
        if nx.is_strongly_connected(network):
            return float(nx.average_shortest_path_length(network))
        # Find largest strongly connected component
        largest = max(nx.strongly_connected_components(network), key=len)
        if len(largest) < 2:
            return 0.0
        subgraph = network.subgraph(largest)
        return float(nx.average_shortest_path_length(subgraph))
    except nx.NetworkXError:
        return 0.0


def compute_all_metrics(graph: AttributionGraph) -> dict[str, float]:
    """Return a dictionary containing all supported metrics."""

    return {
        "path_coherence": compute_path_coherence(graph),
        "entropy": compute_graph_entropy(graph),
        "centrality_concentration": compute_centrality_concentration(graph),
        "average_path_length": compute_average_path_length(graph),
        "num_nodes": float(len(graph.nodes)),
        "num_edges": float(len(graph.edges)),
    }
