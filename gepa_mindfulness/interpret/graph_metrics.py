"""Graph metrics computed on attribution graphs."""

from __future__ import annotations

from typing import Dict

import networkx as nx
import numpy as np

from gepa_mindfulness.interpret.attribution_graphs import AttributionGraph


def compute_path_coherence(graph: AttributionGraph) -> float:
    """Return a coherence score describing attribution concentration."""

    network = graph.to_networkx()
    if network.number_of_nodes() == 0:
        return 0.0

    attributions = np.array(
        [network.nodes[node]["attribution"] for node in network.nodes],
        dtype=float,
    )
    magnitudes = np.abs(attributions)
    total = magnitudes.sum()
    if np.isclose(total, 0.0):
        return 0.0

    sorted_attrs = np.sort(magnitudes)
    size = sorted_attrs.size
    if size == 0:
        return 0.0

    count = float(size)
    index = np.arange(1, size + 1, dtype=float)
    weights = 2.0 * index - count - 1.0
    gini = np.sum(weights * sorted_attrs) / (count * total)
    max_gini = (count - 1.0) / count if count > 1.0 else 0.0
    if np.isclose(max_gini, 0.0):
        return 0.0

    coherence = gini / max_gini
    return float(np.clip(coherence, 0.0, 1.0))


def compute_graph_entropy(graph: AttributionGraph) -> float:
    """Return the Shannon entropy of node attributions."""

    network = graph.to_networkx()
    if network.number_of_nodes() == 0:
        return 0.0

    attrs = np.array(
        [network.nodes[node]["attribution"] for node in network.nodes],
        dtype=float,
    )
    magnitudes = np.abs(attrs)
    total = magnitudes.sum()
    if np.isclose(total, 0.0):
        return 0.0

    probs = magnitudes / total
    safe = np.clip(probs, 1e-12, None)
    entropy = -np.sum(safe * np.log(safe))
    return float(entropy)


def compute_centrality_concentration(graph: AttributionGraph) -> float:
    """Return the relative dispersion of degree centrality values."""

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

    network = graph.to_networkx()
    if network.number_of_nodes() < 2:
        return 0.0

    try:
        if nx.is_weakly_connected(network):
            return float(nx.average_shortest_path_length(network))
        largest = max(nx.weakly_connected_components(network), key=len)
        subgraph = network.subgraph(largest)
        return float(nx.average_shortest_path_length(subgraph))
    except nx.NetworkXError:
        return 0.0


def compute_all_metrics(graph: AttributionGraph) -> Dict[str, float]:
    """Return a dictionary containing all supported metrics."""

    return {
        "path_coherence": compute_path_coherence(graph),
        "entropy": compute_graph_entropy(graph),
        "centrality_concentration": compute_centrality_concentration(graph),
        "average_path_length": compute_average_path_length(graph),
        "num_nodes": float(len(graph.nodes)),
        "num_edges": float(len(graph.edges)),
    }
