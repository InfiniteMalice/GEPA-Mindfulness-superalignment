"""Graph metrics computed on attribution graphs."""

from __future__ import annotations

from typing import Dict

try:  # pragma: no cover - optional dependency
    import networkx as nx
except ImportError:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - executed when numpy missing
    np = None  # type: ignore[assignment]

from gepa_mindfulness.interpret.attribution_graphs import AttributionGraph


def compute_path_coherence(graph: AttributionGraph) -> float:
    """Return a coherence score describing attribution concentration."""

    _require_dependency("networkx", nx is not None)
    _require_dependency("NumPy", np is not None)

    network = graph.to_networkx()
    if network.number_of_nodes() == 0:
        return 0.0

    attributions = [network.nodes[node]["attribution"] for node in network.nodes]
    sorted_attrs = np.sort(np.array(attributions))
    total = sorted_attrs.sum()
    if np.isclose(total, 0.0):
        return 0.0

    cumulative = np.cumsum(sorted_attrs)
    count = float(len(sorted_attrs))
    gini = (count + 1.0 - 2.0 * cumulative.sum() / total) / count
    return float(np.clip(gini, 0.0, 1.0))


def compute_graph_entropy(graph: AttributionGraph) -> float:
    """Return the Shannon entropy of node attributions."""

    _require_dependency("networkx", nx is not None)
    _require_dependency("NumPy", np is not None)

    network = graph.to_networkx()
    if network.number_of_nodes() == 0:
        return 0.0

    attrs = np.array([network.nodes[node]["attribution"] for node in network.nodes])
    total = attrs.sum()
    if np.isclose(total, 0.0):
        return 0.0

    probs = attrs / total
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)))
    return float(entropy)


def compute_centrality_concentration(graph: AttributionGraph) -> float:
    """Return the relative dispersion of degree centrality values."""

    _require_dependency("networkx", nx is not None)
    _require_dependency("NumPy", np is not None)

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

    _require_dependency("networkx", nx is not None)

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

    _require_dependency("networkx", nx is not None)
    _require_dependency("NumPy", np is not None)

    return {
        "path_coherence": compute_path_coherence(graph),
        "entropy": compute_graph_entropy(graph),
        "centrality_concentration": compute_centrality_concentration(graph),
        "average_path_length": compute_average_path_length(graph),
        "num_nodes": float(len(graph.nodes)),
        "num_edges": float(len(graph.edges)),
    }


def _require_dependency(name: str, available: bool) -> None:
    """Raise a helpful error for optional dependency usage."""

    if not available:
        raise RuntimeError(
            f"{name} is required for graph metrics. "
            "Install the optional analytics extras to enable this feature."
        )
