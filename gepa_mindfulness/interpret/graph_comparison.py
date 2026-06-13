"""Utilities for comparing attribution graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gepa_mindfulness.interpret.attribution_graphs import AttributionGraph
from gepa_mindfulness.interpret.graph_metrics import compute_all_metrics
from gepa_mindfulness.interpret.networkx_compat import nx

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx


def _require_networkx() -> None:
    return None


def compare_graphs(graph_a: AttributionGraph, graph_b: AttributionGraph) -> dict[str, float]:
    """Return similarity scores between two attribution graphs."""

    _require_networkx()
    net_a = graph_a.to_networkx()
    net_b = graph_b.to_networkx()
    structural = compute_structural_similarity(net_a, net_b)
    attribution = compute_attribution_similarity(net_a, net_b)
    metrics_a = compute_all_metrics(graph_a)
    metrics_b = compute_all_metrics(graph_b)
    metric_similarity = compute_metric_similarity(metrics_a, metrics_b)
    overall = (structural + attribution + metric_similarity) / 3.0
    return {
        "structural_similarity": structural,
        "attribution_similarity": attribution,
        "metric_similarity": metric_similarity,
        "overall_similarity": float(np.clip(overall, 0.0, 1.0)),
    }


def compute_structural_similarity(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> float:
    """Return a spectral similarity score between two graphs."""

    _require_networkx()
    if graph_a.number_of_nodes() == 0 or graph_b.number_of_nodes() == 0:
        return 0.0

    try:
        eig_a = np.sort(_laplacian_spectrum(graph_a))
        eig_b = np.sort(_laplacian_spectrum(graph_b))
    except nx.NetworkXError:
        return 0.0

    max_len = max(len(eig_a), len(eig_b))
    eig_a = np.pad(eig_a, (0, max_len - len(eig_a)))
    eig_b = np.pad(eig_b, (0, max_len - len(eig_b)))

    if np.allclose(eig_a, 0.0) or np.allclose(eig_b, 0.0):
        return 0.0

    similarity = _cosine_similarity(eig_a, eig_b)
    return float(np.clip(similarity, 0.0, 1.0))


def compute_attribution_similarity(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> float:
    """Return similarity of attribution histograms."""

    _require_networkx()
    if graph_a.number_of_nodes() == 0 or graph_b.number_of_nodes() == 0:
        return 0.0

    attrs_a = _normalised_attribution_masses(graph_a)
    attrs_b = _normalised_attribution_masses(graph_b)
    if attrs_a.size == 0 or attrs_b.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, 11)
    hist_a, _ = np.histogram(attrs_a, bins=bins, weights=attrs_a)
    hist_b, _ = np.histogram(attrs_b, bins=bins, weights=attrs_b)
    distance = np.abs(hist_a - hist_b).sum() / 2.0
    similarity = 1.0 - distance
    if not np.isfinite(similarity):
        return 0.0
    return float(np.clip(similarity, 0.0, 1.0))


def compute_metric_similarity(metrics_a: dict[str, float], metrics_b: dict[str, float]) -> float:
    """Return cosine similarity between the metric vectors."""

    shared = sorted(set(metrics_a) & set(metrics_b))
    if not shared:
        return 0.0

    vec_a = np.array([metrics_a[key] for key in shared])
    vec_b = np.array([metrics_b[key] for key in shared])
    if np.allclose(vec_a, 0.0) or np.allclose(vec_b, 0.0):
        return 0.0

    similarity = _cosine_similarity(vec_a, vec_b)
    return float(np.clip(similarity, 0.0, 1.0))


def find_distinctive_subgraphs(
    honest_graphs: list[AttributionGraph],
    deceptive_graphs: list[AttributionGraph],
    *,
    min_frequency: float = 0.3,
) -> dict[str, list[nx.DiGraph]]:
    """Distinctive subgraph mining is not yet a supported public API."""

    _ = (honest_graphs, deceptive_graphs, min_frequency)
    raise NotImplementedError(
        "find_distinctive_subgraphs is not implemented. "
        "Use graph-level comparison metrics until a verified miner is available."
    )


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity, guarding against zero vectors."""

    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _laplacian_spectrum(graph: nx.DiGraph) -> np.ndarray:
    nodes = list(graph.nodes)
    size = len(nodes)
    if size == 0:
        return np.zeros(0)
    index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = np.zeros((size, size), dtype=float)
    try:
        edges = graph.edges(data=True)
    except TypeError:
        edges = graph.edges()
    for source, target, attrs in edges:
        weight = float(attrs.get("weight", 1.0))
        adjacency[index[source], index[target]] += weight
    adjacency = (adjacency + adjacency.T) / 2.0
    degree = np.diag(adjacency.sum(axis=1))
    return np.linalg.eigvalsh(degree - adjacency)


def _normalised_attribution_masses(graph: nx.DiGraph) -> np.ndarray:
    attrs = np.abs(np.array([graph.nodes[node]["attribution"] for node in graph.nodes], dtype=float))
    total = attrs.sum()
    if np.isclose(total, 0.0):
        return np.zeros(0)
    return attrs / total
