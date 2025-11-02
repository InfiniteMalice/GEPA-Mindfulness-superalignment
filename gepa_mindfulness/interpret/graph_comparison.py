"""Utilities for comparing attribution graphs."""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np

from gepa_mindfulness.interpret.attribution_graphs import AttributionGraph
from gepa_mindfulness.interpret.graph_metrics import compute_all_metrics


def compare_graphs(graph_a: AttributionGraph, graph_b: AttributionGraph) -> Dict[str, float]:
    """Return similarity scores between two attribution graphs."""

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

    if graph_a.number_of_nodes() == 0 or graph_b.number_of_nodes() == 0:
        return 0.0

    try:
        eig_a = np.sort(nx.laplacian_spectrum(graph_a))
        eig_b = np.sort(nx.laplacian_spectrum(graph_b))
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

    if graph_a.number_of_nodes() == 0 or graph_b.number_of_nodes() == 0:
        return 0.0

    attrs_a = np.array([graph_a.nodes[node]["attribution"] for node in graph_a.nodes], dtype=float)
    attrs_b = np.array([graph_b.nodes[node]["attribution"] for node in graph_b.nodes], dtype=float)

    values = np.concatenate((attrs_a, attrs_b))
    min_val = float(values.min())
    max_val = float(values.max())
    if np.isclose(min_val, max_val):
        span = max(abs(min_val), 1.0)
        min_val -= span * 0.5
        max_val += span * 0.5
    bins = np.linspace(min_val, max_val, 11)
    hist_a, _ = np.histogram(attrs_a, bins=bins, density=True)
    hist_b, _ = np.histogram(attrs_b, bins=bins, density=True)
    distance = np.abs(hist_a - hist_b).sum() / 2.0
    similarity = 1.0 - distance
    return float(np.clip(similarity, 0.0, 1.0))


def compute_metric_similarity(metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> float:
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
    honest_graphs: List[AttributionGraph],
    deceptive_graphs: List[AttributionGraph],
    *,
    min_frequency: float = 0.3,
) -> Dict[str, List[nx.DiGraph]]:
    """Return placeholder results for distinctive subgraph mining."""

    _ = (honest_graphs, deceptive_graphs, min_frequency)
    return {"honest_patterns": [], "deceptive_patterns": []}


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity, guarding against zero vectors."""

    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)
