"""Minimal subset of NetworkX used for attribution graph comparisons."""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, Iterable, Iterator, Set, Tuple

import numpy as np


class NetworkXError(Exception):
    """Exception matching the real library's error type."""


class _NodeView:
    """Dictionary-style accessor for node attributes."""

    def __init__(self, nodes: Dict[Hashable, Dict[str, float]]) -> None:
        self._nodes = nodes

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._nodes)

    def __getitem__(self, key: Hashable) -> Dict[str, float]:
        return self._nodes[key]

    def items(self) -> Iterable[Tuple[Hashable, Dict[str, float]]]:
        return self._nodes.items()

    def __len__(self) -> int:
        return len(self._nodes)


class DiGraph:
    """Very small directed graph implementation."""

    def __init__(self) -> None:
        self._nodes: Dict[Hashable, Dict[str, float]] = {}
        self._adj: Dict[Hashable, Dict[Hashable, Dict[str, float]]] = {}

    def add_node(self, node: Hashable, **attrs: float) -> None:
        self._nodes.setdefault(node, {}).update(attrs)
        self._adj.setdefault(node, {})

    def add_edge(self, source: Hashable, target: Hashable, **attrs: float) -> None:
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)
        self._adj[source][target] = attrs

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> _NodeView:
        return _NodeView(self._nodes)

    def edges(self) -> Iterator[Tuple[Hashable, Hashable, Dict[str, float]]]:
        for source, targets in self._adj.items():
            for target, attrs in targets.items():
                yield source, target, attrs

    def out_degree(self, node: Hashable) -> int:
        return len(self._adj.get(node, {}))

    def in_degree(self, node: Hashable) -> int:
        count = 0
        for targets in self._adj.values():
            if node in targets:
                count += 1
        return count

    def neighbors(self, node: Hashable) -> Iterator[Hashable]:
        for target in self._adj.get(node, {}):
            yield target

    def predecessors(self, node: Hashable) -> Iterator[Hashable]:
        for source, targets in self._adj.items():
            if node in targets:
                yield source

    def subgraph(self, nodes: Iterable[Hashable]) -> "DiGraph":
        subset = set(nodes)
        new_graph = DiGraph()
        for node in subset:
            if node in self._nodes:
                new_graph.add_node(node, **self._nodes[node])
        for source, target, attrs in self.edges():
            if source in subset and target in subset:
                new_graph.add_edge(source, target, **attrs)
        return new_graph


def laplacian_spectrum(graph: DiGraph) -> np.ndarray:
    """Compute eigenvalues of the symmetrised Laplacian matrix."""

    nodes = list(graph.nodes)
    size = len(nodes)
    if size == 0:
        return np.zeros(0)
    index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = np.zeros((size, size), dtype=float)
    for source, target, attrs in graph.edges():
        weight = float(attrs.get("weight", 1.0))
        i = index[source]
        j = index[target]
        adjacency[i, j] += weight
    # Symmetrise to keep the Laplacian positive semi-definite.
    adjacency = (adjacency + adjacency.T) / 2.0
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    try:
        return np.linalg.eigvalsh(laplacian)
    except np.linalg.LinAlgError as exc:
        raise NetworkXError("Unable to compute Laplacian spectrum") from exc


def degree_centrality(graph: DiGraph) -> Dict[Hashable, float]:
    """Return a simple degree centrality mapping."""

    size = graph.number_of_nodes()
    if size <= 1:
        return {node: 0.0 for node in graph.nodes}
    denom = size - 1
    result: Dict[Hashable, float] = {}
    for node in graph.nodes:
        degree = graph.out_degree(node) + graph.in_degree(node)
        result[node] = degree / denom
    return result


def is_weakly_connected(graph: DiGraph) -> bool:
    """Return True if the underlying undirected graph is connected."""

    nodes = list(graph.nodes)
    if not nodes:
        return True
    visited = _bfs(graph, nodes[0])
    return len(visited) == len(nodes)


def weakly_connected_components(graph: DiGraph) -> Iterator[Set[Hashable]]:
    """Yield node sets for each weakly connected component."""

    remaining = set(graph.nodes)
    while remaining:
        node = next(iter(remaining))
        component = _bfs(graph, node)
        yield component
        remaining -= component


def average_shortest_path_length(graph: DiGraph) -> float:
    """Compute the average shortest path length treating edges as unweighted."""

    nodes = list(graph.nodes)
    size = len(nodes)
    if size < 2:
        raise NetworkXError("Graph must contain at least two nodes")

    total = 0.0
    count = 0
    for source in nodes:
        distances = _bfs_distances(graph, source)
        if len(distances) != size:
            raise NetworkXError("Graph is not connected")
        for target, distance in distances.items():
            if target == source:
                continue
            total += distance
            count += 1
    if count == 0:
        raise NetworkXError("No paths found")
    return total / count


def _bfs(graph: DiGraph, start: Hashable) -> Set[Hashable]:
    queue: deque[Hashable] = deque([start])
    visited: Set[Hashable] = {start}
    while queue:
        node = queue.popleft()
        for neighbor in _undirected_neighbors(graph, node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def _bfs_distances(graph: DiGraph, start: Hashable) -> Dict[Hashable, float]:
    queue: deque[Hashable] = deque([start])
    distances: Dict[Hashable, float] = {start: 0.0}
    while queue:
        node = queue.popleft()
        current = distances[node]
        for neighbor in graph.neighbors(node):
            if neighbor not in distances:
                distances[neighbor] = current + 1.0
                queue.append(neighbor)
    return distances


def _undirected_neighbors(graph: DiGraph, node: Hashable) -> Set[Hashable]:
    neighbors: Set[Hashable] = set(graph.neighbors(node))
    neighbors.update(graph.predecessors(node))
    return neighbors


__all__ = [
    "DiGraph",
    "NetworkXError",
    "average_shortest_path_length",
    "degree_centrality",
    "is_weakly_connected",
    "laplacian_spectrum",
    "weakly_connected_components",
]
