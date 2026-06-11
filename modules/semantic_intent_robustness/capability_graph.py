"""Behavioral capability disclosure graph for multi-source fragment aggregation."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Local
from .disclosure_events import DisclosureEvent


@dataclass(frozen=True)
class CapabilityFragment:
    """A redacted capability fragment from one or more disclosure sources."""

    fragment_id: str
    capability_domain: str
    capability_role: str
    operational_specificity: float
    executionality: float
    confidence: float
    source_event_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...] = field(default_factory=tuple)
    enables_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityDependency:
    """Directed dependency between capability fragments."""

    source_fragment_id: str
    target_fragment_id: str
    dependency_type: str
    strength: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafeRedactionRecommendation:
    """Minimal redaction recommendation for closure-producing fragments."""

    minimum_safe_redaction_possible: bool
    fragment_ids: tuple[str, ...]
    redaction_summary: str


@dataclass(frozen=True)
class CapabilityDisclosureGraph:
    """Disclosure graph that aggregates fragments across source types."""

    fragments: tuple[CapabilityFragment, ...]
    dependencies: tuple[CapabilityDependency, ...]
    disclosure_events: tuple[DisclosureEvent, ...]
    graph_version: str = "1.0"

    def nodes(self) -> tuple[CapabilityFragment, ...]:
        return self.fragments

    def edges(self) -> tuple[CapabilityDependency, ...]:
        return self.dependencies

    def paths_to(self, target_id: str, max_depth: int = 3) -> tuple[tuple[str, ...], ...]:
        if max_depth < 1:
            return ()

        def walk(current_id: str, depth_remaining: int, visited: set[str]) -> list[tuple[str, ...]]:
            paths: list[tuple[str, ...]] = []
            for dep in self.dependencies:
                if dep.target_fragment_id != current_id:
                    continue
                source_id = dep.source_fragment_id
                if source_id in visited:
                    continue
                paths.append((source_id, current_id))
                if depth_remaining > 1:
                    for prefix in walk(source_id, depth_remaining - 1, visited | {source_id}):
                        paths.append(prefix + (current_id,))
            return paths

        return tuple(walk(target_id, max_depth, {target_id}))


def build_capability_disclosure_graph(
    disclosure_events: tuple[DisclosureEvent, ...] | list[DisclosureEvent],
) -> CapabilityDisclosureGraph:
    """Build a deterministic graph from redacted disclosure events."""

    events = tuple(disclosure_events)
    fragment_map: dict[str, CapabilityFragment] = {}
    ordered_fragment_ids: list[str] = []
    for event in events:
        for fragment_id in event.fragment_ids:
            if fragment_id not in ordered_fragment_ids:
                ordered_fragment_ids.append(fragment_id)
            existing = fragment_map.get(fragment_id)
            source_ids = (existing.source_event_ids if existing else ()) + (event.event_id,)
            new_confidence = max(0.5, event.operational_specificity, event.executionality)
            fragment_map[fragment_id] = CapabilityFragment(
                fragment_id=fragment_id,
                capability_domain=event.capability_domain,
                capability_role=_role(fragment_id),
                operational_specificity=max(
                    event.operational_specificity,
                    existing.operational_specificity if existing else 0.0,
                ),
                executionality=max(
                    event.executionality,
                    existing.executionality if existing else 0.0,
                ),
                confidence=max(existing.confidence if existing else 0.0, new_confidence),
                source_event_ids=tuple(dict.fromkeys(source_ids)),
            )
    dependencies = []
    for source, target in zip(ordered_fragment_ids, ordered_fragment_ids[1:]):
        dependencies.append(
            CapabilityDependency(
                source_fragment_id=source,
                target_fragment_id=target,
                dependency_type="enables",
                strength=0.7,
            )
        )
    return CapabilityDisclosureGraph(
        fragments=tuple(fragment_map[fragment_id] for fragment_id in ordered_fragment_ids),
        dependencies=tuple(dependencies),
        disclosure_events=events,
    )


def merge_disclosure_event(
    graph: CapabilityDisclosureGraph,
    event: DisclosureEvent,
) -> CapabilityDisclosureGraph:
    """Merge one disclosure event into an existing graph."""

    return build_capability_disclosure_graph(graph.disclosure_events + (event,))


def estimate_capability_completion(graph: CapabilityDisclosureGraph) -> float:
    """Estimate whether fragments cover component, parameter, and integration roles."""

    roles = {fragment.capability_role for fragment in graph.fragments}
    role_score = (
        sum(role in roles for role in ("component", "parameter", "integration", "sequence")) / 4
    )
    source_diversity = len({event.source_type for event in graph.disclosure_events})
    diversity_score = min(1.0, source_diversity / 3)
    dependency_score = min(1.0, len(graph.dependencies) / 3)
    return min(1.0, 0.55 * role_score + 0.25 * diversity_score + 0.20 * dependency_score)


def detect_harm_enabling_closure(graph: CapabilityDisclosureGraph) -> bool:
    """Detect likely closure from accumulated abstract fragments."""

    return estimate_capability_completion(graph) >= 0.6


def find_earliest_closure_point(graph: CapabilityDisclosureGraph) -> int | None:
    """Return the first turn where the graph reaches closure, if any."""

    partial_events: list[DisclosureEvent] = []
    for event in sorted(graph.disclosure_events, key=lambda item: item.turn_index):
        partial_events.append(event)
        if detect_harm_enabling_closure(build_capability_disclosure_graph(tuple(partial_events))):
            return event.turn_index
    return None


def recommend_minimum_safe_redaction(
    graph: CapabilityDisclosureGraph,
) -> SafeRedactionRecommendation:
    """Prefer redacting closure-producing fragments over full refusal when possible."""

    closure_roles = {"integration", "concealment", "sequence"}
    fragment_ids = tuple(
        fragment.fragment_id
        for fragment in graph.fragments
        if fragment.capability_role in closure_roles
    )
    if not fragment_ids:
        return SafeRedactionRecommendation(False, (), "No bounded redaction target found.")
    return SafeRedactionRecommendation(
        minimum_safe_redaction_possible=True,
        fragment_ids=fragment_ids,
        redaction_summary="Redact " + ", ".join(fragment_ids),
    )


def _role(fragment_id: str) -> str:
    if fragment_id.startswith("COMPONENT_"):
        return "component"
    if fragment_id.startswith("PARAMETER_"):
        return "parameter"
    if fragment_id.startswith("INTEGRATION_"):
        return "integration"
    if fragment_id.startswith("SEQUENCE_") or fragment_id.startswith("STEP_"):
        return "sequence"
    if fragment_id.startswith("CONCEALMENT_"):
        return "concealment"
    return "context"


__all__ = [
    "CapabilityDependency",
    "CapabilityDisclosureGraph",
    "CapabilityFragment",
    "SafeRedactionRecommendation",
    "build_capability_disclosure_graph",
    "detect_harm_enabling_closure",
    "estimate_capability_completion",
    "find_earliest_closure_point",
    "merge_disclosure_event",
    "recommend_minimum_safe_redaction",
]
