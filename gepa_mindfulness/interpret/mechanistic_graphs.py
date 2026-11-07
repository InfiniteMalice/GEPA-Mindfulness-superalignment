"""Opt-in edge evidence overlays; attribution weights and legacy graph APIs are unchanged."""

# Standard library
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.verification.cva_records import CVARecord, _record, array, digest, text

from .attribution_graphs import AttributionGraph, AttributionNode
from .mechanistic_evidence import EvidenceLevel, MechanisticFinding, MechanisticRegistry
from .native_causal_records import CausalCommit, InterventionMode, InterventionSite
from .native_records import finite, index


@dataclass(frozen=True)
class GraphEvidenceBinding(CVARecord):
    edge_index: int
    finding_id: str
    uncertainty: float | None = None

    schema_version = "mechanistic-graph-binding-v1"

    def __post_init__(self) -> None:
        index(self.edge_index, maximum=1024)
        digest(self.finding_id)
        if self.uncertainty is not None:
            finite(self.uncertainty)
            if not 0 <= self.uncertainty <= 1:
                raise ValueError("declared uncertainty must be in [0,1] or unknown")


def graph_evidence_digest(graph: AttributionGraph) -> str:
    """Digest exact graph data and mapping metadata, excluding only this overlay itself."""
    if (
        type(graph) is not AttributionGraph
        or type(graph.nodes) is not list
        or type(graph.edges) is not list
    ):
        raise ValueError("expected exact attribution graph")
    if len(graph.nodes) > 512 or len(graph.edges) > 1024:
        raise ValueError("graph exceeds experimental overlay budget")
    nodes = []
    for n in graph.nodes:
        if type(n) is not AttributionNode:
            raise ValueError("expected exact attribution nodes")
        for value in (n.activation_value, n.attribution_score):
            finite(value)
        if type(n.layer) is not int or type(n.token_position) is not int:
            raise ValueError("graph coordinates require exact integers")
        text(n.component_type, "component_type")
        if n.component_idx is not None:
            index(n.component_idx)
        nodes.append(
            {
                "layer": n.layer,
                "kind": n.component_type,
                "component": n.component_idx,
                "position": n.token_position,
                "activation": n.activation_value,
                "attribution": n.attribution_score,
            }
        )
    identities = {id(n): i for i, n in enumerate(graph.nodes)}
    edges = []
    for e in graph.edges:
        finite(e.attribution_weight)
        edges.append(
            {
                "source": identities.get(id(e.source_node)),
                "target": identities.get(id(e.target_node)),
                "weight": e.attribution_weight,
            }
        )
    metadata = dict(graph.metadata)
    metadata.pop("mechanistic_evidence", None)
    body = {
        "prompt": graph.prompt,
        "response": graph.response,
        "method": graph.method,
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }
    validate_input_envelope(body)
    return content_digest(body)


def _site_matches(graph: AttributionGraph, node_index: int, site: InterventionSite) -> bool:
    node = graph.nodes[node_index]
    group_digests = graph.metadata.get("node_site_digests", {})
    if (
        node.component_type == "intervention_group"
        and type(group_digests) is dict
        and group_digests.get(str(node_index)) == site.digest
    ):
        return node.layer == site.hook.layer
    # A grouped intervention is not evidence for every member's individual edge.
    if len(site.component_indices) != 1 or len(site.token_positions) != 1:
        return False
    hooks = graph.metadata.get("node_hook_digests", {})
    return (
        type(hooks) is dict
        and hooks.get(str(node_index)) == site.hook.digest
        and node.layer == site.hook.layer
        and node.component_type == site.hook.kind.value
        and node.component_idx == site.component_indices[0]
        and node.token_position == site.token_positions[0]
    )


def _edge_matches(graph: AttributionGraph, edge_index: int, finding: MechanisticFinding) -> bool:
    if finding.source_kind != "causal":
        return False
    edge = graph.edges[edge_index]
    identities = {id(n): i for i, n in enumerate(graph.nodes)}
    source = identities.get(id(edge.source_node))
    target = identities.get(id(edge.target_node))
    if source is None or target is None:
        return False
    for raw in finding.source_reports_json:
        report = json.loads(raw)
        commit = CausalCommit.from_dict(report["commit"])
        plan = commit.plan
        if graph.metadata.get("model_digest") != plan.model_digest:
            continue
        primary_context = (
            plan.donor
            if plan.mode in (InterventionMode.ABLATION, InterventionMode.RESTORATION)
            else plan.recipient
        )
        if graph.metadata.get("input_digest") != primary_context.digest:
            continue
        if not _site_matches(graph, source, plan.source):
            continue
        if plan.mediator is not None:
            if _site_matches(graph, target, plan.mediator):
                return True
        else:
            node = graph.nodes[target]
            contrasts = graph.metadata.get("contrast_nodes", {})
            if (
                node.component_type == "logit_contrast"
                and node.component_idx is None
                and node.token_position == plan.metric.position
                and type(contrasts) is dict
                and contrasts.get(str(target)) == plan.metric.to_dict()
            ):
                return True
    return False


def attach_graph_evidence(
    graph: AttributionGraph,
    registry: MechanisticRegistry,
    bindings: tuple[GraphEvidenceBinding, ...],
    *,
    trusted_registry_digest: str,
    enabled: bool = False,
) -> AttributionGraph:
    """Attach host-pinned registry data without changing weights or treating labels as grants.

    The expected digest is trusted host configuration from the protected persisted snapshot,
    never an actor-supplied copy. Serialized overlays/registries are not bearer credentials.
    Unverified graph mapping, aggregate ambiguity and unknown backends remain unresolved.
    """
    if enabled is not True:
        raise ValueError("mechanistic graph overlays require enabled=True")
    state = _record(registry, MechanisticRegistry)
    digest(trusted_registry_digest)
    if state.digest != trusted_registry_digest:
        raise PermissionError("registry does not match host-pinned accepted evidence")
    fingerprint = graph_evidence_digest(graph)
    snapshot = copy.deepcopy(graph)
    proposed = tuple(_record(b, GraphEvidenceBinding) for b in array(bindings, 1024))
    if len({b.edge_index for b in proposed}) != len(proposed):
        raise ValueError("duplicate edge bindings")
    if any(b.edge_index >= len(snapshot.edges) for b in proposed):
        raise ValueError("edge binding outside graph")
    indexed = {e.finding_id: e for e in state.entries}
    if any(b.finding_id not in indexed for b in proposed):
        raise ValueError("edge binding refers to unknown accepted evidence")
    by_edge = {b.edge_index: b for b in proposed}
    rows = []
    for i, _ in enumerate(snapshot.edges):
        level = (
            EvidenceLevel.DISCOVERED_CORRELATION
            if snapshot.method in ("gradient_x_activation", "native_activation_observation")
            else EvidenceLevel.UNRESOLVED
        )
        uncertainty = None
        refs: dict[str, Any] = {}
        finding_ids = []
        reasons = ["Attribution weights are discovery estimates; causal role remains unverified."]
        if i in by_edge:
            binding = by_edge[i]
            finding = indexed[binding.finding_id]
            relevant = [e for e in state.entries if e.hypothesis_key == finding.hypothesis_key]
            finding_ids = [e.finding_id for e in relevant]
            uncertainty = binding.uncertainty
            for e in relevant:
                for ref in e.evidence_refs:
                    refs[ref.reference_id] = ref.to_dict()
            if not _edge_matches(snapshot, i, finding):
                level = EvidenceLevel.UNRESOLVED
                reasons = [
                    "Graph model/input/native-site mapping is missing, ambiguous or foreign."
                ]
            else:
                level = finding.level
                if any(e.level is EvidenceLevel.CONTRADICTED for e in relevant):
                    level = EvidenceLevel.CONTRADICTED
                reasons = [
                    "Support concerns the exact intervention under registered assumptions.",
                    "Related findings are retained, including counterexamples to the prediction.",
                    "Attribution weight is not a causal-strength estimate or evidence of intent.",
                ]
        rows.append(
            {
                "edge_index": i,
                "level": level.value,
                "uncertainty": uncertainty,
                "finding_ids": finding_ids,
                "evidence_refs": list(refs.values()),
                "remaining_uncertainty": reasons,
            }
        )
    snapshot.metadata["mechanistic_evidence"] = {
        "schema_version": "mechanistic-graph-overlay-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "graph_digest": fingerprint,
        "registry_digest": state.digest,
        "edges": rows,
    }
    validate_input_envelope(snapshot.metadata["mechanistic_evidence"])
    if graph_evidence_digest(snapshot) != fingerprint:
        raise ValueError("graph changed while constructing evidence overlay")
    return snapshot
