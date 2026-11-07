"""Graph evidence is opt-in, exact-site scoped and separate from attribution weights."""

# Standard library
from copy import deepcopy
from dataclasses import replace

# Third-party
import pytest
from test_mechanistic_evidence import context, known_cause, measure, write_authority

# Local
from gepa_mindfulness.interpret.attribution_graphs import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
)
from gepa_mindfulness.interpret.mechanistic_evidence import (
    EvidenceLevel,
    MechanisticRegistry,
    admit_mechanistic_finding,
    verify_causal_runs,
)
from gepa_mindfulness.interpret.mechanistic_graphs import (
    GraphEvidenceBinding,
    attach_graph_evidence,
)


def setup():
    native, plan = known_cause()
    finding = verify_causal_runs((measure(native, plan),), context=context(native), enabled=True)
    before = MechanisticRegistry("graph-evidence", 0, ())
    registry = admit_mechanistic_finding(
        before, finding, context=context(native), enabled=True, **write_authority(before, finding)
    )
    source = AttributionNode(0, "mlp_post", 0, 0, 0.1, 0.2)
    contrast = AttributionNode(-1, "logit_contrast", None, 0, 0.3, 0.4)
    unknown = AttributionNode(0, "mlp_post", 1, 0, 0.2, 0.5)
    graph = AttributionGraph(
        "toy",
        "",
        [source, contrast, unknown],
        [AttributionEdge(source, contrast, 0.2), AttributionEdge(unknown, contrast, 0.1)],
        "gradient_x_activation",
        {
            "model_digest": native.identity.digest,
            "input_digest": plan.recipient.digest,
            "node_hook_digests": {"0": plan.source.hook.digest},
            "contrast_nodes": {"1": plan.metric.to_dict()},
        },
    )
    return graph, registry, finding, native, plan


def test_exact_intervention_annotation_preserves_legacy_graph_and_uncertain_edges() -> None:
    graph, registry, finding, _, _ = setup()
    before = deepcopy(graph)
    result = attach_graph_evidence(
        graph,
        registry,
        (GraphEvidenceBinding(0, finding.finding_id),),
        trusted_registry_digest=registry.digest,
        enabled=True,
    )
    assert graph == before
    assert result.to_networkx().edges[0, 1]["weight"] == 0.2
    overlay = result.metadata["mechanistic_evidence"]
    assert overlay["edges"][0]["level"] == EvidenceLevel.INTERVENTION_SUPPORTED.value
    assert overlay["edges"][1]["level"] == EvidenceLevel.DISCOVERED_CORRELATION.value
    assert overlay["edges"][0]["uncertainty"] is None
    assert overlay["edges"][0]["evidence_refs"]


def test_review_labels_aggregate_units_and_missing_bindings_do_not_upgrade_edges() -> None:
    graph, registry, finding, _, _ = setup()
    graph.metadata["reviewer_accepted"] = True
    graph.nodes[0].component_idx = None
    result = attach_graph_evidence(
        graph,
        registry,
        (GraphEvidenceBinding(0, finding.finding_id),),
        trusted_registry_digest=registry.digest,
        enabled=True,
    )
    assert result.metadata["mechanistic_evidence"]["edges"][0]["level"] == "UNRESOLVED"
    graph.method = "unknown_backend"
    plain = attach_graph_evidence(
        graph, registry, (), trusted_registry_digest=registry.digest, enabled=True
    )
    assert all(e["level"] == "UNRESOLVED" for e in plain.metadata["mechanistic_evidence"]["edges"])


def test_changed_registry_and_foreign_context_cannot_claim_edge_verification() -> None:
    graph, registry, finding, _, _ = setup()
    with pytest.raises(PermissionError):
        attach_graph_evidence(graph, registry, (), trusted_registry_digest="0" * 64, enabled=True)
    graph.metadata["input_digest"] = "0" * 64
    result = attach_graph_evidence(
        graph,
        registry,
        (GraphEvidenceBinding(0, finding.finding_id),),
        trusted_registry_digest=registry.digest,
        enabled=True,
    )
    assert result.metadata["mechanistic_evidence"]["edges"][0]["level"] == "UNRESOLVED"
    with pytest.raises(ValueError, match="enabled=True"):
        attach_graph_evidence(None, None, (), trusted_registry_digest=None)


def test_patch_support_cannot_transfer_to_the_unmeasured_donor_context() -> None:
    graph, _, _, native, plan = setup()
    plan = replace(plan, donor=replace(plan.donor, token_ids=(2, 2, 3)))
    report = measure(native, plan)
    assert report["prediction_result"] == "MET"
    assert report["conditions"]["ablation"]["effect"] == 0.0
    finding = verify_causal_runs((report,), context=context(native), enabled=True)
    before = MechanisticRegistry("different-context", 0, ())
    registry = admit_mechanistic_finding(
        before,
        finding,
        context=context(native),
        enabled=True,
        **write_authority(before, finding),
    )
    graph.metadata["input_digest"] = plan.donor.digest
    result = attach_graph_evidence(
        graph,
        registry,
        (GraphEvidenceBinding(0, finding.finding_id),),
        trusted_registry_digest=registry.digest,
        enabled=True,
    )
    assert result.metadata["mechanistic_evidence"]["edges"][0]["level"] == "UNRESOLVED"


def test_contradicting_evidence_is_not_hidden_by_binding_an_older_positive() -> None:
    graph, registry, finding, native, plan = setup()
    failed_plan = replace(
        plan,
        experiment_id="counterexample",
        donor=replace(plan.donor, token_ids=(4, 2, 3)),
        recipient=replace(plan.recipient, token_ids=(1, 2, 3)),
    )
    counter = verify_causal_runs(
        (measure(native, failed_plan),), context=context(native), enabled=True
    )
    assert counter.hypothesis_key == finding.hypothesis_key
    assert counter.level is EvidenceLevel.CONTRADICTED
    registry = admit_mechanistic_finding(
        registry,
        counter,
        context=context(native),
        enabled=True,
        **write_authority(registry, counter),
    )
    result = attach_graph_evidence(
        graph,
        registry,
        (GraphEvidenceBinding(0, finding.finding_id),),
        trusted_registry_digest=registry.digest,
        enabled=True,
    )
    edge = result.metadata["mechanistic_evidence"]["edges"][0]
    assert edge["level"] == "CONTRADICTED"
    assert len(edge["finding_ids"]) == 2
