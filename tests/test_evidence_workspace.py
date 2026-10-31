"""Workspace navigation retains the existing evidence and current-access boundaries."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest

# Local
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from gepa_mindfulness.verification import evidence_workspace as module
from gepa_mindfulness.verification.artifact_records import artifact_digest, payload_digest
from gepa_mindfulness.verification.evidence_workspace import (
    EvidenceWorkspace,
    WorkspaceNode,
    prepare_evidence_workspace,
    retrieve_evidence_workspace,
)
from gepa_mindfulness.verification.state import EvidenceState

# isort: split
from artifact_fixtures import make_derived_snapshot, make_query, make_topology


def workspace_fixture():
    snapshot = make_derived_snapshot()
    nodes = (
        WorkspaceNode("private-summary-label", "summary"),
        WorkspaceNode("private-a-label", "a", "private-summary-label"),
        WorkspaceNode("private-b-label", "b", "private-summary-label"),
    )
    refs = snapshot.sources[0].claim.evidence_refs
    workspace = prepare_evidence_workspace(snapshot, nodes, refs, enabled=True)
    return snapshot, workspace, make_topology(snapshot, "synthesis"), make_query(snapshot)


def test_disabled_before_inputs_or_callbacks():
    with pytest.raises(ValueError, match="enabled=True"):
        prepare_evidence_workspace(None, None, None)
    with pytest.raises(ValueError, match="enabled=True"):
        retrieve_evidence_workspace(None, None, None, None)


def test_round_trip_and_no_optimizer_eligibility():
    snapshot, workspace, _, _ = workspace_fixture()
    encoded = workspace.to_dict()
    assert EvidenceWorkspace.from_dict(json.loads(json.dumps(encoded))) == workspace
    assert encoded["schema_version"] == "evidence-workspace-v1"
    assert encoded["training_eligibility"] == "DEVELOPMENT"
    assert workspace.snapshot_digest == artifact_digest(snapshot)
    for value in (workspace, encoded):
        with pytest.raises(ValueError):
            require_training_eligible(value)
    encoded["nodes"][0]["item_id"] = "a"
    assert workspace.nodes[0].item_id == "summary"
    with pytest.raises(ValueError):
        EvidenceWorkspace.from_dict(encoded)


@pytest.mark.parametrize(
    "nodes",
    [
        (WorkspaceNode("a", "a"),),
        (WorkspaceNode("a", "a"), WorkspaceNode("b", "b"), WorkspaceNode("x", "unknown")),
        (WorkspaceNode("a", "a"), WorkspaceNode("b", "b", "missing")),
        (WorkspaceNode("a", "a"), WorkspaceNode("b", "b", "a")),
        (WorkspaceNode("a", "a"), WorkspaceNode("b", "b"), WorkspaceNode("again", "a")),
        (WorkspaceNode("same", "a"), WorkspaceNode("same", "b")),
    ],
)
def test_reject_missing_raw_unknown_duplicate_or_invalid_parent(nodes):
    snapshot, workspace, _, _ = workspace_fixture()
    with pytest.raises(ValueError):
        prepare_evidence_workspace(snapshot, nodes, workspace.source_refs, enabled=True)


def test_omit_summary_keeps_every_raw_source_and_original_snapshot():
    snapshot, workspace, _, _ = workspace_fixture()
    before = snapshot.to_dict()
    simpler = prepare_evidence_workspace(
        snapshot,
        (WorkspaceNode("a", "a"), WorkspaceNode("b", "b")),
        workspace.source_refs,
        enabled=True,
    )
    assert len(simpler.nodes) == 2
    assert snapshot.to_dict() == before
    assert len(snapshot.interpretations) == 1


def test_authorized_catalog_recovers_sources_without_private_labels():
    snapshot, workspace, topology, query = workspace_fixture()
    seen = []
    result = retrieve_evidence_workspace(
        workspace,
        snapshot,
        topology,
        query,
        authorize=lambda req: seen.append(req) or True,
        enabled=True,
    )
    assert len(seen) == 2
    assert result["optimizer_eligible"] is False
    assert result["confers_authority"] is False
    catalog = result["producer_view"]
    text = json.dumps(catalog)
    for private in ("private-", "principal-secret", "scope-secret", "artifact:", "record:"):
        assert private not in text
    rows = {row["handle"]: row for row in catalog["items"]}
    summary = next(row for row in rows.values() if row["kind"] == "synthetic")
    assert summary["channel"] == "unverified"
    assert len(summary["source_handles"]) == 2
    for handle in summary["source_handles"]:
        source = rows[handle]
        assert source["kind"] == "source"
        assert source["parent_handle"] == summary["handle"]
        assert source["source_handles"] == [handle]
        assert source["location"] == {"kind": "page", "value": "1"}
        assert source["text"] == snapshot.sources[0].quotation
        assert source["observed_at"] == [snapshot.sources[0].quality.recorded_at]
    with pytest.raises(ValueError):
        require_training_eligible(result)


@pytest.mark.parametrize("decision", [None, False, "yes"])
def test_no_access_means_no_catalog_content(decision):
    snapshot, workspace, topology, query = workspace_fixture()
    seen = []
    authorizer = None if decision is None else lambda req: seen.append(req) or decision
    result = retrieve_evidence_workspace(
        workspace, snapshot, topology, query, authorize=authorizer, enabled=True
    )
    assert result["producer_view"]["items"] == []
    assert len(seen) == (0 if decision is None else 2)


def test_partial_denial_withholds_summary_and_leaves_allowed_source_at_root():
    snapshot, workspace, topology, query = workspace_fixture()
    result = retrieve_evidence_workspace(
        workspace,
        snapshot,
        topology,
        query,
        authorize=lambda req: req.artifact_key[0] == "A",
        enabled=True,
    )
    rows = result["producer_view"]["items"]
    assert len(rows) == 1
    assert rows[0]["kind"] == "source"
    assert rows[0]["parent_handle"] is None
    assert "summary" not in json.dumps(result["producer_view"])


def test_access_is_rechecked_on_every_read():
    snapshot, workspace, topology, query = workspace_fixture()
    allowed = True

    def authorize(req):
        return allowed

    first = retrieve_evidence_workspace(
        workspace, snapshot, topology, query, authorize=authorize, enabled=True
    )
    allowed = False
    second = retrieve_evidence_workspace(
        workspace, snapshot, topology, query, authorize=authorize, enabled=True
    )
    assert len(first["producer_view"]["items"]) == 3
    assert second["producer_view"]["items"] == []


def test_snapshot_mismatch_and_mutated_workspace_fail_before_access():
    snapshot, workspace, topology, query = workspace_fixture()
    seen = []
    other = replace(snapshot, state=EvidenceState(tuple(reversed(snapshot.state.claims))))
    with pytest.raises(ValueError):
        retrieve_evidence_workspace(
            workspace,
            other,
            topology,
            query,
            authorize=lambda req: seen.append(req) or True,
            enabled=True,
        )
    object.__setattr__(workspace, "nodes", workspace.nodes[:-1])
    with pytest.raises(ValueError):
        retrieve_evidence_workspace(
            workspace,
            snapshot,
            topology,
            query,
            authorize=lambda req: seen.append(req) or True,
            enabled=True,
        )
    assert seen == []


def test_capacity_fails_before_access(monkeypatch):
    snapshot, workspace, topology, query = workspace_fixture()
    seen = []
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="capacity"):
        retrieve_evidence_workspace(
            workspace,
            snapshot,
            topology,
            query,
            authorize=lambda req: seen.append(req) or True,
            enabled=True,
        )
    assert seen == []


def test_catalog_and_individual_items_retain_non_training_labels():
    snapshot, workspace, topology, query = workspace_fixture()
    result = retrieve_evidence_workspace(
        workspace, snapshot, topology, query, authorize=lambda req: True, enabled=True
    )
    for value in (result["producer_view"], *result["producer_view"]["items"]):
        with pytest.raises(ValueError, match="forbids optimization"):
            require_training_eligible(value)


def test_index_serialization_is_bounded():
    snapshot, workspace, _, _ = workspace_fixture()
    with pytest.raises(ValueError, match="capacity"):
        prepare_evidence_workspace(
            snapshot,
            (WorkspaceNode("x" * 300000, "a"), WorkspaceNode("b", "b")),
            workspace.source_refs,
            enabled=True,
        )


def test_stale_sources_and_summaries_do_not_become_current_evidence():
    snapshot, workspace, topology, query = workspace_fixture()
    query = replace(query, assessed_at="2026-10-10T12:00:00Z")
    current = retrieve_evidence_workspace(
        workspace, snapshot, topology, query, authorize=lambda req: True, enabled=True
    )
    assert current["producer_view"]["items"] == []
    historical = retrieve_evidence_workspace(
        workspace,
        snapshot,
        topology,
        replace(query, purpose="historical"),
        authorize=lambda req: True,
        enabled=True,
    )
    assert len(historical["producer_view"]["items"]) == 3
    assert {row["channel"] for row in historical["producer_view"]["items"]} == {"historical"}
    assert all(
        row["observed_at"] == ["2026-10-09T12:00:00Z"]
        for row in historical["producer_view"]["items"]
    )


def test_hidden_source_restrictions_survive_the_projected_catalog():
    snapshot, workspace, _, _ = workspace_fixture()
    snapshot = replace(
        snapshot,
        artifacts=(
            replace(
                snapshot.artifacts[0], source_training_eligibility=TrainingEligibility.HIDDEN_EVAL
            ),
            snapshot.artifacts[1],
        ),
    )
    workspace = prepare_evidence_workspace(
        snapshot, workspace.nodes, workspace.source_refs, enabled=True
    )
    result = retrieve_evidence_workspace(
        workspace,
        snapshot,
        make_topology(snapshot, "synthesis"),
        make_query(snapshot),
        authorize=lambda req: True,
        enabled=True,
    )
    assert result["training_eligibility"] == "HIDDEN_EVAL"
    assert result["producer_view"]["training_eligibility"] == "HIDDEN_EVAL"
    summary = next(row for row in result["producer_view"]["items"] if row["kind"] == "synthetic")
    assert summary["training_eligibility"] == "HIDDEN_EVAL"


def test_ancestry_expansion_is_reserved_before_access():
    snapshot, workspace, topology, query = workspace_fixture()
    long_id = "source-" + "x" * 100000
    first = replace(snapshot.sources[0], item_id=long_id)
    summary = replace(
        snapshot.interpretations[0],
        inputs=(
            (long_id, artifact_digest(first)),
            (snapshot.sources[1].item_id, artifact_digest(snapshot.sources[1])),
        ),
    )
    derived = [summary]
    for i in range(48):
        derived.append(
            replace(
                summary,
                item_id=f"summary-{i}",
                transform_id=f"transform-{i}",
                inputs=((derived[-1].item_id, artifact_digest(derived[-1])),),
            )
        )
    snapshot = replace(
        snapshot, sources=(first, snapshot.sources[1]), interpretations=tuple(derived)
    )
    workspace = prepare_evidence_workspace(
        snapshot,
        (WorkspaceNode("a", long_id), WorkspaceNode("b", "b")),
        workspace.source_refs,
        enabled=True,
    )
    topology = replace(topology, snapshot_digest=artifact_digest(snapshot), routes=())
    seen = []
    with pytest.raises(ValueError, match="capacity"):
        retrieve_evidence_workspace(
            workspace,
            snapshot,
            topology,
            query,
            authorize=lambda req: seen.append(req) or True,
            enabled=True,
        )
    assert seen == []


@pytest.mark.parametrize("mutation", ["unknown_field", "train_label", "wrong_version"])
def test_record_rejects_authority_and_schema_extensions(mutation):
    _, workspace, _, _ = workspace_fixture()
    value = workspace.to_dict()
    if mutation == "unknown_field":
        value["confers_authority"] = True
    elif mutation == "train_label":
        value["training_eligibility"] = "TRAIN"
    else:
        value["schema_version"] = "evidence-workspace-v999"
    with pytest.raises(ValueError):
        EvidenceWorkspace.from_dict(value)


def test_catalog_omits_inactive_summary_while_preserving_source_fallback():
    snapshot, workspace, topology, query = workspace_fixture()
    workspace = prepare_evidence_workspace(
        snapshot,
        (WorkspaceNode("a", "a"), WorkspaceNode("b", "b")),
        workspace.source_refs,
        enabled=True,
    )
    result = retrieve_evidence_workspace(
        workspace, snapshot, topology, query, authorize=lambda req: True, enabled=True
    )
    assert len(result["producer_view"]["items"]) == 2
    assert all(
        row["kind"] == "source" and row["parent_handle"] is None
        for row in result["producer_view"]["items"]
    )


def test_workspace_rejects_parent_without_matching_direct_input():
    snapshot, workspace, _, _ = workspace_fixture()
    summary = snapshot.interpretations[0]
    claim = replace(summary.claim, evidence_refs=snapshot.sources[0].claim.evidence_refs)
    summary = replace(
        summary,
        claim=claim,
        inputs=(("a", artifact_digest(snapshot.sources[0])),),
        output_digest=payload_digest(claim.to_dict()),
    )
    snapshot = replace(
        snapshot,
        interpretations=(summary,),
        state=EvidenceState(tuple(s.claim for s in snapshot.sources) + (claim,)),
    )
    with pytest.raises(ValueError, match="direct derived-input"):
        prepare_evidence_workspace(snapshot, workspace.nodes, workspace.source_refs, enabled=True)
