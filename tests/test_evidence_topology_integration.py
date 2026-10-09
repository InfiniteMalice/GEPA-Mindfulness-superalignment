"""Offline public guide and preserved numeric, memory, continuity and training boundaries."""

# Standard library
import re
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest
from artifact_fixtures import make_protocol, make_snapshot
from test_epistemic_continuity import assess, commitment, event_sequence
from test_scalar_fusion import measurement

# Local
from evaluation.evidence_topology import analyze_evidence_topology
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.verification.artifact_evidence import retrieve_artifact_evidence
from gepa_mindfulness.verification.artifact_records import artifact_digest
from gepa_mindfulness.verification.artifact_topology import EvidenceTopology
from gepa_mindfulness.verification.claim_graph import ClaimGraph, ClaimNode
from gepa_mindfulness.verification.evidence_use import (
    EvidenceUseAssessment,
    MemoryInfluence,
    MemoryKind,
)
from gepa_mindfulness.verification.state import EvidenceClaim, EvidenceState
from semantic_intent_robustness.epistemic_continuity import (
    EvidenceWindow,
    recall_historical_support,
)


def test_offline_guide(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("guide attempted network access")

    monkeypatch.setattr("socket.socket.connect", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    path = Path(__file__).resolve().parents[1] / "docs" / "evidence_topology.md"
    code = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)[0]
    ns = {}
    exec(compile(code, str(path), "exec"), ns)
    assert ns["report"]["training_effect_established"] is False
    assert ns["revoked"].selected_item_ids == ("b",)
    assert ns["report"]["paired"][0]["deltas"]["correctness"] is None
    assert ns["redundant"]["slots"][0]["surviving_route_ids"] == ["via-b"]
    assert ns["bridge"]["slots"][0]["surviving_route_ids"] == []
    assert ns["stale"].selected_item_ids == ("b",)
    assert ns["same_named"].selected_item_ids == ()
    assert ns["old_state"].to_dict() == ns["original_state"]
    with pytest.raises(ValueError):
        require_training_eligible(ns["report"])


@pytest.mark.parametrize("age", [0, 61])
def test_numeric_eligibility_unchanged(age):
    p = make_protocol()
    source = p.snapshot.sources[0]
    at = "2026-10-09T12:01:01Z" if age else p.query.assessed_at
    numeric = EvidenceUseAssessment(
        state=p.snapshot.state,
        claim_id=source.claim.claim_id,
        measurement=measurement(0, evidence_refs=source.claim.evidence_refs),
        memory=source.memory,
        kind=MemoryKind.FACT,
        target_influence=MemoryInfluence.BOUND,
        quality=source.quality,
        policy=p.query.policy,
        assessed_at=at,
    )
    if age:
        with pytest.raises(ValueError, match="ineligible"):
            numeric.measurement_for_update()
    else:
        before = numeric.measurement_for_update().to_dict()
    analyze_evidence_topology(
        replace(p, query=replace(p.query, assessed_at=at)),
        None,
        authorize=lambda _: True,
        enabled=True,
    )
    if age:
        with pytest.raises(ValueError, match="ineligible"):
            numeric.measurement_for_update()
    else:
        assert numeric.measurement_for_update().to_dict() == before
    assert numeric.memory == source.memory
    assert numeric.quality.authority == "information_only"


def test_continuity_provenance_unchanged():
    item = commitment()
    window = EvidenceWindow.build(event_sequence(), "proposed-2")
    before = window.digest
    source_events = item.source_event_refs
    s = make_snapshot()
    claim = EvidenceClaim(item.commitment_id, item.claim_summary, item.evidence_refs, "observed")
    a = replace(
        s.artifacts[0],
        observation=replace(s.artifacts[0].observation, evidence_refs=item.evidence_refs),
    )
    source = replace(
        s.sources[0],
        item_id=item.commitment_id,
        claim=claim,
        memory=item.memory,
        quotation=item.claim_summary,
    )
    s = replace(s, artifacts=(a,), sources=(source,), state=EvidenceState((claim,)))
    topology = EvidenceTopology(
        artifact_digest(s),
        ClaimGraph((ClaimNode(claim, "host", None, "LEGACY_UNSPECIFIED", 1, 1),), ()),
        (),
    )
    from artifact_fixtures import make_query

    result = retrieve_artifact_evidence(
        s, topology, make_query(s), authorize=lambda _: True, enabled=True
    )
    assert result.selected_item_ids == ("k",)
    recalled = recall_historical_support(
        commitments=(item,), assessment=assess(), current_state=None, prior_states=()
    )
    assert recalled.commitments[0] == item
    assert recalled.commitments[0].source_event_refs == source_events
    assert recalled.commitments[0].memory.source_identity == source.memory.source_identity
    assert window.digest == before


def test_export_never_grants_training_or_action_authority():
    report = analyze_evidence_topology(make_protocol(), None, enabled=True)
    assert report["optimizer_input"] is False and report["confers_authority"] is False
    with pytest.raises(ValueError):
        require_training_eligible(report)
