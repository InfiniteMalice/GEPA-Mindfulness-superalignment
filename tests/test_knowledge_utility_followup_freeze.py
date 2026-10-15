"""Offline acceptance contracts; all approving decisions here are synthetic."""

# Standard library
import importlib
from collections import Counter
from copy import deepcopy

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from tests.followup_freeze_fixtures import install_synthetic_profile, synthetic_evidence
from tests.followup_scoring_fixtures import fresh_packet


@pytest.fixture
def prepared(monkeypatch):
    core = importlib.import_module("scripts.knowledge_utility_followup_freeze")
    packet = fresh_packet()
    evidence = synthetic_evidence(packet)
    install_synthetic_profile(monkeypatch, packet, evidence)
    return core, packet, evidence


def test_candidate_preserves_review_scope_and_nested_findings(prepared):
    core, packet, evidence = prepared
    before = deepcopy(packet), deepcopy(evidence)
    result = core.prepare_acceptance(packet, evidence)
    assert len(result["evidence"]) == 14
    assert len(result["tasks"]) == 48
    counts = Counter(c["assessment"] for t in result["tasks"] for c in t["checks"].values())
    assert counts == dict(reviewed_clear=184, resolved_by_recheck=5, limitation_retained=51)
    cases = {t["task_id"]: t for t in result["tasks"]}
    assert cases["followup-task_precedence-0"]["disposition"] == "carried_forward"
    assert cases["followup-task_precedence-0"]["checks"]["variant_structure"]["assessment"] == (
        "resolved_by_recheck"
    )
    assert cases["followup-flow_conservation-0"]["checks"]["answer"]["assessment"] == (
        "resolved_by_recheck"
    )
    assert {t["task_id"] for t in result["tasks"] if t["disposition"] != "carried_forward"} == {
        "followup-flow_conservation-0",
        "followup-task_precedence-1",
    }
    assert result["pending_task_count"] == 0
    assert "packet_frozen" not in result and "decision" not in result
    assert core.prepare_acceptance(packet, evidence) == result
    assert (packet, evidence) == before
    result["tasks"].clear()
    assert len(core.prepare_acceptance(packet, evidence)["tasks"]) == 48


@pytest.mark.parametrize("mutation", ["byte", "count", "missing", "extra", "first_pass", "type"])
def test_evidence_tampering_rejects_whole_candidate(prepared, mutation):
    core, packet, evidence = prepared
    if mutation == "byte":
        evidence["comparison"] = b"[]"
    elif mutation == "count":
        evidence["comparison"] += b" "
    elif mutation == "missing":
        del evidence["comparison"]
    elif mutation == "extra":
        evidence["extra"] = b"{}"
    elif mutation == "first_pass":
        evidence["first_pass"] = b'{"answer":"B"}'
    else:
        evidence["comparison"] = "{}"
    with pytest.raises(ValueError):
        core.prepare_acceptance(packet, evidence)


@pytest.mark.parametrize("field", ["schedule", "knowledge", "rubric", "report_template"])
def test_resealed_source_change_is_not_supported(prepared, field):
    core, packet, evidence = prepared
    packet[field]["unexpected"] = True
    packet["packet_digest"] = content_digest(
        {k: v for k, v in packet.items() if k != "packet_digest"}
    )
    with pytest.raises(ValueError):
        core.prepare_acceptance(packet, evidence)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "digest", "key", "option"])
def test_changed_task_roster_rejects_candidate(prepared, mutation):
    core, packet, evidence = prepared
    case = packet["evaluator_cases"][0]
    if mutation == "duplicate":
        packet["evaluator_cases"][1] = deepcopy(case)
    elif mutation == "unknown":
        case["task_id"] = "unknown"
    elif mutation == "digest":
        case["task_digest"] = "0" * 64
    elif mutation == "key":
        case["expected_option"] = "Z"
    else:
        case["options"]["A"] = "other"
    packet["packet_digest"] = content_digest(
        {k: v for k, v in packet.items() if k != "packet_digest"}
    )
    with pytest.raises(ValueError):
        core.prepare_acceptance(packet, evidence)
