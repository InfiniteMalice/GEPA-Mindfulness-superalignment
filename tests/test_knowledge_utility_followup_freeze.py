"""Offline acceptance contracts; all approving decisions here are synthetic."""

# Standard library
import importlib
from collections import Counter
from copy import deepcopy
from hashlib import sha256

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from tests.followup_freeze_fixtures import (
    install_synthetic_profile,
    synthetic_decision,
    synthetic_evidence,
)
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


def test_freeze_only_changes_declared_metadata(prepared):
    core, packet, evidence = prepared
    candidate = core.prepare_acceptance(packet, evidence)
    decision = synthetic_decision(candidate)
    before = deepcopy((packet, candidate, decision))
    frozen = core.freeze_followup(packet, evidence, candidate, decision)
    assert frozen["schema_version"] == "knowledge-utility-followup-frozen-packet-v1"
    assert frozen["status"] == "frozen_with_accepted_task_review"
    assert frozen["packet_frozen"] is True
    assert frozen["execution_authorized"] is False
    assert frozen["training_eligibility"] == "DEVELOPMENT"
    assert frozen["confers_authority"] is False and frozen["optimizer_input"] is False
    changed = {k for k in packet if packet[k] != frozen[k]}
    assert changed == {"schema_version", "status", "packet_frozen", "packet_digest"}
    assert set(frozen) - set(packet) == {"freeze_record"}
    assert frozen["freeze_record"]["acceptance"] == candidate
    assert frozen["freeze_record"]["decision"] == decision
    assert frozen == core.freeze_followup(packet, evidence, candidate, decision)
    assert (packet, candidate, decision) == before
    frozen["freeze_record"]["decision"]["maintainer_id"] = "changed"
    assert decision["maintainer_id"] == "synthetic-test"


@pytest.mark.parametrize(
    "field,value",
    [
        ("accept_task_review", False),
        ("accept_task_review", 1),
        ("authorize_freeze", False),
        ("authorize_freeze", 1),
        ("execution_authorized", True),
        ("execution_authorized", 0),
        ("maintainer_id", ""),
        ("maintainer_id", "x" * 1025),
        ("authorization_ref", " "),
        ("authorization_ref", "é" * 513),
        ("candidate_digest", "0" * 64),
        ("source_packet_digest", "0" * 64),
        ("accepted_limitations", []),
        ("schema_version", "unknown"),
    ],
)
def test_freeze_rejects_invalid_resealed_decisions(prepared, field, value):
    core, packet, evidence = prepared
    candidate = core.prepare_acceptance(packet, evidence)
    decision = synthetic_decision(candidate)
    decision[field] = value
    decision["decision_digest"] = content_digest(
        {k: v for k, v in decision.items() if k != "decision_digest"}
    )
    with pytest.raises(ValueError):
        core.freeze_followup(packet, evidence, candidate, decision)


@pytest.mark.parametrize("mutation", ["missing", "extra", "digest", "duplicate", "extra_limit"])
def test_freeze_rejects_incomplete_decisions(prepared, mutation):
    core, packet, evidence = prepared
    candidate = core.prepare_acceptance(packet, evidence)
    decision = synthetic_decision(candidate)
    if mutation == "missing":
        del decision["authorize_freeze"]
    elif mutation == "extra":
        decision["unknown"] = True
    elif mutation == "digest":
        decision["decision_digest"] = "0" * 64
    elif mutation == "duplicate":
        decision["accepted_limitations"].append(decision["accepted_limitations"][0])
    else:
        decision["accepted_limitations"].append("unknown")
    with pytest.raises(ValueError):
        core.freeze_followup(packet, evidence, candidate, decision)


@pytest.mark.parametrize("field", ["tasks", "evidence", "candidate_digest", "pending_task_count"])
def test_freeze_revalidates_entire_candidate(prepared, field):
    core, packet, evidence = prepared
    candidate = core.prepare_acceptance(packet, evidence)
    decision = synthetic_decision(candidate)
    candidate[field] = None
    with pytest.raises(ValueError):
        core.freeze_followup(packet, evidence, candidate, decision)


def test_candidate_uses_captured_bytes_not_reopened_file(prepared, tmp_path):
    core, packet, evidence = prepared
    source = tmp_path / "review.json"
    source.write_bytes(evidence["comparison"])
    evidence["comparison"] = source.read_bytes()
    source.write_bytes(b"replaced after reading")
    assert len(core.prepare_acceptance(packet, evidence)["tasks"]) == 48


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff", b"[" * 33 + b"]" * 33])
def test_pinned_evidence_still_requires_strict_json(prepared, monkeypatch, raw):
    core, packet, evidence = prepared
    evidence["comparison"] = raw
    monkeypatch.setitem(
        core.profile.EVIDENCE_PINS, "comparison", (sha256(raw).hexdigest(), len(raw))
    )
    with pytest.raises(ValueError):
        core.prepare_acceptance(packet, evidence)


def test_candidate_boolean_is_not_zero(prepared):
    core, packet, evidence = prepared
    candidate = core.prepare_acceptance(packet, evidence)
    decision = synthetic_decision(candidate)
    candidate["pending_task_count"] = False
    with pytest.raises(ValueError):
        core.freeze_followup(packet, evidence, candidate, decision)


def test_evidence_total_and_publication_byte_boundaries(prepared, monkeypatch):
    core, packet, evidence = prepared
    total = sum(map(len, evidence.values()))
    monkeypatch.setattr(core, "MAX_EVIDENCE_BYTES", total)
    candidate = core.prepare_acceptance(packet, evidence)
    monkeypatch.setattr(core, "MAX_EVIDENCE_BYTES", total - 1)
    with pytest.raises(ValueError):
        core.prepare_acceptance(packet, evidence)
    monkeypatch.setattr(core, "MAX_EVIDENCE_BYTES", total)
    size = len(canonical_json(candidate).encode())
    monkeypatch.setattr(core, "MAX_RECORD_BYTES", size + 1)
    assert core.prepare_acceptance(packet, evidence) == candidate
    monkeypatch.setattr(core, "MAX_RECORD_BYTES", size)
    with pytest.raises(ValueError):
        core.prepare_acceptance(packet, evidence)


def test_frozen_packet_publication_boundary(prepared, monkeypatch):
    core, packet, evidence = prepared
    candidate = core.prepare_acceptance(packet, evidence)
    decision = synthetic_decision(candidate)
    frozen = core.freeze_followup(packet, evidence, candidate, decision)
    size = len(canonical_json(frozen).encode())
    monkeypatch.setattr(core, "MAX_INPUT_BYTES", size + 1)
    assert core.freeze_followup(packet, evidence, candidate, decision) == frozen
    monkeypatch.setattr(core, "MAX_INPUT_BYTES", size)
    with pytest.raises(ValueError):
        core.freeze_followup(packet, evidence, candidate, decision)


@pytest.mark.parametrize("value", [None, float("nan"), {"padding": "x" * (1024 * 1024)}])
def test_invalid_candidate_or_decision_rejects_without_output(prepared, value):
    core, packet, evidence = prepared
    candidate = core.prepare_acceptance(packet, evidence)
    with pytest.raises(ValueError):
        core.freeze_followup(packet, evidence, value, synthetic_decision(candidate))
    with pytest.raises(ValueError):
        core.freeze_followup(packet, evidence, candidate, value)
