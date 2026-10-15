"""Follow-up capture binds public requests and exact operational constraints."""

# Standard library
from copy import deepcopy

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from scripts.prepare_knowledge_utility_followup import prepare_followup
from scripts.prepare_knowledge_utility_pilot import prepare_pilot
from tests.followup_capture_fixtures import approved_packet, price_record


@pytest.fixture
def records():
    from scripts import knowledge_utility_followup_capture_records

    return knowledge_utility_followup_capture_records


def test_manifest_projects_all_384_slots(records):
    packet = approved_packet()
    before = deepcopy(packet)
    result = records.build_manifest(packet)
    assert len(result["requests"]) == 384
    assert result["schema_version"] == "knowledge-utility-followup-requests-v1"
    assert result["manifest_digest"] == (
        "af85b29c1f52f70ba5a05a7f56aa11e6848fe8ed8f13d4aaf4c04df81bea74c2"
    )
    assert len((canonical_json(result) + "\n").encode()) == 700160
    for i, (request, actor, slot) in enumerate(
        zip(result["requests"], packet["actor_requests"], packet["schedule"]["slots"])
    ):
        assert request["sequence"] == i
        assert request["slot_id"] == slot["slot_id"] == actor["slot_id"]
        assert request["payload"] == dict(result["profile"], input=actor["messages"])
        assert request["count_payload"] == {
            key: request["payload"][key] for key in ("model", "input", "reasoning", "tools")
        }
        assert request["request_digest"] == content_digest(request["payload"])
    result["requests"][0]["payload"]["input"][0]["content"] = "mutated"
    result["requests"][0]["payload"]["tools"].append({})
    assert (
        result["requests"][0]["count_payload"]["input"] == packet["actor_requests"][0]["messages"]
    )
    assert result["profile"]["tools"] == result["requests"][1]["payload"]["tools"] == []
    assert packet == before


@pytest.mark.parametrize("kind", ["draft", "pilot", "modified", "resealed", "future_registered"])
def test_manifest_rejects_other_packets(records, monkeypatch, kind):
    packet = approved_packet()
    if kind == "draft":
        packet = prepare_followup()
    elif kind == "pilot":
        packet = prepare_pilot()
    else:
        packet["as_of"] = "different"
        if kind != "modified":
            packet["packet_digest"] = content_digest(
                {key: value for key, value in packet.items() if key != "packet_digest"}
            )
        if kind == "future_registered":
            monkeypatch.setattr(records, "validate_supported_packet", lambda value: value)
    with pytest.raises(ValueError):
        records.build_manifest(packet)


@pytest.mark.parametrize("policy,cost", [("no_charge", "0"), ("unverified_accepted", None)])
def test_preparation_is_offline_and_labels_unknowns(records, policy, cost):
    packet, prices = approved_packet(), price_record()
    prices["counting_policy"] = policy
    before = deepcopy((packet, prices))
    no_price = records.prepare_capture(packet)
    assert no_price["reservation_estimate"] is None
    assert no_price["token_counts"] == "unknown"
    result = records.prepare_capture(packet, prices)
    assert result["reservation_estimate"] == dict(
        per_request_micro_usd=2688,
        roster_micro_usd=1032192,
        budget_scope="generation_only",
        counting_policy=policy,
        counting_cost_usd=cost,
    )
    assert result["token_counts"] == "unknown"
    assert len((canonical_json(result) + "\n").encode()) <= 4 * 1024 * 1024
    assert (packet, prices) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_usd_per_million", "0"),
        ("output_usd_per_million", "0"),
        ("input_usd_per_million", 0.75),
        ("input_usd_per_million", "-1"),
        ("input_usd_per_million", "1e-6"),
        ("input_usd_per_million", "1" * 33),
        ("verified_at", "2026-10-15"),
        ("verified_at", "2026-10-15T12:00:00+01:00Z"),
        ("model", "other"),
        ("extra", True),
    ],
)
def test_preparation_rejects_invalid_rates_and_fields(records, field, value):
    prices = price_record()
    prices[field] = value
    with pytest.raises(ValueError):
        records.prepare_capture(approved_packet(), prices)
