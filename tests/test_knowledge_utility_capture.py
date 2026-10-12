"""Frozen pilot capture cannot replace attempts or grant evidence authority."""

# Standard library
from copy import deepcopy

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from scripts.prepare_knowledge_utility_pilot import prepare_pilot


@pytest.fixture
def records():
    from scripts import knowledge_utility_capture_records

    return knowledge_utility_capture_records


def prices(input_rate="0.75", output_rate="4.5"):
    return dict(
        model="gpt-5.4-mini-2026-03-17",
        service_tier="default",
        currency="USD",
        verified_at="2026-10-12T00:00:00Z",
        source_urls=["https://developers.openai.com/api/docs/pricing"],
        input_usd_per_million=input_rate,
        output_usd_per_million=output_rate,
        counting_policy="no_charge",
    )


def execution(manifest, run_dir, budget="5"):
    return dict(
        schema_version="knowledge-utility-pilot-execution-v1",
        run_id="authored-test-run",
        packet_digest=manifest["packet_digest"],
        manifest_digest=manifest["manifest_digest"],
        implementation_commit="a" * 40,
        run_directory=str(run_dir.resolve()),
        profile=manifest["profile"],
        prices=prices(),
        approved_budget_usd=budget,
        approval_reference="simulated fixture; not live authorization",
    )


def test_manifest_matches_all_frozen_messages(records):
    packet = prepare_pilot()
    manifest = records.build_manifest(packet)
    assert len(manifest["requests"]) == 72
    for seq, (actual, actor) in enumerate(zip(manifest["requests"], packet["actor_requests"])):
        assert actual["sequence"] == seq
        assert actual["slot_id"] == actor["slot_id"]
        payload = actual["payload"]
        assert payload["input"] == actor["messages"]
        assert set(payload) == {
            "input",
            "model",
            "reasoning",
            "temperature",
            "max_output_tokens",
            "tools",
            "store",
            "service_tier",
            "truncation",
            "stream",
        }
        assert payload["model"] == "gpt-5.4-mini-2026-03-17"
        assert payload["reasoning"] == {"effort": "none"}
        assert payload["temperature"] == 1 and payload["max_output_tokens"] == 256
        assert payload["tools"] == [] and payload["store"] is False
        assert payload["truncation"] == "disabled" and payload["stream"] is False
        assert actual["request_digest"] == content_digest(payload)
        assert actual["count_payload"]["input"] == actor["messages"]


@pytest.mark.parametrize("reseal", [False, True])
def test_manifest_rejects_changed_packet(records, reseal):
    packet = prepare_pilot()
    packet["actor_requests"][0]["messages"][0]["content"] += " injected"
    if reseal:
        packet["packet_digest"] = content_digest(
            {k: v for k, v in packet.items() if k != "packet_digest"}
        )
    with pytest.raises(ValueError):
        records.build_manifest(packet)


@pytest.mark.parametrize("inp,out,want", [("0.75", "4.5", 2688), ("0.000001", "0", 1)])
def test_decimal_reservations(records, inp, out, want):
    assert records.reservation_micro_usd(records.parse_prices(prices(inp, out))) == want
    assert records.budget_micro_usd("0.002688") == 2688
    assert records.budget_micro_usd("0.002687") == 2687


@pytest.mark.parametrize("bad", ["-1", "+1", "1e0", "NaN", True, "0.0000001", "9" * 33])
def test_budget_rejects_nonexact_money(records, bad):
    with pytest.raises(ValueError):
        records.budget_micro_usd(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("counting_policy", "unknown"),
        ("counting_policy", "charged"),
        ("source_urls", ["https://developers.openai.com.evil.example/pricing"]),
        ("currency", "EUR"),
        ("verified_at", "tomorrow"),
        ("model", "other"),
    ],
)
def test_prices_reject_unreviewable_profile(records, field, value):
    raw = prices()
    raw[field] = value
    with pytest.raises(ValueError):
        records.parse_prices(raw)


def test_execution_bindings_reject_before_side_effects(records, tmp_path):
    manifest = records.build_manifest(prepare_pilot())
    run_dir = tmp_path / "run"
    raw = execution(manifest, run_dir)
    checked = records.validate_execution(raw, manifest, run_dir, "a" * 40)
    assert checked.budget_micro_usd == 5_000_000
    assert checked.reservation_micro_usd == 2688
    for field, value in [
        ("packet_digest", "b" * 64),
        ("manifest_digest", "b" * 64),
        ("implementation_commit", "b" * 40),
        ("run_directory", str(tmp_path)),
        ("profile", {}),
        ("approval_reference", ""),
    ]:
        changed = deepcopy(raw)
        changed[field] = value
        with pytest.raises(ValueError):
            records.validate_execution(changed, manifest, run_dir, "a" * 40)
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "data", [b'{"a":1,"a":2}', b"\xff", b"[" * 33 + b"]" * 33, b" " * (4 * 1024 * 1024 + 1)]
)
def test_bounded_json_rejects_bad_input(records, tmp_path, data):
    path = tmp_path / "input.json"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        records.read_bounded_json(path)
