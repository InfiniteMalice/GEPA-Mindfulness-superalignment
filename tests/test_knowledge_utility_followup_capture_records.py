"""Follow-up capture binds public requests and exact operational constraints."""

# Standard library
from copy import deepcopy
from datetime import timedelta

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from scripts.prepare_knowledge_utility_followup import prepare_followup
from scripts.prepare_knowledge_utility_pilot import prepare_pilot
from tests.followup_capture_fixtures import (
    COMMIT,
    NOW,
    approved_packet,
    execution_record,
    price_record,
    seal_execution,
)


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


@pytest.fixture
def approval_inputs(records, tmp_path):
    manifest = records.build_manifest(approved_packet())
    path = tmp_path / "run"
    return manifest, path, execution_record(manifest, path)


def test_execution_full_roster_budget_and_binding(records, approval_inputs):
    manifest, path, raw = approval_inputs
    result = records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)
    assert result.reservation_micro_usd == 2688
    assert result.budget_micro_usd == 1032192
    assert not path.exists()
    raw["profile"]["tools"].append("changed")
    assert result.raw["profile"]["tools"] == []
    raw = execution_record(manifest, path, budget="5")
    assert records.validate_execution(raw, manifest, path).budget_micro_usd == 5000000


@pytest.mark.parametrize(
    "field,value",
    [
        ("approved_budget_usd", v)
        for v in ("1.032191", "0", "5.000001", "-1", "2e0", 2.0, True, "1" * 33)
    ]
    + [
        ("schema_version", "other"),
        ("packet_digest", "0" * 64),
        ("manifest_digest", "0" * 64),
        ("implementation_commit", "b" * 40),
        ("approved_count_requests", True),
        ("approved_generation_requests", 383),
        ("approved_generation_requests", 384.0),
        ("execution_authorized", 1),
        ("approval_reference", ""),
        ("run_id", "bad\x00id"),
        ("run_id", "é" * 513),
        ("run_directory", "relative/run"),
        ("profile.temperature", True),
        ("profile.tools", None),
        ("prices.input_usd_per_million", "0"),
        ("provider_review.model_available", 1),
        ("provider_review.profile_compatible", False),
        ("provider_review.source_urls", ["https://example.com/prices"]),
        ("provider_review.counting_charge_reference", "\x00"),
        ("provider_review.verified_at", "2026-10-15T12:00:00+00:00"),
        ("provider_review.verified_at", "x" * 65),
        ("extra", 0),
    ],
)
def test_execution_rejects_resealed_invalid_fields(records, approval_inputs, field, value):
    manifest, path, raw = approval_inputs
    target = raw
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    seal_execution(raw)
    with pytest.raises(ValueError):
        records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)
    assert not path.exists()


@pytest.mark.parametrize("mode", ["missing", "bad_digest", "naive_now", "only_now", "only_commit"])
def test_execution_rejects_incomplete_bindings(records, approval_inputs, mode):
    manifest, path, raw = approval_inputs
    kwargs = dict(live_commit=COMMIT, now=NOW)
    if mode == "missing":
        del raw["approval_reference"]
        seal_execution(raw)
    elif mode == "bad_digest":
        raw["approval_reference"] = "changed without reseal"
    elif mode == "naive_now":
        kwargs["now"] = NOW.replace(tzinfo=None)
    else:
        del kwargs["live_commit" if mode == "only_now" else "now"]
    with pytest.raises(ValueError):
        records.validate_execution(raw, manifest, path, **kwargs)


@pytest.mark.parametrize(
    "age,valid", [(0, True), (86400, True), (-0.000001, False), (86400.000001, False)]
)
def test_live_freshness_and_historical_export(records, approval_inputs, age, valid):
    manifest, path, raw = approval_inputs
    stamp = (NOW - timedelta(seconds=age)).isoformat().replace("+00:00", "Z")
    raw["prices"]["verified_at"] = raw["provider_review"]["verified_at"] = stamp
    seal_execution(raw)
    records.validate_execution(raw, manifest, path)
    if valid:
        records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)
    else:
        with pytest.raises(ValueError):
            records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)


@pytest.mark.parametrize("policy", ["no_charge", "unverified_accepted"])
@pytest.mark.parametrize("accepted", [True, False, 1])
def test_counting_policy_requires_matching_acceptance(records, approval_inputs, policy, accepted):
    manifest, path, raw = approval_inputs
    raw["prices"]["counting_policy"] = policy
    raw["accept_unknown_counting_cost"] = accepted
    seal_execution(raw)
    if type(accepted) is bool and accepted == (policy == "unverified_accepted"):
        records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)
    else:
        with pytest.raises(ValueError):
            records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_link"])
def test_run_directory_alias_binding_and_reuse(records, approval_inputs, kind):
    manifest, path, raw = approval_inputs
    records.validate_execution(raw, manifest, path / ".." / path.name, live_commit=COMMIT, now=NOW)
    with pytest.raises(ValueError):
        records.validate_execution(
            raw, manifest, path.parent / "other", live_commit=COMMIT, now=NOW
        )
    if kind == "directory":
        path.mkdir()
    elif kind == "file":
        path.write_text("preserve")
    else:
        try:
            path.symlink_to(path.parent / "absent", target_is_directory=True)
        except OSError as error:
            pytest.skip(f"link creation unavailable: {error.winerror}")
        raw["run_directory"] = str(path.resolve())
        seal_execution(raw)
    records.validate_execution(raw, manifest, path)
    with pytest.raises(ValueError):
        records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)


def test_store_creates_canonical_directory_from_alias(records, approval_inputs):
    manifest, path, raw = approval_inputs
    execution = records.validate_execution(raw, manifest, path, live_commit=COMMIT, now=NOW)
    alias = path / ".." / path.name
    store = records.FollowupRunStore.create(alias, execution, manifest)
    assert store.root == path.resolve()
    assert (path / "execution-started.json").is_file()
    with pytest.raises(FileExistsError):
        records.FollowupRunStore.create(alias, execution, manifest)


def test_store_rejects_dangling_link(records, approval_inputs):
    manifest, path, raw = approval_inputs
    execution = records.validate_execution(raw, manifest, path)
    target = path.parent / "absent"
    try:
        path.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("OS denied test symlink creation")
    with pytest.raises(FileExistsError):
        records.FollowupRunStore.create(path, execution, manifest)
    assert path.is_symlink()
    assert not target.exists()
