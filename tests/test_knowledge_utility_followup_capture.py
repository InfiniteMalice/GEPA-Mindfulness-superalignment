"""Real durable follow-up orchestration with a synthetic provider boundary."""

# Standard library
import json
import multiprocessing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest

# Local
from scripts import knowledge_utility_followup_capture_records as records
from tests.followup_capture_fixtures import (
    COMMIT,
    NOW,
    FakeTransport,
    approved_packet,
    execution_record,
    provider_result,
    seal_execution,
    transport_result,
)


@pytest.fixture
def runner(monkeypatch):
    from scripts import capture_knowledge_utility_followup

    monkeypatch.setattr(capture_knowledge_utility_followup, "implementation_commit", lambda: COMMIT)
    monkeypatch.setattr(capture_knowledge_utility_followup, "utc_now", lambda: NOW.isoformat())
    return capture_knowledge_utility_followup


@pytest.fixture
def live_inputs(tmp_path):
    packet = approved_packet()
    path = tmp_path / "run"
    manifest = records.build_manifest(packet)
    return packet, path, execution_record(manifest, path)


def test_full_roster_counts_before_generations(runner, live_inputs):
    packet, path, raw = live_inputs
    original = deepcopy(packet)
    fake = FakeTransport()

    def responder(endpoint, payload, index):
        slot = index if index < 384 else index - 384
        relative = f"counts/{slot:03}-intent.json" if index < 384 else f"attempts/{slot:03}.json"
        intent = json.loads((path / relative).read_text())
        assert intent["sequence"] == slot
        assert intent["execution_digest"] == raw["execution_digest"]
        assert (path / "execution-started.json").exists()
        assert payload["input"] == packet["actor_requests"][slot]["messages"]
        return FakeTransport().post(endpoint, payload)

    fake.responder = responder
    summary = runner.run_live(packet, raw, path, lambda: fake)
    assert [endpoint for endpoint, _ in fake.calls] == (
        ["responses/input_tokens"] * 384 + ["responses"] * 384
    )
    assert summary["state"] == "finished"
    assert summary["reserved_micro_usd"] == 1032192
    assert summary["count_requests"] == summary["generation_attempts"] == 384
    assert summary["total_billed_cost_usd"] is None
    count = json.loads((path / "counts/383-result.json").read_text())
    assert count["usage"] == dict(input_tokens=100, output_tokens=0)
    assert count["cost_scope"] == "counting"
    assert count["estimated_cost_usd"] is count["billed_cost_usd"] is None
    result = json.loads((path / "results/383.json").read_text())
    assert result["estimated_cost_usd"] == "0.00012"
    assert result["billed_cost_usd"] is None
    assert packet == original


@pytest.mark.parametrize("index", [0, 191, 383])
@pytest.mark.parametrize("fault", ["excessive", "bool", "malformed", "http", "redacted", "timeout"])
def test_count_failure_prevents_all_generations(runner, live_inputs, index, fault):
    packet, path, raw = live_inputs
    good = transport_result(dict(object="response.input_tokens", input_tokens=100))
    bad = {
        "excessive": transport_result(dict(object="response.input_tokens", input_tokens=2049)),
        "bool": transport_result(dict(object="response.input_tokens", input_tokens=True)),
        "malformed": replace(good, body=b"{"),
        "http": replace(good, status=429),
        "redacted": replace(good, redacted=True),
        "timeout": replace(good, error_code="deadline_exceeded"),
    }[fault]
    fake = FakeTransport(lambda endpoint, payload, i: bad if i == index else good)
    summary = runner.run_live(packet, raw, path, lambda: fake)
    assert summary["reason"] == "count_failed"
    assert len(fake.calls) == index + 1
    assert summary["generation_attempts"] == summary["reserved_micro_usd"] == 0
    assert not list((path / "attempts").iterdir())


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("refused", 384),
        ("truncated", 384),
        ("model", 1),
        ("usage", 1),
        ("malformed", 1),
        ("timeout", 1),
    ],
)
def test_generation_outcomes_control_progress(runner, live_inputs, fault, expected):
    packet, path, raw = live_inputs
    response = json.loads(provider_result().body)
    if fault == "refused":
        response["output"][0]["content"] = [{"type": "refusal", "refusal": "no"}]
    if fault == "model":
        response["model"] = "other"
    if fault == "usage":
        response["usage"]["output_tokens"] = 257
    result = transport_result(response)
    if fault == "truncated":
        result = provider_result(status="incomplete")
    if fault == "malformed":
        result = replace(result, body=b"{")
    if fault == "timeout":
        result = replace(result, error_code="deadline_exceeded")
    fake = FakeTransport(
        lambda endpoint, payload, i: FakeTransport().post(endpoint, payload) if i < 384 else result
    )
    summary = runner.run_live(packet, raw, path, lambda: fake)
    assert len(fake.calls) == 384 + expected
    assert summary["reserved_micro_usd"] == expected * 2688
    assert summary["state"] == ("finished" if expected == 384 else "stopped")


@pytest.mark.parametrize(
    "relative,calls,reserved",
    [
        ("execution.json", 0, 0),
        ("counts/000-intent.json", 0, 0),
        ("counts/000-result.json", 1, 0),
        ("attempts/000.json", 384, 2688),
        ("results/000.json", 385, 2688),
        ("terminal.json", 768, 1032192),
    ],
)
def test_write_failure_never_resends(runner, live_inputs, monkeypatch, relative, calls, reserved):
    packet, path, raw = live_inputs
    original = records.FollowupRunStore.write_record
    fake = FakeTransport()

    def broken(self, name, data):
        if name == relative:
            (self.root / name).write_bytes(b'{"partial":')
            raise OSError("synthetic sensitive error")
        return original(self, name, data)

    monkeypatch.setattr(records.FollowupRunStore, "write_record", broken)
    if relative in ("execution.json", "terminal.json"):
        with pytest.raises(OSError):
            runner.run_live(packet, raw, path, lambda: fake)
    else:
        summary = runner.run_live(packet, raw, path, lambda: fake)
        assert summary["state"] == "uncertain"
        assert summary["reserved_micro_usd"] == reserved
    assert len(fake.calls) == calls
    assert (path / relative).read_bytes() == b'{"partial":'


@pytest.mark.parametrize("index", [0, 384])
def test_interrupt_never_resends(runner, live_inputs, index):
    packet, path, raw = live_inputs

    def interrupt(endpoint, payload, i):
        if i == index:
            raise KeyboardInterrupt
        return FakeTransport().post(endpoint, payload)

    fake = FakeTransport(interrupt)
    result = runner.run_live(packet, raw, path, lambda: fake)
    assert result["state"] == "uncertain"
    assert len(fake.calls) == index + 1
    assert result["reserved_micro_usd"] == (2688 if index else 0)


def test_invalid_approval_never_constructs_transport(runner, live_inputs):
    packet, path, raw = live_inputs
    raw["approved_budget_usd"] = "0"
    seal_execution(raw)
    with pytest.raises(ValueError):
        runner.run_live(packet, raw, path, lambda: pytest.fail("credential access"))
    assert not path.exists()


def _claim_store(path, raw, manifest, queue):
    checked = records.validate_execution(raw, manifest, Path(path))
    try:
        records.FollowupRunStore.create(Path(path), checked, manifest)
        queue.put("created")
    except FileExistsError:
        queue.put("exists")


def test_store_single_owner_and_write_boundaries(live_inputs, monkeypatch):
    packet, path, raw = live_inputs
    manifest = records.build_manifest(packet)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    workers = [
        ctx.Process(target=_claim_store, args=(path, raw, manifest, queue)) for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        assert sorted(queue.get(timeout=45) for _ in workers) == ["created", "exists"]
    finally:
        for worker in workers:
            worker.join(5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        queue.close()
    store = records.FollowupRunStore(
        path,
        {
            key: raw[key]
            for key in ("run_id", "packet_digest", "manifest_digest", "execution_digest")
        },
    )
    for name in ("attempts/000.json", "attempts/383.json"):
        store.write_record(name, {})
    for name in ("attempts/384.json", "../escape.json", "counts/00-intent.json"):
        with pytest.raises(ValueError):
            store.write_record(name, {})
    with pytest.raises(ValueError):
        store.write_record("results/000.json", dict(run_id="overwrite"))
    with pytest.raises(ValueError):
        store.write_record("results/000.json", dict(large="x" * (4 * 1024 * 1024)))
    monkeypatch.setattr(records.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        store.write_record("results/000.json", {})
    assert (path / "results/000.json").exists()
    with pytest.raises(FileExistsError):
        store.write_record("results/000.json", {})


def test_retained_result_bounds_and_roundtrip(runner, live_inputs):
    packet, path, raw = live_inputs
    execution = records.validate_execution(raw, records.build_manifest(packet), path)
    result = provider_result()
    saved = runner.retained_result(result, execution, generation=True)
    assert runner.restore_transport(saved["transport"]) == result
    counting = runner.retained_result(result, execution, generation=False)
    assert counting["usage"] == dict(input_tokens=100, output_tokens=10)
    assert counting["estimated_cost_usd"] is counting["billed_cost_usd"] is None
    assert (
        runner.retained_result(replace(result, body=b"{"), execution, generation=True)["provider"]
        is None
    )
    for key, value in [
        ("body", "!"),
        ("body", "A" * 1398108),
        ("redacted", 1),
        ("status", True),
        ("elapsed_seconds", float("inf")),
        ("elapsed_seconds", -1),
    ]:
        changed = dict(saved["transport"], **{key: value})
        with pytest.raises(ValueError):
            runner.restore_transport(changed)


@pytest.mark.parametrize("index", [0, 1])
def test_uncreated_generation_intent_preserves_reservation_and_exports(
    runner, live_inputs, monkeypatch, index
):
    packet, path, raw = live_inputs
    original = records.FollowupRunStore.write_record

    def fail_before_creation(self, name, data):
        if name == f"attempts/{index:03}.json":
            raise OSError("synthetic disk full before file creation")
        return original(self, name, data)

    monkeypatch.setattr(records.FollowupRunStore, "write_record", fail_before_creation)
    fake = FakeTransport()
    summary = runner.run_live(packet, raw, path, lambda: fake)
    assert summary["state"] == "uncertain"
    assert summary["reserved_micro_usd"] == (index + 1) * 2688
    assert len(fake.calls) == 384 + index
    assert not (path / f"attempts/{index:03}.json").exists()
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    rows = runner.export_run(packet, path)["captures"]
    assert all(row["status"] == "completed" for row in rows[:index])
    assert all(row["status"] == "missing" for row in rows[index:])
    assert {p: p.read_bytes() for p in before} == before
    terminal = path / "terminal.json"
    changed = json.loads(terminal.read_text())
    changed["reserved_micro_usd"] += 2688
    terminal.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        runner.export_run(packet, path)


@pytest.mark.parametrize("field", ["started_at", "ended_at"])
@pytest.mark.parametrize(
    "stamp", ["not-a-time", "", "2026-10-15T12:00:00", "2026-10-15T12:00:00+01:00"]
)
def test_transport_timestamps_require_utc_syntax(runner, live_inputs, field, stamp):
    packet, path, raw = live_inputs
    execution = records.validate_execution(raw, records.build_manifest(packet), path)
    saved = runner.retained_result(provider_result(), execution, generation=True)
    saved["transport"][field] = stamp
    with pytest.raises(ValueError):
        runner.restore_transport(saved["transport"])


def test_historical_transport_timestamps_remain_valid(runner, live_inputs):
    packet, path, raw = live_inputs
    execution = records.validate_execution(raw, records.build_manifest(packet), path)
    result = replace(
        provider_result(), started_at="2000-01-01T00:00:00Z", ended_at="2000-01-01T00:00:01+00:00"
    )
    saved = runner.retained_result(result, execution, generation=True)
    assert runner.restore_transport(saved["transport"]) == result
