"""Frozen pilot capture cannot replace attempts or grant evidence authority."""

# Standard library
import base64
import json
import multiprocessing
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

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
    "data",
    [b'{"a":1,"a":2}', b"\xff", b"[" * 33 + b"]" * 33, b" " * (4 * 1024 * 1024 + 1)],
    ids=["duplicate", "utf8", "depth", "bytes"],
)
def test_bounded_json_rejects_bad_input(records, tmp_path, data):
    path = tmp_path / "input.json"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        records.read_bounded_json(path)


def _create_run_worker(directory, manifest, raw, queue):
    from scripts import knowledge_utility_capture_records as records

    try:
        checked = records.validate_execution(raw, manifest, Path(directory), "a" * 40)
        records.RunStore.create(Path(directory), checked, manifest)
        queue.put("created")
    except FileExistsError:
        queue.put("exists")


def test_run_directory_is_single_use(records, tmp_path):
    manifest = records.build_manifest(prepare_pilot())
    directory = tmp_path / "run"
    raw = execution(manifest, directory)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    workers = [
        ctx.Process(target=_create_run_worker, args=(directory, manifest, raw, queue))
        for _ in range(2)
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
    assert (directory / "execution-started.json").is_file()
    checked = records.validate_execution(raw, manifest, directory, "a" * 40)
    with pytest.raises(FileExistsError):
        records.RunStore.create(directory / ".." / "run", checked, manifest)


def test_intent_is_durable_before_dispatch(records, tmp_path, monkeypatch):
    manifest = records.build_manifest(prepare_pilot())
    directory = tmp_path / "run"
    checked = records.validate_execution(
        execution(manifest, directory), manifest, directory, "a" * 40
    )
    store = records.RunStore.create(directory, checked, manifest)
    called = []

    def broken_sync(fd):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(records.os, "fsync", broken_sync)
    with pytest.raises(OSError):
        store.write_record("attempts/000.json", {"reservation_micro_usd": 2688})
        called.append("dispatch")
    assert called == []
    assert (directory / "attempts/000.json").exists()
    with pytest.raises(FileExistsError):
        store.write_record("attempts/000.json", {})


@pytest.mark.parametrize("relative", ["../escape.json", "/escape.json", "attempts/999.json"])
def test_run_store_rejects_uncontrolled_filenames(records, tmp_path, relative):
    manifest = records.build_manifest(prepare_pilot())
    directory = tmp_path / "run"
    checked = records.validate_execution(
        execution(manifest, directory), manifest, directory, "a" * 40
    )
    store = records.RunStore.create(directory, checked, manifest)
    with pytest.raises(ValueError):
        store.write_record(relative, {})


def test_publication_preserves_competing_destination(records, tmp_path, monkeypatch):
    destination = tmp_path / "captures.json"

    def competing_link(source, target):
        Path(target).write_text("other writer", encoding="utf-8")
        raise FileExistsError("simulated race")

    monkeypatch.setattr(records.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        records.publish_json(destination, {"captures": []})
    assert destination.read_text(encoding="utf-8") == "other writer"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.fixture
def runner():
    from scripts import capture_knowledge_utility_pilot

    return capture_knowledge_utility_pilot


def provider_response(text='{"answer":"6"}', status="completed", content=None):
    return dict(
        object="response",
        id="resp_authored",
        status=status,
        model="gpt-5.4-mini-2026-03-17",
        error=None,
        incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
        usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        output=[
            dict(
                type="message",
                id="msg_authored",
                role="assistant",
                status=status,
                content=(
                    content
                    if content is not None
                    else [{"type": "output_text", "text": text, "annotations": []}]
                ),
            )
        ],
    )


def result_for(response, **changes):
    from scripts.knowledge_utility_capture_transport import TransportResult

    fields = dict(
        status=200,
        request_id="req_authored",
        body=json.dumps(response).encode(),
        error_code=None,
        redacted=False,
        incomplete=False,
        started_at="2026-10-12T00:00:00Z",
        ended_at="2026-10-12T00:00:01Z",
        elapsed_seconds=1.0,
    )
    fields.update(changes)
    return TransportResult(**fields)


@pytest.mark.parametrize(
    "response,want,text",
    [
        (provider_response(), "completed", '{"answer":"6"}'),
        (provider_response("bad JSON"), "completed", "bad JSON"),
        (provider_response("I refuse"), "completed", "I refuse"),
        (provider_response(content=[]), "completed", ""),
        (provider_response(status="incomplete"), "truncated", '{"answer":"6"}'),
        (
            provider_response(
                status="incomplete",
                content=[
                    {"type": "refusal", "refusal": "refused"},
                    {"type": "output_text", "text": '{"answer":"6"}', "annotations": []},
                ],
            ),
            "refused",
            '{"answer":"6"}',
        ),
        (
            provider_response(
                content=[{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}]
            ),
            "failed",
            None,
        ),
    ],
    ids=[
        "correct",
        "malformed-answer",
        "refusal-wording",
        "empty",
        "truncated",
        "refusal-precedence",
        "multiple-text",
    ],
)
def test_normalization_precedence(runner, response, want, text):
    normalized = runner.normalize_result(result_for(response))
    assert normalized["status"] == want
    assert normalized["response_text"] == text
    assert (normalized["reason"] is None) == (want == "completed")


@pytest.mark.parametrize(
    "fault",
    [
        "model",
        "usage",
        "boolean",
        "input-limit",
        "output-limit",
        "tool",
        "unknown",
        "redacted",
        "http",
        "envelope",
    ],
)
def test_normalization_rejects_protocol_violations(runner, fault):
    response = provider_response()
    changes = {}
    if fault == "model":
        response["model"] = "different"
    elif fault == "usage":
        response["usage"] = None
    elif fault in ("boolean", "input-limit", "output-limit"):
        key = "output_tokens" if fault == "output-limit" else "input_tokens"
        response["usage"][key] = {"boolean": True, "input-limit": 2049, "output-limit": 257}[fault]
    elif fault in ("tool", "unknown"):
        response["output"].append({"type": "function_call" if fault == "tool" else "surprise"})
    elif fault == "redacted":
        changes["redacted"] = True
    elif fault == "http":
        changes.update(status=500, error_code="http_error")
    else:
        changes["body"] = b'{"status":"completed","status":"incomplete"}'
    assert runner.normalize_result(result_for(response, **changes))["status"] == "failed"


@pytest.mark.parametrize("chars,want", [(2048, "completed"), (2049, "failed")])
def test_export_counts_unicode_bytes(runner, chars, want):
    assert (
        runner.normalize_result(result_for(provider_response("\U0001f600" * chars)))["status"]
        == want
    )


def captured_run(records, directory, count=1):
    packet = prepare_pilot()
    manifest = records.build_manifest(packet)
    checked = records.validate_execution(
        execution(manifest, directory), manifest, directory, "a" * 40
    )
    store = records.RunStore.create(directory, checked, manifest)
    for req in manifest["requests"][:count]:
        seq = req["sequence"]
        store.write_record(
            f"attempts/{seq:03}.json",
            dict(
                sequence=seq,
                slot_id=req["slot_id"],
                request_digest=req["request_digest"],
                attempt_number=1,
                reservation_micro_usd=2688,
                cumulative_micro_usd=(seq + 1) * 2688,
                utc="2026-10-12T00:00:00Z",
            ),
        )
    return packet, manifest, store


def test_export_preserves_uncertain_attempt(records, runner, tmp_path):
    packet, manifest, store = captured_run(records, tmp_path / "run")
    envelope = runner.export_run(packet, store.root)
    assert len(envelope["captures"]) == 72
    assert envelope["captures"][0]["status"] == "failed"
    assert envelope["captures"][1]["status"] == "missing"
    (store.root / "attempts/000.json").write_bytes(b'{"sequence":')
    envelope = runner.export_run(packet, store.root)
    assert envelope["captures"][0]["status"] == "failed"
    assert envelope["captures"][0]["reason"] == "uncertain_attempt"


@pytest.mark.parametrize("fault", ["identity", "orphan", "sequence", "metadata"])
def test_export_rejects_corrupt_identity(records, runner, tmp_path, fault):
    packet, manifest, store = captured_run(records, tmp_path / "run")
    value = asdict(result_for(provider_response()))
    value["body"] = base64.b64encode(value["body"]).decode()
    request = manifest["requests"][0]
    store.write_record(
        "results/000.json",
        dict(
            sequence=0,
            slot_id=request["slot_id"],
            request_digest=request["request_digest"],
            transport=value,
        ),
    )
    if fault == "identity":
        path = store.root / "results/000.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["run_id"] = "foreign"
        path.write_text(json.dumps(raw), encoding="utf-8")
    elif fault == "orphan":
        (store.root / "attempts/000.json").unlink()
    elif fault == "sequence":
        (store.root / "attempts/099.json").write_text("{}", encoding="utf-8")
    else:
        (store.root / "requests.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.export_run(packet, store.root)

