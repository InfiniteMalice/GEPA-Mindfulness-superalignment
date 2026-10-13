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


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", True),
        ("temperature", 1.0),
        ("max_output_tokens", 256.0),
        ("store", 0),
        ("stream", 0),
    ],
)
def test_execution_rejects_profile_type_substitutions(records, tmp_path, field, value):
    """Equal Python values cannot replace the approved JSON profile types."""
    manifest = records.build_manifest(prepare_pilot())
    run_dir = tmp_path / "run"
    raw = deepcopy(execution(manifest, run_dir))
    raw["profile"][field] = value
    with pytest.raises(ValueError, match="execution binding mismatch"):
        records.validate_execution(raw, manifest, run_dir, "a" * 40)
    assert not run_dir.exists()


def test_execution_accepts_reordered_profile_keys(records, tmp_path):
    """JSON object key order does not change an otherwise exact approval."""
    manifest = records.build_manifest(prepare_pilot())
    run_dir = tmp_path / "run"
    raw = deepcopy(execution(manifest, run_dir))
    raw["profile"] = dict(reversed(list(raw["profile"].items())))
    checked = records.validate_execution(raw, manifest, run_dir, "a" * 40)
    assert checked.raw["profile"] == manifest["profile"]
    assert not run_dir.exists()


def test_accepted_unknown_counting_keeps_generation_budget(records, runner, tmp_path):
    """Accepting counting uncertainty does not refund or expand generation reservations."""
    packet = prepare_pilot()
    manifest = records.build_manifest(packet)
    directory = tmp_path / "run"
    raw = execution(manifest, directory, "0.002688")
    raw["prices"]["counting_policy"] = "unverified_accepted"
    checked = records.validate_execution(raw, manifest, directory, "a" * 40)
    assert checked.prices.counting_policy == "unverified_accepted"
    fake = FakeTransport(directory)
    summary = runner.run_live(packet, checked, directory, lambda: fake)
    assert fake.endpoints == ["responses/input_tokens"] * 72 + ["responses"]
    assert summary["reason"] == "budget_exhausted"
    assert summary["reserved_micro_usd"] == 2688
    assert summary["budget_scope"] == "generation_only"
    assert summary["counting_policy"] == "unverified_accepted"
    assert summary["counting_cost_usd"] is None
    assert summary["total_billed_cost_usd"] is None
    terminal = json.loads((directory / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["counting_cost_usd"] is None
    for seq in range(72):
        counted = json.loads(
            (directory / f"counts/{seq:03}-result.json").read_text(encoding="utf-8")
        )
        assert counted["usage"] == {"input_tokens": fake.count_value, "output_tokens": 0}
        assert counted["estimated_cost_usd"] is None
        assert counted["billed_cost_usd"] is None
        assert counted["cost_scope"] == "counting"
    captures = runner.export_run(packet, directory)["captures"]
    assert captures[0]["status"] == "completed"
    assert sum(row["status"] == "missing" for row in captures) == 71


def test_counting_usage_cannot_be_priced_as_generation(records, runner, tmp_path):
    """Unexpected counting usage is retained without inventing a counting price."""
    packet, checked, directory = live_fixture(records, tmp_path)
    retained = runner._retained_result(result_for(provider_response()), checked, generation=False)
    assert retained["usage"]["input_tokens"] == 100
    assert retained["estimated_cost_usd"] is None
    assert retained["billed_cost_usd"] is None
    assert retained["cost_scope"] == "counting"


def test_unknown_counting_dry_run_labels_generation_estimate(records, runner, tmp_path):
    """An offline estimate must not imply that the total bill is capped."""
    packet_path, prices_path = tmp_path / "packet.json", tmp_path / "prices.json"
    output = tmp_path / "requests.json"
    packet_path.write_text(json.dumps(prepare_pilot()), encoding="utf-8")
    raw = prices()
    raw["counting_policy"] = "unverified_accepted"
    prices_path.write_text(json.dumps(raw), encoding="utf-8")
    runner.main(
        ["--packet", str(packet_path), "--prices", str(prices_path), "--output", str(output)]
    )
    estimate = json.loads(output.read_text(encoding="utf-8"))["reservation_estimate"]
    assert estimate["roster_micro_usd"] == 193536
    assert estimate["budget_scope"] == "generation_only"
    assert estimate["counting_cost_usd"] is None


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


class FakeTransport:
    def __init__(self, directory, *, count_value=100, generation_fault=None):
        self.directory = directory
        self.count_value = count_value
        self.generation_fault = generation_fault
        self.endpoints = []
        self.packet = prepare_pilot()

    def post(self, endpoint, payload):
        self.endpoints.append(endpoint)
        if endpoint == "responses/input_tokens":
            seq = self.endpoints.count(endpoint) - 1
            assert (self.directory / f"counts/{seq:03}-intent.json").is_file()
            return result_for(
                {
                    "object": "response.input_tokens",
                    "input_tokens": self.count_value,
                    "usage": {"input_tokens": self.count_value, "output_tokens": 0},
                }
            )
        seq = self.endpoints.count("responses") - 1
        assert self.endpoints.count("responses/input_tokens") == 72
        assert (self.directory / f"attempts/{seq:03}.json").is_file()
        if self.generation_fault == "failed":
            return result_for({}, error_code="transport_error")
        if self.generation_fault == "truncated":
            return result_for(provider_response(status="incomplete"))
        if self.generation_fault == "refused":
            return result_for(provider_response(content=[{"type": "refusal", "refusal": "no"}]))
        packet = self.packet
        actor = packet["actor_requests"][seq]
        assert payload["input"] == actor["messages"]
        answers = {row["case_id"]: row["expected_answer"] for row in packet["evaluator_cases"]}
        return result_for(provider_response(json.dumps({"answer": answers[actor["case_id"]]})))


def live_fixture(records, tmp_path, budget="5"):
    packet = prepare_pilot()
    manifest = records.build_manifest(packet)
    directory = tmp_path / "live-run"
    raw = execution(manifest, directory, budget)
    checked = records.validate_execution(raw, manifest, directory, "a" * 40)
    return packet, checked, directory


def test_all_counts_precede_generation_and_full_scorer_roundtrip(records, runner, tmp_path):
    from scripts.score_knowledge_utility_pilot import score_pilot

    packet, checked, directory = live_fixture(records, tmp_path)
    fake = FakeTransport(directory)
    summary = runner.run_live(packet, checked, directory, lambda: fake)
    assert summary["state"] == "finished"
    assert fake.endpoints == ["responses/input_tokens"] * 72 + ["responses"] * 72
    report = score_pilot(packet, runner.export_run(packet, directory))
    assert all(row["successes"] == 18 for row in report["conditions"].values())
    assert all(not row["verified"] for row in report["comparisons"].values())
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["confers_authority"] is False and report["optimizer_input"] is False
    result = json.loads((directory / "results/000.json").read_text(encoding="utf-8"))
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert result["estimated_cost_usd"] == "0.000165"
    assert result["billed_cost_usd"] is None
    with pytest.raises(FileExistsError):
        runner.run_live(packet, checked, directory, lambda: pytest.fail("must not construct"))


@pytest.mark.parametrize("value", [2049, True, None, -1])
def test_count_failure_never_generates(records, runner, tmp_path, value):
    packet, checked, directory = live_fixture(records, tmp_path)
    fake = FakeTransport(directory, count_value=value)
    summary = runner.run_live(packet, checked, directory, lambda: fake)
    assert summary["state"] == "stopped"
    assert fake.endpoints == ["responses/input_tokens"]
    assert all(
        row["status"] == "missing" for row in runner.export_run(packet, directory)["captures"]
    )


@pytest.mark.parametrize("budget,requests", [("0.002688", 1), ("0.002687", 0), ("0", 0)])
def test_budget_failure_retains_full_roster(records, runner, tmp_path, budget, requests):
    packet, checked, directory = live_fixture(records, tmp_path, budget)
    fake = FakeTransport(directory)
    summary = runner.run_live(packet, checked, directory, lambda: fake)
    assert summary["state"] == "stopped"
    assert fake.endpoints.count("responses") == requests
    assert summary["reserved_micro_usd"] == requests * 2688
    rows = runner.export_run(packet, directory)["captures"]
    assert len(rows) == 72
    assert sum(row["status"] == "missing" for row in rows) == 72 - requests


@pytest.mark.parametrize("fault,count", [("failed", 1), ("refused", 72), ("truncated", 72)])
def test_generation_failure_stops_but_censoring_continues(records, runner, tmp_path, fault, count):
    packet, checked, directory = live_fixture(records, tmp_path)
    fake = FakeTransport(directory, generation_fault=fault)
    summary = runner.run_live(packet, checked, directory, lambda: fake)
    assert fake.endpoints.count("responses") == count
    assert summary["reserved_micro_usd"] == count * 2688
    assert runner.export_run(packet, directory)["captures"][0]["status"] == fault


def test_result_write_failure_keeps_reservation_and_stops(records, runner, tmp_path, monkeypatch):
    packet, checked, directory = live_fixture(records, tmp_path)
    fake = FakeTransport(directory)
    original = records.RunStore.write_record

    def fail_result(self, relative, data):
        if relative == "results/000.json":
            raise OSError("simulated write failure")
        return original(self, relative, data)

    monkeypatch.setattr(records.RunStore, "write_record", fail_result)
    summary = runner.run_live(packet, checked, directory, lambda: fake)
    assert summary["state"] == "uncertain"
    assert summary["reserved_micro_usd"] == 2688
    assert fake.endpoints.count("responses") == 1
    assert runner.export_run(packet, directory)["captures"][0]["status"] == "failed"


def test_offline_modes_never_construct_transport_or_read_key(
    records, runner, tmp_path, monkeypatch
):
    packet, manifest, store = captured_run(records, tmp_path / "run", count=0)
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    monkeypatch.setattr(runner, "make_live_transport", lambda *args: pytest.fail("credential read"))
    output = tmp_path / "dry.json"
    runner.main(["--packet", str(packet_path), "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8"))["manifest"] == manifest
    exported = tmp_path / "export.json"
    runner.main(
        [
            "--export",
            "--packet",
            str(packet_path),
            "--run-dir",
            str(store.root),
            "--output",
            str(exported),
        ]
    )
    assert len(json.loads(exported.read_text(encoding="utf-8"))["captures"]) == 72
    with pytest.raises(SystemExit) as error:
        runner.main(["--export", "--live", "--packet", str(packet_path)])
    assert error.value.code == 2


def test_live_cli_rejects_binding_or_dirty_code(records, runner, tmp_path, monkeypatch):
    packet, checked, directory = live_fixture(records, tmp_path)
    packet_path, record_path = tmp_path / "packet.json", tmp_path / "execution.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    record_path.write_text(json.dumps(checked.raw), encoding="utf-8")
    monkeypatch.setattr(runner, "make_live_transport", lambda *args: pytest.fail("credential read"))
    monkeypatch.setattr(runner, "implementation_commit", lambda: "b" * 40)
    with pytest.raises(SystemExit) as error:
        runner.main(
            [
                "--live",
                "--packet",
                str(packet_path),
                "--execution",
                str(record_path),
                "--run-dir",
                str(directory),
                "--credential-file",
                str(tmp_path / "no-key"),
            ]
        )
    assert error.value.code == 2
    assert not directory.exists()


def test_zero_budget_with_zero_rates_never_generates(records, runner, tmp_path):
    packet = prepare_pilot()
    manifest = records.build_manifest(packet)
    directory = tmp_path / "run"
    raw = execution(manifest, directory, "0")
    raw["prices"] = prices("0", "0")
    checked = records.validate_execution(raw, manifest, directory, "a" * 40)
    fake = FakeTransport(directory)
    runner.run_live(packet, checked, directory, lambda: fake)
    assert "responses" not in fake.endpoints


def test_text_size_failure_has_explicit_reason(runner):
    normalized = runner.normalize_result(result_for(provider_response("x" * 8193)))
    assert normalized["status"] == "failed"
    assert normalized["reason"] == "response_text_too_large"


@pytest.mark.parametrize("phase", ["write", "flush", "close"])
def test_intent_io_failure_prevents_dispatch_and_preserves_evidence(
    records, runner, tmp_path, monkeypatch, phase
):
    packet, checked, directory = live_fixture(records, tmp_path)
    fake = FakeTransport(directory)
    original_open = Path.open

    class BrokenFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()
            if phase == "close":
                raise OSError("authored close failure")

        def write(self, data):
            self.stream.write(data)
            if phase == "write":
                raise OSError("authored write failure")

        def flush(self):
            if phase == "flush":
                raise OSError("authored flush failure")
            self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    def patched_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path == directory / "attempts/000.json" and mode == "xb":
            return BrokenFile(stream)
        return stream

    monkeypatch.setattr(Path, "open", patched_open)
    summary = runner.run_live(packet, checked, directory, lambda: fake)
    assert "responses" not in fake.endpoints
    assert (directory / "attempts/000.json").exists()
    assert summary["reserved_micro_usd"] == 2688
    assert runner.export_run(packet, directory)["captures"][0]["status"] == "failed"


@pytest.mark.parametrize("tokens", [0, 2048])
def test_count_boundary_permits_one_budgeted_generation(records, runner, tmp_path, tokens):
    packet, checked, directory = live_fixture(records, tmp_path, "0.002688")
    fake = FakeTransport(directory, count_value=tokens)
    runner.run_live(packet, checked, directory, lambda: fake)
    assert fake.endpoints.count("responses") == 1


def test_incomplete_terminal_does_not_block_recovery(records, runner, tmp_path):
    packet, manifest, store = captured_run(records, tmp_path / "run")
    (store.root / "terminal.json").write_bytes(b'{"run_id":')
    rows = runner.export_run(packet, store.root)["captures"]
    assert len(rows) == 72
    assert rows[0]["status"] == "failed" and rows[0]["reason"] == "uncertain_attempt"
    assert rows[1]["status"] == "missing"
    (store.root / "terminal.json").write_text('{"run_id":"foreign"}', encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        runner.export_run(packet, store.root)


@pytest.mark.parametrize("failed_file", ["execution.json", "terminal.json"])
def test_live_io_diagnostic_preserves_run_location_and_uncertainty(
    records, runner, tmp_path, monkeypatch, capsys, failed_file
):
    packet, checked, directory = live_fixture(records, tmp_path)
    packet_path, record_path = tmp_path / "packet.json", tmp_path / "execution.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    record_path.write_text(json.dumps(checked.raw), encoding="utf-8")
    monkeypatch.setattr(runner, "implementation_commit", lambda: "a" * 40)

    # A transport setup failure reaches terminal persistence without a real provider.
    def no_transport(path):
        raise OSError("sk-test-sentinel")

    monkeypatch.setattr(runner, "make_live_transport", no_transport)
    original = records.RunStore.write_record

    def partial_write(self, relative, data):
        if relative == failed_file:
            (self.root / relative).write_bytes(b'{"partial":')
            raise OSError("sk-test-sentinel")
        return original(self, relative, data)

    monkeypatch.setattr(records.RunStore, "write_record", partial_write)
    with pytest.raises(SystemExit) as error:
        runner.main(
            [
                "--live",
                "--packet",
                str(packet_path),
                "--execution",
                str(record_path),
                "--run-dir",
                str(directory),
                "--credential-file",
                str(tmp_path / "no-key"),
            ]
        )
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert str(directory) in diagnostic and "uncertain" in diagnostic
    assert "sk-test-sentinel" not in diagnostic
    assert (directory / failed_file).exists()
