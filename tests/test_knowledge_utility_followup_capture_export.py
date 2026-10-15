"""Recovery trusts ordered durable evidence, never mutable result projections."""

# Standard library
import json
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest

# Local
from scripts import capture_knowledge_utility_followup as runner
from scripts import knowledge_utility_followup_capture_records as records
from scripts.score_knowledge_utility_followup import score_followup
from tests.followup_capture_fixtures import (
    COMMIT,
    NOW,
    FakeTransport,
    approved_packet,
    execution_record,
    provider_result,
)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    packet = approved_packet()
    path = tmp_path_factory.mktemp("export") / "run"
    raw = execution_record(records.build_manifest(packet), path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "implementation_commit", lambda: COMMIT)
        patch.setattr(runner, "utc_now", lambda: NOW.isoformat())
        assert runner.run_live(packet, raw, path, FakeTransport)["state"] == "finished"
    snapshot = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    return packet, path, snapshot


@pytest.fixture
def history(completed_run):
    packet, path, snapshot = completed_run
    yield packet, path
    # Restore only test mutations; the production exporter never repairs its source.
    for p in path.rglob("*"):
        if p.is_file() and p.relative_to(path) not in snapshot:
            p.unlink()
    for relative, data in snapshot.items():
        p = path / relative
        if p.is_symlink():
            p.unlink()
        if p.is_dir():
            p.rmdir()
        if not p.exists() or p.read_bytes() != data:
            p.write_bytes(data)


def edit(path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_full_export_is_read_only_and_historical(history, monkeypatch):
    packet, path = history
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner, "implementation_commit", lambda: pytest.fail("current HEAD"))
    monkeypatch.setattr(runner, "utc_now", lambda: pytest.fail("current clock"))
    envelope = runner.export_run(packet, path)
    assert envelope == runner.export_run(packet, path)
    assert len(envelope["captures"]) == 384
    assert [r["slot_id"] for r in envelope["captures"]] == [
        r["slot_id"] for r in packet["actor_requests"]
    ]
    assert all(r["response_text"] == '{"answer":"A"}' for r in envelope["captures"])
    report = score_followup(packet, envelope)
    assert report["packet_state"]["execution_authorized"] is False
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("fragment", [None, b"{", b"\xff"])
@pytest.mark.parametrize("kind", ["results/383.json", "attempts/383.json", "terminal.json"])
def test_final_partial_records_are_recoverable(history, fragment, kind):
    packet, path = history
    if kind.startswith("attempts"):
        (path / "results/383.json").unlink()
    if fragment is None:
        (path / kind).unlink()
    else:
        (path / kind).write_bytes(fragment)
    # A missing/partial terminal carries no authority over the durable prefix.
    if kind != "terminal.json":
        (path / "terminal.json").unlink()
    last = runner.export_run(packet, path)["captures"][-1]
    expected = "completed" if kind == "terminal.json" else "failed"
    if kind.startswith("attempts") and fragment is None:
        expected = "missing"
    assert last["status"] == expected
    if expected == "failed":
        assert last["reason"] == "uncertain_attempt"


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("results/000.json", "usage", {}),
        ("results/000.json", "provider", {}),
        ("results/000.json", "estimated_cost_usd", "0"),
        ("results/000.json", "billed_cost_usd", "0"),
        ("results/000.json", "returned_model", "other"),
        ("results/000.json", "sequence", False),
        ("results/000.json", "request_digest", "bad"),
        ("results/000.json", "slot_id", "bad"),
        ("results/000.json", "run_id", "bad"),
        ("results/000.json", "execution_digest", "bad"),
        ("attempts/000.json", "cumulative_micro_usd", 0),
        ("attempts/000.json", "reservation_micro_usd", True),
        ("counts/000-intent.json", "payload", {}),
        ("counts/000-result.json", "cost_scope", "generation"),
        ("terminal.json", "execution_digest", "bad"),
        ("execution-started.json", "extra", 1),
    ],
)
def test_complete_conflicting_records_reject(history, kind, field, value):
    packet, path = history
    edit(path / kind, **{field: value})
    with pytest.raises(ValueError):
        runner.export_run(packet, path)


@pytest.mark.parametrize(
    "damage",
    [
        "count_gap",
        "generation_gap",
        "orphan",
        "partial_intent",
        "late_after_uncertain",
        "late_after_failure",
        "generation_before_counts",
    ],
)
def test_invalid_prefixes_reject(history, damage):
    packet, path = history
    if damage == "count_gap":
        for name in ("counts/000-intent.json", "counts/000-result.json"):
            (path / name).unlink()
    if damage == "generation_gap":
        for name in ("attempts/000.json", "results/000.json"):
            (path / name).unlink()
    if damage == "orphan":
        (path / "attempts/000.json").unlink()
    if damage == "partial_intent":
        (path / "attempts/383.json").write_bytes(b"{")
    if damage == "late_after_uncertain":
        (path / "results/000.json").unlink()
    if damage == "generation_before_counts":
        (path / "counts/383-result.json").unlink()
    if damage == "late_after_failure":
        execution = records.validate_execution(
            json.loads((path / "execution.json").read_text())["execution"],
            records.build_manifest(packet),
            path,
        )
        failed = replace(provider_result(), error_code="deadline_exceeded")
        edit(
            path / "results/000.json", **runner.retained_result(failed, execution, generation=True)
        )
    with pytest.raises(ValueError):
        runner.export_run(packet, path)


@pytest.mark.parametrize("status", ["refused", "truncated"])
def test_retained_provider_status_is_preserved(history, status):
    packet, path = history
    execution = records.validate_execution(
        json.loads((path / "execution.json").read_text())["execution"],
        records.build_manifest(packet),
        path,
    )
    result = provider_result(status="incomplete")
    if status == "refused":
        value = json.loads(provider_result().body)
        value["output"][0]["content"] = [{"type": "refusal", "refusal": "no"}]
        result = replace(result, body=json.dumps(value).encode())
    edit(path / "results/383.json", **runner.retained_result(result, execution, generation=True))
    assert runner.export_run(packet, path)["captures"][-1]["status"] == status


@pytest.mark.parametrize("partial", [False, True])
def test_failed_or_partial_count_leaves_generations_missing(history, partial):
    packet, path = history
    for folder in ("attempts", "results"):
        for file in (path / folder).iterdir():
            file.unlink()
    (path / "terminal.json").unlink()
    target = path / "counts/383-result.json"
    if partial:
        target.write_bytes(b"{")
    else:
        execution = records.validate_execution(
            json.loads((path / "execution.json").read_text())["execution"],
            records.build_manifest(packet),
            path,
        )
        failed = replace(provider_result(), status=429, error_code="http_error")
        edit(target, **runner.retained_result(failed, execution, generation=False))
    rows = runner.export_run(packet, path)["captures"]
    assert all(row["status"] == "missing" for row in rows)
    assert all(row["reason"] == "count_failed" for row in rows)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b"[" * 33,
        b" " * (4 * 1024 * 1024 + 1),
        b"[]",
        b"{}",
        b'{"x":NaN}',
    ],
    ids=["duplicates", "depth", "oversize", "array", "fields", "nonfinite"],
)
def test_invalid_structured_records_are_not_uncertainty(history, raw):
    packet, path = history
    (path / "results/383.json").write_bytes(raw)
    (path / "terminal.json").unlink()
    with pytest.raises(ValueError):
        runner.export_run(packet, path)


@pytest.mark.parametrize("name", ["extra.json", "counts/384-intent.json", "results/stray"])
def test_unknown_tree_entries_reject(history, name):
    packet, path = history
    (path / name).write_bytes(b"{}")
    with pytest.raises(ValueError):
        runner.export_run(packet, path)


def test_directory_record_rejects(history):
    packet, path = history
    target = path / "results/383.json"
    target.unlink()
    target.mkdir()
    with pytest.raises(ValueError):
        runner.export_run(packet, path)


def test_symlink_rejects(history):
    packet, path = history
    target = path / "results/383.json"
    target.unlink()
    try:
        target.symlink_to(path / "results/000.json")
    except OSError:
        pytest.skip("OS denied test symlink creation")
    with pytest.raises(ValueError):
        runner.export_run(packet, path)


def test_reparse_attribute_and_read_errors_reject(history, monkeypatch):
    packet, path = history
    original = Path.lstat

    class Reparse:
        st_file_attributes = 0x400
        st_mode = 0o100644

    monkeypatch.setattr(Path, "lstat", lambda p: Reparse() if p == path else original(p))
    with pytest.raises(ValueError):
        runner.export_run(packet, path)
    monkeypatch.setattr(Path, "lstat", original)
    original_open = Path.open

    def unreadable(p, *args, **kwargs):
        if p == path / "results/383.json":
            raise OSError("synthetic permission error")
        return original_open(p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    with pytest.raises(OSError):
        runner.export_run(packet, path)
