"""CLI mode boundaries and exclusive publication, with no real provider access."""

# Standard library
import json
import subprocess
import sys
from pathlib import Path

# Third-party
import pytest

# Local
from scripts import capture_knowledge_utility_followup as runner
from scripts import knowledge_utility_capture_records as publication
from scripts import knowledge_utility_followup_capture_records as records
from tests.followup_capture_fixtures import (
    COMMIT,
    NOW,
    FakeTransport,
    approved_packet,
    execution_record,
    price_record,
    seal_execution,
)

OFFLINE_ERROR = "cannot process follow-up capture: invalid input or file operation"
LIVE_ERROR = "follow-up capture stopped; preserve run directory and inspect intents"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(approved_packet()), encoding="utf-8")
    output = tmp_path / "output.json"
    monkeypatch.setattr(runner, "implementation_commit", lambda: COMMIT)
    monkeypatch.setattr(runner, "utc_now", lambda: NOW.isoformat())
    return packet, output


def rejected(args, capsys, message):
    with pytest.raises(SystemExit) as error:
        runner.main([str(arg) for arg in args])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "synthetic-sensitive-value" not in stderr


@pytest.mark.parametrize("priced", [False, True])
def test_offline_prepare_never_constructs_transport(inputs, monkeypatch, priced):
    packet, output = inputs
    monkeypatch.setattr(runner, "make_live_transport", lambda p: pytest.fail("credential access"))
    args = ["--packet", str(packet), "--output", str(output)]
    if priced:
        prices = output.with_name("prices.json")
        prices.write_text(json.dumps(price_record()))
        args += ["--prices", str(prices)]
    runner.main(args)
    value = json.loads(output.read_text())
    assert value["token_counts"] == "unknown"
    assert len(value["manifest"]["requests"]) == 384
    if priced:
        assert value["reservation_estimate"]["roster_micro_usd"] == 1032192
    else:
        assert value["reservation_estimate"] is None


def live_args(inputs):
    packet, output = inputs
    run_dir = output.with_name("run")
    raw = execution_record(records.build_manifest(json.loads(packet.read_text())), run_dir)
    approval = output.with_name("execution.json")
    approval.write_text(json.dumps(raw))
    args = [
        "--live",
        "--packet",
        str(packet),
        "--execution",
        str(approval),
        "--run-dir",
        str(run_dir),
        "--credential-file",
        "unused-synthetic-path",
    ]
    return args, raw, approval, run_dir


def test_live_cli_fake_roster_completes(inputs, monkeypatch):
    args, raw, approval, path = live_args(inputs)
    fake = FakeTransport()

    def factory(credential):
        assert (path / "execution-started.json").is_file()
        return fake

    monkeypatch.setattr(runner, "make_live_transport", factory)
    runner.main(args)
    assert len(fake.calls) == 768
    assert json.loads((path / "terminal.json").read_text())["state"] == "finished"


@pytest.mark.parametrize("fault", ["binding", "dirty", "stale", "existing", "oversize"])
def test_invalid_live_input_precedes_credential_access(inputs, monkeypatch, capsys, fault):
    args, raw, approval, path = live_args(inputs)
    monkeypatch.setattr(runner, "make_live_transport", lambda p: pytest.fail("credential access"))
    if fault == "dirty":

        def dirty():
            raise ValueError("synthetic-sensitive-value")

        monkeypatch.setattr(runner, "implementation_commit", dirty)
    if fault == "binding":
        raw["manifest_digest"] = "bad"
    if fault == "stale":
        raw["prices"]["verified_at"] = "2026-10-01T12:00:00Z"
    if fault == "existing":
        path.mkdir()
    approval.write_text(json.dumps(seal_execution(raw)))
    if fault == "oversize":
        approval.write_bytes(b" " * (1024 * 1024 + 1))
    rejected(args, capsys, LIVE_ERROR)


@pytest.mark.parametrize("state", ["stopped", "uncertain"])
def test_nonfinished_live_status_exits_two(inputs, monkeypatch, capsys, state):
    args, _, _, _ = live_args(inputs)
    monkeypatch.setattr(runner, "run_live", lambda *a: {"state": state})
    rejected(args, capsys, LIVE_ERROR)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--live"],
        ["--export"],
        ["--live", "--export"],
        ["--output", "output", "--credential-file", "key"],
        ["--output", "output", "--execution", "approval"],
        ["--output", "output", "--run-dir", "run"],
        ["--export", "--run-dir", "run", "--output", "out", "--prices", "prices"],
        ["--export", "--run-dir", "run", "--output", "out", "--credential-file", "key"],
        [
            "--live",
            "--run-dir",
            "run",
            "--execution",
            "approval",
            "--credential-file",
            "key",
            "--output",
            "out",
        ],
    ],
)
def test_wrong_modes_reject_before_file_access(inputs, monkeypatch, capsys, extra):
    monkeypatch.setattr(runner, "read_json_object", lambda *a, **k: pytest.fail("file access"))
    rejected(["--packet", "absent", *extra], capsys, "error:")


@pytest.mark.parametrize(
    "raw",
    [b'{"x":1,"x":2}', b"[" * 33, b" " * (4 * 1024 * 1024 + 1)],
    ids=["duplicates", "depth", "oversize"],
)
def test_invalid_packet_input_is_controlled(inputs, capsys, raw):
    packet, output = inputs
    packet.write_bytes(raw)
    rejected(["--packet", packet, "--output", output], capsys, OFFLINE_ERROR)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["existing", "parent", "link", "race", "cleanup", "size"])
def test_publication_preserves_destinations(inputs, monkeypatch, capsys, fault):
    packet, output = inputs
    if fault == "existing":
        output.write_bytes(b"original")
    if fault == "parent":
        output = output.parent / "missing" / "out.json"
    if fault in ("link", "race"):

        def broken_link(source, destination):
            if fault == "race":
                Path(destination).write_bytes(b"winner")
            raise OSError("synthetic-sensitive-value")

        monkeypatch.setattr(publication.os, "link", broken_link)
    if fault == "cleanup":
        original = publication.TemporaryDirectory

        class CleanupFailure(original):
            def __exit__(self, *args):
                super().__exit__(*args)
                raise OSError("synthetic-sensitive-value")

        monkeypatch.setattr(publication, "TemporaryDirectory", CleanupFailure)
    if fault == "size":
        monkeypatch.setattr(runner, "prepare_capture", lambda *a: {"x": "x" * (4 * 1024 * 1024)})
    rejected(["--packet", packet, "--output", output], capsys, OFFLINE_ERROR)
    if fault in ("existing", "race"):
        assert output.read_bytes() == (b"original" if fault == "existing" else b"winner")
    elif fault == "cleanup":
        assert json.loads(output.read_text())["token_counts"] == "unknown"
    else:
        assert not output.exists()


def test_offline_subprocess_prepare_export_and_help(inputs, monkeypatch):
    packet, output = inputs
    path = output.with_name("empty-run")
    manifest = records.build_manifest(json.loads(packet.read_text()))
    execution = records.validate_execution(execution_record(manifest, path), manifest, path)
    records.FollowupRunStore.create(path, execution, manifest)
    before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner, "make_live_transport", lambda p: pytest.fail("credential access"))
    runner.main(
        ["--export", "--packet", str(packet), "--run-dir", str(path), "--output", str(output)]
    )
    expected = json.loads(output.read_text())
    assert all(row["status"] == "missing" for row in expected["captures"])
    commands = [
        ["--help"],
        ["--packet", str(packet), "--output", str(output.with_name("prepared.json"))],
        [
            "--export",
            "--packet",
            str(packet),
            "--run-dir",
            str(path),
            "--output",
            str(output.with_name("exported.json")),
        ],
    ]
    for args in commands:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.capture_knowledge_utility_followup", *args],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assert json.loads(output.with_name("exported.json").read_text()) == expected
    assert {p: p.read_bytes() for p in before} == before
