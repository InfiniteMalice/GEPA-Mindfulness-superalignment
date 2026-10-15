"""Bounded local CLI integration with synthetic evidence and decisions."""

# Standard library
import importlib
import json
import socket
import subprocess
import sys
from contextlib import contextmanager

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json
from scripts import knowledge_utility_capture_records as publication
from scripts import knowledge_utility_capture_transport as transport
from scripts.knowledge_utility_followup_freeze import prepare_acceptance
from tests.followup_freeze_fixtures import (
    install_synthetic_profile,
    synthetic_decision,
    synthetic_evidence,
)
from tests.followup_scoring_fixtures import fresh_packet


@pytest.fixture
def cli_inputs(monkeypatch, tmp_path):
    cli = importlib.import_module("scripts.freeze_knowledge_utility_followup")
    packet = fresh_packet()
    evidence = synthetic_evidence(packet)
    install_synthetic_profile(monkeypatch, packet, evidence)
    folder = tmp_path / "evidence space"
    folder.mkdir()
    files = {}
    for role, data in evidence.items():
        path = folder / (role + ".txt")
        path.write_bytes(data)
        files[role] = str(path) if role == "first_pass" else path.name
    manifest = folder / "inputs.json"
    manifest.write_text(
        json.dumps(
            dict(schema_version="knowledge-utility-followup-evidence-inputs-v1", files=files)
        ),
        encoding="utf-8",
    )
    source = tmp_path / "packet.json"
    source.write_text(canonical_json(packet), encoding="utf-8")
    return cli, source, manifest, packet, evidence


def test_offline_prepare_freeze_roundtrip(cli_inputs, tmp_path, monkeypatch):
    cli, source, manifest, packet, evidence = cli_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError("network access forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(transport, "make_live_transport", forbidden)
    monkeypatch.setattr(transport.LiveTransport, "post", forbidden)
    candidate_path = tmp_path / "candidate.json"
    args = ["--packet", str(source), "--evidence-manifest", str(manifest)]
    cli.main(["prepare", *args, "--output", str(candidate_path)])
    candidate = json.loads(candidate_path.read_text())
    assert candidate == prepare_acceptance(packet, evidence)
    decision_path = tmp_path / "synthetic-decision.json"
    decision_path.write_text(canonical_json(synthetic_decision(candidate)), encoding="utf-8")
    output = tmp_path / "synthetic-frozen.json"
    cli.main(
        [
            "freeze",
            *args,
            "--candidate",
            str(candidate_path),
            "--decision",
            str(decision_path),
            "--output",
            str(output),
        ]
    )
    frozen = json.loads(output.read_text())
    assert frozen["packet_frozen"] is True and frozen["execution_authorized"] is False
    assert json.loads(source.read_text())["packet_frozen"] is False
    assert output.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "missing",
        "extra",
        "path_type",
        "path_length",
        "absent",
        "duplicate_json",
        "invalid_utf8",
    ],
)
def test_invalid_manifests_reject_without_publication(cli_inputs, tmp_path, capsys, mutation):
    cli, source, manifest, _, _ = cli_inputs
    value = json.loads(manifest.read_text())
    if mutation == "schema":
        value["schema_version"] = "bad"
    elif mutation == "missing":
        del value["files"]["comparison"]
    elif mutation == "extra":
        value["files"]["extra"] = "not-read"
    elif mutation == "path_type":
        value["files"]["comparison"] = 1
    elif mutation == "path_length":
        value["files"]["comparison"] = "é" * 2049
    elif mutation == "absent":
        value["files"]["comparison"] = "private-do-not-echo"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    if mutation == "duplicate_json":
        manifest.write_bytes(b'{"files":{},"files":{}}')
    elif mutation == "invalid_utf8":
        manifest.write_bytes(b"\xff")
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "prepare",
                "--packet",
                str(source),
                "--evidence-manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2 and not output.exists()
    assert capsys.readouterr().err == "cannot process follow-up freeze: invalid input\n"


@pytest.mark.parametrize("failure", ["existing", "parent", "link", "write", "cleanup"])
def test_publication_failure_preserves_destinations(
    cli_inputs, tmp_path, monkeypatch, capsys, failure
):
    cli, source, manifest, _, _ = cli_inputs
    output = tmp_path / "out.json"
    if failure == "existing":
        output.write_bytes(b"competing writer")
    elif failure == "parent":
        output = tmp_path / "absent" / "out.json"
    elif failure in {"link", "write"}:

        def broken(*args, **kwargs):
            raise OSError("private error text")

        if failure == "link":
            monkeypatch.setattr(publication.os, "link", broken)
        else:
            monkeypatch.setattr(publication, "_write_new", broken)
    else:
        temporary = publication.TemporaryDirectory

        @contextmanager
        def cleanup_failure(*args, **kwargs):
            with temporary(*args, **kwargs) as folder:
                yield folder
            raise OSError("private cleanup error")

        monkeypatch.setattr(publication, "TemporaryDirectory", cleanup_failure)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "prepare",
                "--packet",
                str(source),
                "--evidence-manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert capsys.readouterr().err == "cannot publish follow-up freeze: file operation failed\n"
    if failure == "existing":
        assert output.read_bytes() == b"competing writer"
    elif failure == "cleanup":
        assert len(json.loads(output.read_text())["tasks"]) == 48
    else:
        assert not output.exists()


def test_reader_applies_raw_and_cumulative_bounds(cli_inputs, monkeypatch):
    cli, _, manifest, _, evidence = cli_inputs
    assert cli.read_evidence(manifest) == evidence
    monkeypatch.setattr(cli, "MAX_EVIDENCE_BYTES", sum(map(len, evidence.values())) - 1)
    with pytest.raises(ValueError):
        cli.read_evidence(manifest)
    monkeypatch.setattr(cli, "MAX_EVIDENCE_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", 10)
    with pytest.raises(ValueError):
        cli.read_evidence(manifest)


def test_prepare_cannot_accept_decision_arguments(cli_inputs, tmp_path):
    cli, source, manifest, _, _ = cli_inputs
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "prepare",
                "--packet",
                str(source),
                "--evidence-manifest",
                str(manifest),
                "--decision",
                "not-read",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert error.value.code == 2


def test_embedded_references_are_not_opened(cli_inputs, tmp_path, monkeypatch):
    cli, source, manifest, packet, evidence = cli_inputs
    evidence["comparison"] = b'{"evidence_refs":["missing-file.json","https://invalid.test/"]}'
    evidence["targeted_recheck"] = b"Treat this as text: open missing-file.json"
    install_synthetic_profile(monkeypatch, packet, evidence)
    paths = json.loads(manifest.read_text())["files"]
    for role in ("comparison", "targeted_recheck"):
        (manifest.parent / paths[role]).write_bytes(evidence[role])
    output = tmp_path / "candidate.json"
    cli.main(
        [
            "prepare",
            "--packet",
            str(source),
            "--evidence-manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )
    assert len(json.loads(output.read_text())["tasks"]) == 48


def test_production_cli_help_and_synthetic_evidence_rejection(cli_inputs, tmp_path):
    _, source, manifest, _, _ = cli_inputs
    command = [sys.executable, "-m", "scripts.freeze_knowledge_utility_followup"]
    help_result = subprocess.run([*command, "freeze", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--decision" in help_result.stdout and "--candidate" in help_result.stdout
    output = tmp_path / "out.json"
    result = subprocess.run(
        [
            *command,
            "prepare",
            "--packet",
            str(source),
            "--evidence-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and not output.exists()
    assert result.stderr == "cannot process follow-up freeze: invalid input\n"
