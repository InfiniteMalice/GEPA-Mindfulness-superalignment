"""Historical source audit remains offline and distinct from scientific verification."""

# Standard library
import hashlib
import json
import os
import subprocess
import sys

# Third-party
import pytest

# Local
from evaluation.phase2_reproducibility import audit_stages, historical_blob, main


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


def fixture_repo(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "source.txt").write_bytes(b"original")
    manifest = {"sha256": {"source.txt": hashlib.sha256(b"original").hexdigest()}}
    payload = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(payload)
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-m",
        "first",
    )
    rev = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "source.txt").write_bytes(b"later")
    return [
        {
            "stage": 3,
            "commit": rev,
            "manifest": "manifest.json",
            "manifest_digest": hashlib.sha256(payload).hexdigest(),
        }
    ]


def test_historical_blobs_are_checked_instead_of_mutable_checkout(tmp_path):
    pins = fixture_repo(tmp_path)
    report = audit_stages(tmp_path, pins, enabled=True)
    assert report["stages"][0]["verified_files"] == 1
    assert report["scientific_verification"] == "NOT_ESTABLISHED"
    assert report["optimizer_input"] is False
    with pytest.raises(ValueError, match="disabled"):
        audit_stages(tmp_path, pins)
    pins[0]["manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        audit_stages(tmp_path, pins, enabled=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("commit", "--help"),
        ("manifest", "../outside"),
        ("manifest", "/etc/passwd"),
        ("manifest", "x:y"),
    ],
)
def test_invalid_git_selectors_fail_before_git_execution(tmp_path, field, value):
    pins = fixture_repo(tmp_path)
    pins[0][field] = value
    with pytest.raises(ValueError):
        audit_stages(tmp_path, pins, enabled=True)


def test_cli_is_opt_in_and_never_overwrites_output(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pins = fixture_repo(repo)
    config = tmp_path / "pins.json"
    config.write_text(json.dumps(pins))
    output = tmp_path / "report.json"
    args = ["--repository-root", str(repo), "--pins", str(config), "--output", str(output)]
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()
    assert main(["--enabled", *args]) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--enabled", *args])
    assert output.read_bytes() == original


def test_inline_pins_cover_stages_without_original_json_manifests(tmp_path):
    pins = fixture_repo(tmp_path)
    inline = {
        "stage": 1,
        "commit": pins[0]["commit"],
        "sha256": {"source.txt": hashlib.sha256(b"original").hexdigest()},
    }
    assert audit_stages(tmp_path, [inline], enabled=True)["stages"][0]["verified_files"] == 1
    inline["sha256"]["source.txt"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        audit_stages(tmp_path, [inline], enabled=True)


def test_missing_promised_blob_never_invokes_transport(tmp_path, monkeypatch):
    fixture_repo(tmp_path)
    missing = "f" * 40
    tree = (
        subprocess.check_output(
            ["git", "-C", str(tmp_path), "mktree", "--missing"],
            input=("100644 blob " + missing + "\tmissing.json\n").encode(),
        )
        .decode()
        .strip()
    )
    commit = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.org",
                "commit-tree",
                tree,
            ],
            input=b"promised source\n",
        )
        .decode()
        .strip()
    )
    marker = tmp_path / "transport-was-invoked"
    helper = tmp_path / "git-remote-phase2canary"
    helper.write_text(
        "#!"
        + sys.executable
        + "\nfrom pathlib import Path\n"
        + "Path("
        + repr(str(marker))
        + ").write_text('attempted')\n"
        + "raise SystemExit(1)\n"
    )
    helper.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "phase2canary")
    monkeypatch.delenv("GIT_NO_LAZY_FETCH", raising=False)
    git(tmp_path, "config", "remote.origin.promisor", "true")
    git(tmp_path, "config", "remote.origin.url", "phase2canary::unused")
    with pytest.raises(ValueError, match="unavailable"):
        historical_blob(tmp_path, commit, "missing.json")
    assert not marker.exists(), "offline audit attempted a Git transport helper"


def test_root_level_files_manifest_uses_canonical_source_paths(tmp_path):
    pins = fixture_repo(tmp_path)
    manifest = tmp_path / "manifest.json"
    body = json.loads(manifest.read_text())
    raw = json.dumps({"files": body["sha256"]}).encode()
    manifest.write_bytes(raw)
    git(tmp_path, "add", "manifest.json")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-m",
        "root files manifest",
    )
    pins[0].update(
        commit=git(tmp_path, "rev-parse", "HEAD"), manifest_digest=hashlib.sha256(raw).hexdigest()
    )
    report = audit_stages(tmp_path, pins, enabled=True)
    assert report["stages"][0]["files"][0]["path"] == "source.txt"
