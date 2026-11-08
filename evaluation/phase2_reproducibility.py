"""Default-off offline audit of historical research sources; no scientific admission."""

# Standard library
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

# Local
from evaluation.phase2_reporting import _load_json
from gepa_mindfulness.verification.research_archive import MAX_SOURCE_BYTES


def _path(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or any(c in value for c in (":", "\\", "\x00", "\n"))
    ):
        raise ValueError("invalid historical path")
    parts = PurePosixPath(value)
    if parts.is_absolute() or any(p in (".", "..") for p in value.split("/")):
        raise ValueError("historical path must be relative without traversal")
    return value


def historical_blob(root: Path, commit: str, path: str) -> bytes:
    """Read an exact bounded Git blob without shell, filters, replacement refs or network."""
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("historical commit requires a full SHA-1")
    spec = commit + ":" + _path(path)
    command = ["git", "--no-lazy-fetch", "--no-replace-objects", "-C", str(root), "cat-file"]
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""}
    try:
        size = int(
            subprocess.check_output([*command, "-s", spec], stderr=subprocess.DEVNULL, env=env)
        )
        if not 0 <= size <= MAX_SOURCE_BYTES:
            raise ValueError("historical blob exceeds source byte bound")
        payload = subprocess.check_output(
            [*command, "blob", spec], stderr=subprocess.DEVNULL, env=env
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ValueError("historical blob unavailable") from exc
    if len(payload) != size:
        raise ValueError("historical blob size changed")
    return payload


def _hash(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid source digest")
    return value


def audit_stages(root: Path, pins: Any, *, enabled: bool = False) -> dict:
    """Verify retained bytes at pinned commits; authored summaries remain authored claims."""
    if enabled is not True:
        raise ValueError("historical research audit is disabled")
    if type(pins) is not list or not 1 <= len(pins) <= 16:
        raise ValueError("expected bounded stage pins")
    stages = []
    seen = set()
    for pin in pins:
        forms = ({"stage", "commit", "manifest", "manifest_digest"}, {"stage", "commit", "sha256"})
        if type(pin) is not dict or set(pin) not in forms:
            raise ValueError("invalid stage pin fields")
        stage = pin["stage"]
        if type(stage) is not int or not 0 <= stage <= 9 or stage in seen:
            raise ValueError("invalid or duplicate stage")
        seen.add(stage)
        prefix = ""
        if "manifest" in pin:
            payload = historical_blob(root, pin["commit"], pin["manifest"])
            if hashlib.sha256(payload).hexdigest() != _hash(pin["manifest_digest"]):
                raise ValueError("historical manifest digest mismatch")
            manifest = _load_json(payload, "json")
            if type(manifest) is not dict:
                raise ValueError("invalid historical manifest")
            mapping = manifest.get("sha256", manifest.get("files"))
            if "files" in manifest:
                parent = str(PurePosixPath(pin["manifest"]).parent)
                prefix = "" if parent == "." else parent + "/"
        else:
            manifest = {}
            mapping = pin["sha256"]
        if type(mapping) is not dict or not 1 <= len(mapping) <= 256:
            raise ValueError("manifest requires bounded source hashes")
        files = []
        for relative, digest in sorted(mapping.items()):
            path = _path(prefix + _path(relative))
            source = historical_blob(root, pin["commit"], path)
            actual = hashlib.sha256(source).hexdigest()
            if actual != _hash(digest):
                raise ValueError("historical source digest mismatch: " + path)
            files.append({"path": path, "sha256": actual, "bytes": len(source)})
        stages.append(
            {
                **pin,
                "verified_files": len(files),
                "files": files,
                "retained_summary": {
                    k: manifest[k]
                    for k in ("full_suite", "probe", "final_console_summary", "runs")
                    if k in manifest
                },
            }
        )
    return {
        "schema_version": "phase2-historical-audit-v1",
        "maturity": "experimental",
        "diagnostic_status": "diagnostic",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "scientific_verification": "NOT_ESTABLISHED",
        "stages": stages,
        "limitations": [
            "Hashes establish retained byte identity, not truth of summaries.",
            "No historical experiment or model capability test is rerun.",
            "Pins require host custody; this audit cannot grant authority.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.enabled:
        parser.error("research audit requires --enabled")
    with args.pins.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("pins exceed source byte bound")
    result = audit_stages(args.repository_root, _load_json(raw, "json"), enabled=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
