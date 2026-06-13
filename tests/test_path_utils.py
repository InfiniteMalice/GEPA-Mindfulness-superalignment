"""Tests for filesystem safety helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindful_trace_gepa import path_utils


def test_atomic_write_text_preserves_existing_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "summary.json"
    target.write_text("original", encoding="utf-8")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(path_utils.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        path_utils.atomic_write_text(target, "new content")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_atomic_write_json_writes_pretty_json(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"

    path_utils.atomic_write_json(target, {"z": [1, 2], "a": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"z": [1, 2], "a": True}
    assert target.read_text(encoding="utf-8").endswith("\n")
