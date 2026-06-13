"""Tests for optional dependency import behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindful_trace_gepa.utils.imports import optional_import


def test_optional_import_returns_none_for_missing_module() -> None:
    assert optional_import("definitely_missing_optional_dependency") is None


def test_optional_import_does_not_mask_runtime_import_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "broken_optional.py"
    module_path.write_text("raise RuntimeError('programming bug')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(RuntimeError, match="programming bug"):
        optional_import("broken_optional")
