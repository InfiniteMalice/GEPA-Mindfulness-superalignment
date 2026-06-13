"""Python 3.10 compatibility checks for runtime source."""

from __future__ import annotations

from pathlib import Path


def test_runtime_code_does_not_import_datetime_utc_constant() -> None:
    """datetime.UTC is Python 3.11+; CI still runs Python 3.10."""

    roots = [Path("gepa_mindfulness"), Path("src"), Path("modules"), Path("evaluation")]
    offenders = [
        str(path)
        for root in roots
        for path in root.rglob("*.py")
        if "from datetime import UTC" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
