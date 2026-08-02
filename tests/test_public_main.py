"""Public package entry-point contracts."""

from __future__ import annotations

import mindful_trace_gepa


def test_exported_main_forwards_explicit_argv(monkeypatch) -> None:
    """Embedding callers must be able to bypass process-global command-line arguments."""
    observed: list[list[str] | None] = []

    def fake_cli_main(argv: list[str] | None = None) -> int:
        observed.append(argv)
        return 17

    monkeypatch.setattr(mindful_trace_gepa, "cli_main", fake_cli_main)
    argv = ["rl", "doctor"]

    assert mindful_trace_gepa.main(argv) == 17
    assert observed == [argv]
