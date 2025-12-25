"""Deprecated entry point for dual-path circuit tracing."""

# NOTE: New implementation lives in src/dual_path_circuit_tracer.py; keep this file as a thin shim.

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    target = Path(__file__).resolve().parent / "src" / "dual_path_circuit_tracer.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
