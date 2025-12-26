"""Deprecated entry point for dual-path circuit tracing."""

# NOTE: New implementation lives in dual_path_circuit_tracer.py; keep this file as a thin shim.

from __future__ import annotations

from src.dual_path_circuit_tracer import main

if __name__ == "__main__":
    raise SystemExit(main())
