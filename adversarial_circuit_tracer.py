"""Deprecated entry point for dual-path circuit tracing."""

# NOTE: New implementation lives in src/dual_path_circuit_tracer.py; keep this file as a thin shim.

from __future__ import annotations

import warnings

from src.dual_path_circuit_tracer import main

warnings.warn(
    "adversarial_circuit_tracer is deprecated; use src.dual_path_circuit_tracer instead.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    raise SystemExit(main())
