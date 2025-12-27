"""Deprecated entry point for dual-path circuit tracing."""

from __future__ import annotations

from mindful_trace_gepa.dual_path_circuit_tracer import main

# NOTE: New implementation lives in mindful_trace_gepa.dual_path_circuit_tracer.


if __name__ == "__main__":
    raise SystemExit(main())
