"""Deprecated entry point for dual-path circuit tracing."""

from __future__ import annotations

import warnings

from mindful_trace_gepa.dual_path_circuit_tracer import main

# NOTE: New implementation lives in mindful_trace_gepa.dual_path_circuit_tracer.


if __name__ == "__main__":
    warnings.warn(
        "src/dual_path_circuit_tracer.py is deprecated. "
        "Use mindful_trace_gepa.dual_path_circuit_tracer instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main())
