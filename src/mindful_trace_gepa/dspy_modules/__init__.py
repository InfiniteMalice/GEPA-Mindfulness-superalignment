"""DSPy module exports for Mindful Trace GEPA.

The DSPy-related modules are optional dependencies. Importing them directly at
package import time would raise ``ImportError`` when the optional dependency is
missing, preventing consumers from accessing the package entirely. To avoid
that, we attempt to import the DSPy helpers lazily and expose ``None``
placeholders when they are unavailable. This mirrors the behaviour of the CLI
entry points which check for optional dependencies at runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _optional(name: str) -> Any:
    try:
        module = import_module(name)
    except ModuleNotFoundError:
        return None
    except Exception:
        return None
    return module


pipeline = _optional("mindful_trace_gepa.dspy_modules.pipeline")
compile = _optional("mindful_trace_gepa.dspy_modules.compile")
signatures = _optional("mindful_trace_gepa.dspy_modules.signatures")

GEPAChain = getattr(pipeline, "GEPAChain", None) if pipeline else None
DualPathGEPAChain = getattr(pipeline, "DualPathGEPAChain", None) if pipeline else None
GEPAChainResult = getattr(pipeline, "GEPAChainResult", None) if pipeline else None
ModuleResult = getattr(pipeline, "ModuleResult", None) if pipeline else None
ensure_dspy_enabled = getattr(pipeline, "ensure_dspy_enabled", None) if pipeline else None

GEPACompiler = getattr(compile, "GEPACompiler", None) if compile else None
create_gepa_metric = getattr(compile, "create_gepa_metric", None) if compile else None

ALL_SIGNATURES = getattr(signatures, "ALL_SIGNATURES", None) if signatures else None

__all__ = [
    "GEPAChain",
    "DualPathGEPAChain",
    "GEPAChainResult",
    "ModuleResult",
    "ensure_dspy_enabled",
    "GEPACompiler",
    "create_gepa_metric",
    "ALL_SIGNATURES",
]
