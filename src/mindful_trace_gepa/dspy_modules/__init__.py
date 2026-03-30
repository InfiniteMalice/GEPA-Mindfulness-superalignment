"""DSPy module exports for Mindful Trace GEPA.

The DSPy-related modules are optional dependencies. Importing them directly at
package import time would raise ``ImportError`` when the optional dependency is
missing, preventing consumers from accessing the package entirely. To avoid
that, we attempt to import the DSPy helpers lazily and expose ``None``
placeholders when they are unavailable. This mirrors the behaviour of the CLI
entry points which check for optional dependencies at runtime.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from mindful_trace_gepa.import_utils import optional_repo_module

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULES_PATH = REPO_ROOT / "modules"


def _optional(name: str) -> Any:
    try:
        module = import_module(name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name == name:
            return None
        if missing_name in {"dspy", "dspy.teleprompt"}:
            return None
        raise
    return module


def _optional_repo_module(name: str) -> Any:
    inserted_path = str(MODULES_PATH)
    added_path = False
    if MODULES_PATH.is_dir() and inserted_path not in sys.path:
        sys.path.insert(0, inserted_path)
        added_path = True
    try:
        return _optional(name)
    finally:
        if added_path and inserted_path in sys.path:
            sys.path.remove(inserted_path)


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
DSPyCompiler = GEPACompiler

ALL_SIGNATURES = getattr(signatures, "ALL_SIGNATURES", None) if signatures else None
semantic_package = optional_repo_module("semantic_intent_robustness", MODULES_PATH)
SEMANTIC_PIPELINE_REGISTRY = (
    getattr(semantic_package, "SEMANTIC_PIPELINE_REGISTRY", None) if semantic_package else None
)

__all__ = [
    "GEPAChain",
    "DualPathGEPAChain",
    "GEPAChainResult",
    "ModuleResult",
    "ensure_dspy_enabled",
    "GEPACompiler",
    "DSPyCompiler",
    "create_gepa_metric",
    "ALL_SIGNATURES",
    "SEMANTIC_PIPELINE_REGISTRY",
]
