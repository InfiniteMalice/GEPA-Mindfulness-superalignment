"""Proxy package exposing ``src/mindful_trace_gepa`` without installation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
_MODULE_PATH = _SRC_ROOT / "mindful_trace_gepa" / "__init__.py"
if not _MODULE_PATH.exists():
    raise ImportError("mindful_trace_gepa source package not found")

_SPEC = importlib.util.spec_from_file_location(__name__, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - safety guard
    raise ImportError("Unable to load mindful_trace_gepa source module")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[__name__] = _MODULE
_SPEC.loader.exec_module(_MODULE)
