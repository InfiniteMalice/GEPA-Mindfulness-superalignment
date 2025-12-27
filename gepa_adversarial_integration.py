"""Deprecated entry point for dual-path evaluation integration."""

# NOTE: New implementation lives in gepa_dual_path_integration.py.
# Keep this file as a thin shim.

from __future__ import annotations

import warnings

from gepa_dual_path_integration import enumerate_checkpoint_metadata, evaluate_gepa_model

warnings.warn(
    "gepa_adversarial_integration is deprecated; use gepa_dual_path_integration instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["evaluate_gepa_model", "enumerate_checkpoint_metadata"]
