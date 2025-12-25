"""Deprecated entry point for dual-path evaluation integration."""

# NOTE: New implementation lives in gepa_dual_path_integration.py; keep this file as a thin shim.

from __future__ import annotations

from gepa_dual_path_integration import evaluate_gepa_model, track_training_progress

__all__ = ["evaluate_gepa_model", "track_training_progress"]
