"""Deprecated entry point for dual-path evaluation integration."""

# NOTE: New implementation lives in gepa_dual_path_integration.py.
# Keep this file as a thin shim.

from __future__ import annotations

import warnings
from typing import Any

from gepa_dual_path_integration import (
    enumerate_checkpoint_metadata as _enumerate_checkpoint_metadata,
)
from gepa_dual_path_integration import evaluate_gepa_model as _evaluate_gepa_model


def evaluate_gepa_model(*args: Any, **kwargs: Any) -> Any:
    warnings.warn(
        "gepa_adversarial_integration is deprecated; use gepa_dual_path_integration instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _evaluate_gepa_model(*args, **kwargs)


def enumerate_checkpoint_metadata(*args: Any, **kwargs: Any) -> Any:
    warnings.warn(
        "gepa_adversarial_integration is deprecated; use gepa_dual_path_integration instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _enumerate_checkpoint_metadata(*args, **kwargs)


__all__ = ["evaluate_gepa_model", "enumerate_checkpoint_metadata"]
