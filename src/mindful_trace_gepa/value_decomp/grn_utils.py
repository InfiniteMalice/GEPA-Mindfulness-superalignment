"""Shared utilities for applying GRN normalization in value decomposition."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ..utils.imports import optional_import
from .deep_value_spaces import to_float_list, to_tensor

logger = logging.getLogger(__name__)
torch = optional_import("torch")


@lru_cache(maxsize=8)
def _get_grn_instance(dim: int = -1) -> Any:
    if torch is None:
        return None
    grn_module = optional_import("mindful_trace_gepa.train.grn")
    if grn_module is None:
        return None
    build_grn = getattr(grn_module, "build_grn", None)
    if build_grn is None:
        return None
    try:
        return build_grn({"enabled": True, "dim": dim})
    except Exception as exc:
        logger.debug(
            "Failed to build GRN instance (%s); falling back to raw features",
            type(exc).__name__,
            exc_info=True,
        )
        return None


def apply_grn_vector(vector: list[float], *, dim: int = -1) -> list[float]:
    """Apply optional GRN normalization to a flat feature vector."""

    if torch is None:
        return vector

    grn = _get_grn_instance(dim)
    if grn is None:
        return vector

    try:
        tensor = to_tensor(vector)
    except Exception:
        logger.debug("Failed to convert vector for GRN; returning raw vector", exc_info=True)
        return vector
    if callable(grn):
        try:
            if hasattr(tensor, "dim") and tensor.dim() == 1:  # type: ignore[operator]
                tensor = tensor.unsqueeze(0)  # type: ignore[assignment]
            normalised = grn(tensor)  # type: ignore[operator]
            if hasattr(normalised, "dim") and normalised.dim() > 1:  # type: ignore[operator]
                normalised = normalised.squeeze(0)  # type: ignore[assignment]
            return to_float_list(normalised)
        except Exception:
            logger.debug("GRN normalization failed; using raw vector", exc_info=True)
    return vector


__all__ = ["apply_grn_vector"]
