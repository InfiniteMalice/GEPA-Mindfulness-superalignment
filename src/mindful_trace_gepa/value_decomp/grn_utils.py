"""Shared utilities for applying GRN normalization in value decomposition."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from ..utils.imports import optional_import
from .deep_value_spaces import to_float_list, to_tensor

logger = logging.getLogger(__name__)
torch = optional_import("torch")


@lru_cache(maxsize=None)
def _get_grn_instance(dim: int = -1):
    if torch is None:
        return None
    grn_module = optional_import("mindful_trace_gepa.train.grn")
    if grn_module is None:
        return None
    build_grn = getattr(grn_module, "build_grn", None)
    if build_grn is None:
        return None
    return build_grn({"enabled": True, "dim": dim})


def apply_grn_vector(vector: List[float], *, dim: int = -1) -> List[float]:
    """Apply optional GRN normalization to a flat feature vector."""

    if torch is None:
        return vector

    grn = _get_grn_instance(dim)
    if grn is None:
        return vector

    tensor = to_tensor(vector)
    if callable(grn):
        try:
            normalised = grn(tensor)  # type: ignore[operator]
            return to_float_list(normalised)
        except Exception:
            logger.debug("GRN normalization failed; using raw vector", exc_info=True)
    return vector


__all__ = ["apply_grn_vector"]
