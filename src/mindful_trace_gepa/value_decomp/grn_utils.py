"""Shared utilities for applying GRN normalization in value decomposition."""

from __future__ import annotations

import logging
from typing import List

from ..utils.imports import optional_import
from .deep_value_spaces import to_float_list, to_tensor

logger = logging.getLogger(__name__)
torch = optional_import("torch")
_UNSET = object()
_cached_grn_module: object = _UNSET
_cached_grn: object = _UNSET
# Cache is not synchronised; concurrent init may build multiple GRN instances.


def apply_grn_vector(vector: List[float]) -> List[float]:
    """Apply optional GRN normalization to a flat feature vector."""

    if torch is None:
        return vector

    global _cached_grn_module, _cached_grn
    if _cached_grn_module is _UNSET:
        _cached_grn_module = optional_import("mindful_trace_gepa.train.grn")

    grn_module = _cached_grn_module
    if grn_module is None:
        _cached_grn = None
        return vector

    build_grn = getattr(grn_module, "build_grn", None)
    if build_grn is None:
        _cached_grn = None
        return vector

    if _cached_grn is _UNSET:
        _cached_grn = build_grn({"enabled": True, "dim": -1})

    if _cached_grn is None:
        return vector
    grn = _cached_grn
    tensor = to_tensor(vector)
    if callable(grn):
        try:
            normalised = grn(tensor)  # type: ignore[operator]
            return to_float_list(normalised)
        except Exception:
            logger.debug("GRN normalization failed; using raw vector", exc_info=True)
    return vector


__all__ = ["apply_grn_vector"]
