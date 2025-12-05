"""Decompose GEPA scores into deep and shallow contributions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..utils.imports import optional_import
from .deep_value_spaces import (
    DeepValueVector,
    ShallowPreferenceVector,
    _to_float_list,
    _to_tensor,
)

logger = logging.getLogger(__name__)
torch = optional_import("torch")


@dataclass
class GepaDecomposition:
    deep_contribution: float
    shallow_contribution: float
    residual: float


@dataclass
class LinearValueProbe:
    alpha: List[float]
    beta: List[float]

    @classmethod
    def from_sizes(
        cls, deep_size: int, shallow_size: int, scale: float = 1.0
    ) -> "LinearValueProbe":
        return cls(alpha=[scale] * deep_size, beta=[scale * 0.5] * shallow_size)

    def apply(
        self, deep_values: Sequence[float], shallow_values: Sequence[float]
    ) -> tuple[float, float]:
        deep = list(deep_values)
        shallow = list(shallow_values)
        deep_score = sum(a * b for a, b in zip(deep, self.alpha))
        shallow_score = sum(a * b for a, b in zip(shallow, self.beta))
        return deep_score, shallow_score


_DEFAULT_PROBE: Optional[LinearValueProbe] = None


def register_default_probe(probe: Optional[LinearValueProbe]) -> None:
    global _DEFAULT_PROBE
    _DEFAULT_PROBE = probe


def _flatten_feature_vector(
    deep_values: DeepValueVector, shallow_prefs: ShallowPreferenceVector
) -> List[float]:
    return deep_values.to_list() + shallow_prefs.to_list()


def _reduce_scalar(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    return total / count if count else 0.0


def decompose_gepa_score(
    gepa_vector: Iterable[float],
    deep_values: DeepValueVector,
    shallow_prefs: ShallowPreferenceVector,
    use_grn: bool = False,
    probe: Optional[LinearValueProbe] = None,
) -> GepaDecomposition:
    """Estimate deep vs shallow contributions to a GEPA scalar."""

    probe = (
        probe
        or _DEFAULT_PROBE
        or LinearValueProbe.from_sizes(len(deep_values.ORDER), len(shallow_prefs.ORDER))
    )
    deep_list = deep_values.to_list()
    shallow_list = shallow_prefs.to_list()

    if use_grn:
        feature_vector = _flatten_feature_vector(deep_values, shallow_prefs)
        grn_module = optional_import("mindful_trace_gepa.train.grn")
        if grn_module is not None and hasattr(grn_module, "build_grn") and torch is not None:
            grn = grn_module.build_grn({"enabled": True, "dim": -1})
            tensor = _to_tensor(feature_vector)
            try:
                normalised = grn(tensor)  # type: ignore[operator]
                feature_vector = _to_float_list(normalised)
                deep_list = feature_vector[: len(deep_values.ORDER)]
                shallow_list = feature_vector[len(deep_values.ORDER) :]
            except Exception:
                logger.debug("GRN normalization failed, using raw features", exc_info=True)

    deep_contrib, shallow_contrib = probe.apply(deep_list, shallow_list)
    gepa_scalar = _reduce_scalar(gepa_vector)
    residual = gepa_scalar - (deep_contrib + shallow_contrib)
    return GepaDecomposition(
        deep_contribution=deep_contrib,
        shallow_contribution=shallow_contrib,
        residual=residual,
    )


__all__ = [
    "GepaDecomposition",
    "decompose_gepa_score",
    "LinearValueProbe",
    "register_default_probe",
]
