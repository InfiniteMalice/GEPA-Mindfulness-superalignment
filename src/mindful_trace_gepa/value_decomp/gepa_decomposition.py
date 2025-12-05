"""Decompose GEPA scores into deep and shallow contributions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .deep_value_spaces import DeepValueVector, ShallowPreferenceVector
from .grn_utils import apply_grn_vector

logger = logging.getLogger(__name__)


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
        if len(deep) != len(self.alpha) or len(shallow) != len(self.beta):
            raise ValueError(
                "Probe length mismatch: "
                f"deep={len(deep)} vs alpha={len(self.alpha)}, "
                f"shallow={len(shallow)} vs beta={len(self.beta)}",
            )
        deep_score = sum(a * b for a, b in zip(deep, self.alpha))
        shallow_score = sum(a * b for a, b in zip(shallow, self.beta))
        return deep_score, shallow_score


_DEFAULT_PROBE: Optional[LinearValueProbe] = None


def register_default_probe(probe: Optional[LinearValueProbe]) -> None:
    global _DEFAULT_PROBE
    _DEFAULT_PROBE = probe


def reset_default_probe() -> None:
    """Reset the global default probe (useful for tests)."""

    global _DEFAULT_PROBE
    _DEFAULT_PROBE = None


def _reduce_scalar(values: Iterable[float]) -> float:
    """Compute mean of values; returns 0.0 when values is empty."""
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

    # Probe selection: explicit > global default > auto-generated from sizes
    probe = (
        probe
        or _DEFAULT_PROBE
        or LinearValueProbe.from_sizes(len(deep_values.ORDER), len(shallow_prefs.ORDER))
    )
    deep_list = deep_values.to_list()
    shallow_list = shallow_prefs.to_list()

    if use_grn:
        feature_vector = apply_grn_vector(deep_list + shallow_list)
        expected_len = len(deep_list) + len(shallow_list)
        if len(feature_vector) != expected_len:
            logger.warning(
                "GRN output len %d != expected %d; using raw features",
                len(feature_vector),
                expected_len,
            )
            feature_vector = deep_list + shallow_list
        split = len(deep_list)
        deep_list = feature_vector[:split]
        shallow_list = feature_vector[split:]

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
    "reset_default_probe",
]
