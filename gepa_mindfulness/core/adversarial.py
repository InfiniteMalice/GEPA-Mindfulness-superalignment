"""Deprecated adversarial challenge utilities (use dual_path instead)."""

# NOTE: New implementation lives in dual_path.py; keep this file as a thin shim.

from __future__ import annotations

import random
import warnings
from collections.abc import Iterable

from .dual_path import DualPathProbeScenario, iterate_dual_path_pool, sample_dual_path_batch

AdversarialScenario = DualPathProbeScenario

_DEPRECATION_MSG = (
    "{name} is deprecated; use {replacement} from gepa_mindfulness.core.dual_path instead"
)


def sample_adversarial_batch(
    batch_size: int,
    rng: random.Random | None = None,
) -> list[AdversarialScenario]:
    warnings.warn(
        _DEPRECATION_MSG.format(
            name="sample_adversarial_batch",
            replacement="sample_dual_path_batch",
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return sample_dual_path_batch(batch_size, rng=rng)


def iterate_adversarial_pool() -> Iterable[AdversarialScenario]:
    warnings.warn(
        _DEPRECATION_MSG.format(
            name="iterate_adversarial_pool",
            replacement="iterate_dual_path_pool",
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    yield from iterate_dual_path_pool()


__all__ = [
    "AdversarialScenario",
    "iterate_adversarial_pool",
    "sample_adversarial_batch",
]
