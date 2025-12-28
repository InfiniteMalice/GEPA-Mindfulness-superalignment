"""Deprecated adversarial challenge utilities (use dual_path instead)."""

from __future__ import annotations

import warnings

from .dual_path import DualPathProbeScenario, iterate_dual_path_pool, sample_dual_path_batch

AdversarialScenario = DualPathProbeScenario

warnings.warn(
    "gepa_mindfulness.core.adversarial is deprecated; use dual_path instead.",
    DeprecationWarning,
    stacklevel=2,
)

iterate_adversarial_pool = iterate_dual_path_pool
sample_adversarial_batch = sample_dual_path_batch


__all__ = [
    "AdversarialScenario",
    "iterate_adversarial_pool",
    "iterate_dual_path_pool",
    "sample_adversarial_batch",
    "sample_dual_path_batch",
]
