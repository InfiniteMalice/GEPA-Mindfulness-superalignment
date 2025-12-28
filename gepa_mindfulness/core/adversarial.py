"""Deprecated adversarial challenge utilities (use dual_path instead)."""

from __future__ import annotations

from .dual_path import DualPathProbeScenario, iterate_dual_path_pool, sample_dual_path_batch

AdversarialScenario = DualPathProbeScenario


__all__ = [
    "AdversarialScenario",
    "iterate_dual_path_pool",
    "sample_dual_path_batch",
]
