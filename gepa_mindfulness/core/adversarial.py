"""Deprecated adversarial challenge utilities (use dual_path instead)."""

# NOTE: New implementation lives in dual_path.py; keep this file as a thin shim.

from __future__ import annotations

from collections.abc import Iterable

from .dual_path import DualPathProbeScenario, iterate_dual_path_pool, sample_dual_path_batch

AdversarialScenario = DualPathProbeScenario


def sample_adversarial_batch(batch_size: int) -> list[AdversarialScenario]:
    return sample_dual_path_batch(batch_size)


def iterate_adversarial_pool() -> Iterable[AdversarialScenario]:
    yield from iterate_dual_path_pool()


__all__ = [
    "AdversarialScenario",
    "iterate_adversarial_pool",
    "sample_adversarial_batch",
]
