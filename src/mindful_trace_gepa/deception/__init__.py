"""Deception utilities."""

from .dual_path_core import DualPathRunConfig, DualPathScenario, DualPathTrace
from .dual_path_runner import (
    build_prompt,
    load_scenarios,
    run_dual_path_batch,
    run_dual_path_scenario,
)
from .fingerprints import DeceptionFingerprint, FingerprintCollector
from .score import score_deception, summarize_deception_sources

__all__ = [
    "score_deception",
    "summarize_deception_sources",
    "DeceptionFingerprint",
    "FingerprintCollector",
    "DualPathRunConfig",
    "DualPathScenario",
    "DualPathTrace",
    "build_prompt",
    "load_scenarios",
    "run_dual_path_batch",
    "run_dual_path_scenario",
]
