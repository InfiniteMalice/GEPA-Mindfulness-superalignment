"""GEPA dual-path evaluation integration utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig, DualPathTrace
from mindful_trace_gepa.deception.dual_path_runner import load_scenarios, run_dual_path_batch
from mindful_trace_gepa.storage import read_jsonl


def evaluate_gepa_model(
    model_callable: Callable[..., str],
    scenarios_path: str = "datasets/dual_path/data.jsonl",
    output_dir: str = "results/dual_path",
    model_id: str = "gepa-model",
) -> list[DualPathTrace]:
    """Evaluate a model with dual-path scenarios and write traces to disk."""

    path = Path(scenarios_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenarios_path}")
    records = read_jsonl(path)
    scenarios = load_scenarios(records)
    config = DualPathRunConfig(model_id=model_id, log_dir=output_dir)
    return run_dual_path_batch(scenarios, model_callable, config)


def enumerate_checkpoint_metadata(
    checkpoints_dir: str,
    scenarios_path: str = "datasets/dual_path/data.jsonl",
    output_dir: str = "results/dual_path",
) -> list[dict[str, Any]]:
    """Enumerate checkpoint directories and return metadata for future evaluation."""

    ckpt_path = Path(checkpoints_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    checkpoints = sorted(path for path in ckpt_path.iterdir() if path.is_dir())
    if not checkpoints:
        logging.getLogger(__name__).warning(
            "No checkpoint subdirectories found in %s",
            ckpt_path,
        )
    summaries: list[dict[str, Any]] = []
    run_timestamp = datetime.now(timezone.utc).isoformat()
    for checkpoint in checkpoints:
        summaries.append(
            {
                "checkpoint": checkpoint.name,
                "timestamp": run_timestamp,
                "scenarios_path": scenarios_path,
                "output_dir": output_dir,
            }
        )
    return summaries


__all__ = ["evaluate_gepa_model", "enumerate_checkpoint_metadata"]
