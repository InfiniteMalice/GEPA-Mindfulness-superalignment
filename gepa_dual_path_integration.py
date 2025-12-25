"""GEPA dual-path evaluation integration utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig, DualPathTrace
from mindful_trace_gepa.deception.dual_path_runner import load_scenarios, run_dual_path_batch
from mindful_trace_gepa.storage import read_jsonl


def evaluate_gepa_model(
    model_callable: Callable[..., str],
    scenarios_path: str = "datasets/dual_path/data.jsonl",
    output_dir: str = "results/dual_path",
    model_id: str = "gepa-model",
) -> List[DualPathTrace]:
    """Evaluate a model with dual-path scenarios and write traces to disk."""

    records = read_jsonl(Path(scenarios_path))
    scenarios = load_scenarios(records)
    config = DualPathRunConfig(model_id=model_id, log_dir=output_dir)
    return run_dual_path_batch(scenarios, model_callable, config)


def track_training_progress(
    checkpoints_dir: str,
    scenarios_path: str = "datasets/dual_path/data.jsonl",
    output_dir: str = "results/dual_path",
) -> List[Dict[str, Any]]:
    """Return placeholder summaries for dual-path evaluations across checkpoints."""

    checkpoints = sorted(Path(checkpoints_dir).glob("*"))
    summaries: List[Dict[str, Any]] = []
    for checkpoint in checkpoints:
        summaries.append(
            {
                "checkpoint": checkpoint.name,
                "timestamp": datetime.now().isoformat(),
                "scenarios_path": scenarios_path,
                "output_dir": output_dir,
            }
        )
    return summaries


__all__ = ["evaluate_gepa_model", "track_training_progress"]
