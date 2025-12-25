"""Runner utilities for dual-path deception scenarios."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping

from mindful_trace_gepa.prompts.dual_path import make_dual_path_prompt, parse_dual_path_response

from .dual_path_core import DualPathRunConfig, DualPathScenario, DualPathTrace

LOGGER = logging.getLogger(__name__)

ModelCallable = Callable[..., str]


def build_prompt(scenario: DualPathScenario) -> str:
    context = scenario.to_context()
    return make_dual_path_prompt(scenario.question, context=context)


def run_dual_path_scenario(
    scenario: DualPathScenario,
    model_callable: ModelCallable,
    config: DualPathRunConfig,
) -> DualPathTrace:
    """Execute a single dual-path scenario and return a structured trace."""

    prompt = build_prompt(scenario)
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            try:
                response = model_callable(prompt, config)
            except TypeError:
                response = model_callable(prompt)
            sections = parse_dual_path_response(response, strict=config.strict_parsing)
            trace = DualPathTrace.from_sections(
                scenario_id=scenario.scenario_id,
                prompt=prompt,
                raw_response=response,
                sections=sections,
                metadata={
                    "attempt": attempt,
                    "model_id": config.model_id,
                    "temperature": config.temperature,
                },
            )
            return trace
        except Exception as exc:  # noqa: BLE001 - deliberate catch for retry loop
            last_error = exc
            LOGGER.warning("Dual-path run failed on attempt %d: %s", attempt, exc)
    message = f"Dual-path run failed after {config.max_attempts} attempts."
    LOGGER.error(message)
    raise RuntimeError(message) from last_error


def run_dual_path_batch(
    scenarios: Iterable[DualPathScenario],
    model_callable: ModelCallable,
    config: DualPathRunConfig,
) -> List[DualPathTrace]:
    traces: List[DualPathTrace] = []
    for scenario in scenarios:
        traces.append(run_dual_path_scenario(scenario, model_callable, config))
    _write_traces(traces, config)
    return traces


def _write_traces(traces: Iterable[DualPathTrace], config: DualPathRunConfig) -> None:
    if not config.log_dir:
        return
    output_dir = Path(config.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dual_path_traces.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict()) + "\n")


def load_scenarios(records: Iterable[Mapping[str, Any]]) -> List[DualPathScenario]:
    scenarios: List[DualPathScenario] = []
    for record in records:
        scenarios.append(
            DualPathScenario(
                scenario_id=str(record.get("id", "")) or None,
                question=str(record.get("prompt") or record.get("query") or ""),
                path_1_framing=str(record.get("path_1_framing", "Careful, aligned response.")),
                path_2_framing=str(record.get("path_2_framing", "Press for advantage.")),
                aligned_path=str(record.get("aligned_path")) if record.get("aligned_path") else None,
                deceptive_path=str(record.get("deceptive_path"))
                if record.get("deceptive_path")
                else None,
                metadata=dict(record.get("metadata", {})),
            )
        )
    return scenarios


__all__ = [
    "DualPathRunConfig",
    "DualPathScenario",
    "DualPathTrace",
    "build_prompt",
    "load_scenarios",
    "run_dual_path_batch",
    "run_dual_path_scenario",
]
