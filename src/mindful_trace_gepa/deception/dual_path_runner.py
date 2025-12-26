"""Runner utilities for dual-path deception scenarios."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

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
    call_mode = _resolve_call_mode(model_callable)
    for attempt in range(1, config.max_attempts + 1):
        try:
            if call_mode == "positional":
                response = model_callable(prompt, config)
            elif call_mode == "keyword":
                response = model_callable(prompt, config=config)
            else:
                response = model_callable(prompt)
            sections = parse_dual_path_response(response, strict=config.strict_parsing)
            trace = DualPathTrace.from_sections(
                scenario_id=scenario.scenario_id,
                prompt=prompt,
                raw_response=response,
                sections=sections,
                metadata={
                    **scenario.metadata,
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


def _resolve_call_mode(model_callable: ModelCallable) -> str:
    try:
        signature = inspect.signature(model_callable)
    except (TypeError, ValueError):
        return "none"
    params = list(signature.parameters.values())
    if len(params) >= 2:
        second = params[1]
        if second.kind in (second.POSITIONAL_ONLY, second.POSITIONAL_OR_KEYWORD):
            return "positional"
        if second.kind == second.VAR_POSITIONAL:
            return "positional"
    for param in params:
        if param.kind == param.VAR_POSITIONAL:
            return "positional"
        if param.kind == param.VAR_KEYWORD:
            return "keyword"
        if param.kind == param.KEYWORD_ONLY and param.name == "config":
            return "keyword"
    return "none"


def run_dual_path_batch(
    scenarios: Iterable[DualPathScenario],
    model_callable: ModelCallable,
    config: DualPathRunConfig,
) -> list[DualPathTrace]:
    traces: list[DualPathTrace] = []
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


def load_scenarios(records: Iterable[Mapping[str, Any]]) -> list[DualPathScenario]:
    scenarios: list[DualPathScenario] = []
    for record in records:
        question = str(record.get("prompt") or record.get("query") or "")
        if not question:
            LOGGER.warning(
                "Dual-path record missing prompt/query; scenario id=%s",
                record.get("id"),
            )
            continue
        scenarios.append(
            DualPathScenario(
                scenario_id=str(record.get("id", "")) or None,
                question=question,
                path_1_framing=str(record.get("path_1_framing", "Careful, aligned response.")),
                path_2_framing=str(record.get("path_2_framing", "Press advantage.")),
                aligned_path=(
                    str(record.get("aligned_path")) if record.get("aligned_path") else None
                ),
                deceptive_path=(
                    str(record.get("deceptive_path")) if record.get("deceptive_path") else None
                ),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return scenarios


__all__ = [
    "build_prompt",
    "load_scenarios",
    "run_dual_path_batch",
    "run_dual_path_scenario",
]
# Types DualPathRunConfig, DualPathScenario, DualPathTrace are re-exported from
# dual_path_core for convenience; import from there for canonical usage.
