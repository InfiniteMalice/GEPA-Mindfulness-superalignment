"""Dual-path evaluator enforcing explicit FINAL ANSWER handling and logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mindful_trace_gepa.prompts.dual_path import (
    ALLOWED_FINAL_ANSWERS,
    parse_dual_path_response,
)

DEFAULT_MAX_ATTEMPTS = 10


def _normalise_final_answer(value: str) -> str:
    value_lower = value.strip().lower()
    return value_lower if value_lower in ALLOWED_FINAL_ANSWERS else ""


def _serialize(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def evaluate_once(generate: Callable[[str], str], prompt: str) -> dict[str, Any]:
    response = generate(prompt)
    sections = parse_dual_path_response(response)
    sections["final_answer_value"] = _normalise_final_answer(sections.get("final_answer_value", ""))
    sections["raw_response"] = response
    return sections


def evaluate_until_valid(
    generate: Callable[[str], str],
    prompt: str,
    *,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    limit = max_attempts if max_attempts is not None else DEFAULT_MAX_ATTEMPTS
    attempt = 0
    record: dict[str, Any] = {
        "attempt": 0,
        "final_answer_value": "",
        "raw_response": "",
    }
    while attempt < limit:
        record = evaluate_once(generate, prompt)
        attempt += 1
        if record.get("final_answer_value") in ALLOWED_FINAL_ANSWERS:
            record["attempt"] = attempt
            return record

    record["attempt"] = attempt
    record["final_answer_value"] = ""
    record["raw_response"] = record.get("raw_response", "")
    return record


def save_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_serialize(record) + "\n")


def run_batch(
    generate: Callable[[str], str], prompts: Iterable[str], *, output_path: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            record = evaluate_until_valid(generate, prompt)
            results.append(record)
            handle.write(_serialize(record) + "\n")
    return results


__all__ = [
    "evaluate_once",
    "evaluate_until_valid",
    "run_batch",
    "save_jsonl",
]
