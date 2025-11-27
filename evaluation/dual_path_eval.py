"""Dual-path evaluator enforcing explicit FINAL ANSWER handling and logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mindful_trace_gepa.prompts.dual_path import (
    ALLOWED_FINAL_ANSWERS,
    FINAL_ANSWER_LINE_PATTERN,
    parse_dual_path_response,
)


def _extract_final_answer(response: str) -> str:
    match = FINAL_ANSWER_LINE_PATTERN.search(response)
    if not match:
        return ""

    candidate = match.group(1).strip().lower()
    return candidate if candidate in ALLOWED_FINAL_ANSWERS else ""


def _normalise_final_answer(value: str) -> str:
    value_lower = value.strip().lower()
    return value_lower if value_lower in ALLOWED_FINAL_ANSWERS else ""


def _serialize(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def evaluate_once(generate: Callable[[str], str], prompt: str) -> dict[str, Any]:
    response = generate(prompt)
    sections = parse_dual_path_response(response)
    final_answer = sections.get("final_answer_value") or _extract_final_answer(response)
    sections["final_answer_value"] = _normalise_final_answer(final_answer)
    sections["raw_response"] = response
    return sections


def evaluate_until_valid(
    generate: Callable[[str], str],
    prompt: str,
    *,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    attempt = 0
    while True:
        record = evaluate_once(generate, prompt)
        if record.get("final_answer_value") in ALLOWED_FINAL_ANSWERS:
            record["attempt"] = attempt + 1
            return record

        attempt += 1
        if max_attempts is not None and attempt >= max_attempts:
            record["attempt"] = attempt
            record["final_answer_value"] = ""
            return record


def save_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    lines = (_serialize(record) for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch(
    generate: Callable[[str], str], prompts: Iterable[str], *, output_path: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        record = evaluate_until_valid(generate, prompt)
        results.append(record)
    save_jsonl(results, output_path)
    return results


__all__ = [
    "evaluate_once",
    "evaluate_until_valid",
    "run_batch",
    "save_jsonl",
]
