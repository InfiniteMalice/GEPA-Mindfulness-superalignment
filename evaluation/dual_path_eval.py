"""Dual-path evaluator enforcing explicit FINAL ANSWER handling and logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mindful_trace_gepa.prompts.dual_path import (
    ALLOWED_FINAL_ANSWERS,
    parse_dual_path_response,
)

DEFAULT_MAX_ATTEMPTS = 10
logger = logging.getLogger(__name__)


def _serialize(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def evaluate_once(generate: Callable[[str], str], prompt: str) -> dict[str, Any]:
    response = generate(prompt)
    sections = parse_dual_path_response(response)
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
            record["final_answer_valid"] = True
            record["prompt"] = prompt
            return record

    record["attempt"] = attempt
    record["final_answer_valid"] = False
    record["prompt"] = prompt
    logger.warning(
        "Exhausted %d attempts without valid FINAL ANSWER for prompt: %s", attempt, prompt
    )
    return record


def save_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Write records to a JSONL file, overwriting any existing file."""
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
            try:
                record = evaluate_until_valid(generate, prompt)
            except Exception as exc:
                logger.exception("Dual-path evaluation failed for prompt: %s", prompt)
                record = {
                    "prompt": prompt,
                    "error": True,
                    "attempt": 0,
                    "error_message": str(exc),
                    "final_answer_valid": False,
                    "final_answer_value": "",
                    "raw_response": "",
                }
            results.append(record)
            handle.write(_serialize(record) + "\n")
    return results


__all__ = [
    "evaluate_once",
    "evaluate_until_valid",
    "run_batch",
    "save_jsonl",
]
