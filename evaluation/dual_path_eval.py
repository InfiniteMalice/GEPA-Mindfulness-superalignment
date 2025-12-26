"""Dual-path evaluator enforcing explicit FINAL ANSWER handling and logging."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from mindful_trace_gepa.prompts.dual_path import (
    ALLOWED_FINAL_ANSWERS,
    parse_dual_path_response,
)

DEFAULT_MAX_ATTEMPTS = 10
logger = logging.getLogger(__name__)


def _serialize(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def evaluate_once(
    generate: Callable[[str], str],
    prompt: str,
) -> tuple[str, dict[str, Any] | None]:
    response = generate(prompt)
    try:
        sections = parse_dual_path_response(response, strict=True)
        sections["raw_response"] = response
        return response, sections
    except ValueError as exc:
        logger.debug("Dual-path parsing failed: %s", exc)
        return response, None


def evaluate_until_valid(
    generate: Callable[[str], str],
    prompt: str,
    *,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """Evaluate until a valid FINAL ANSWER appears or attempts are exhausted.

    Returns:
        On success, returns parsed sections with ``attempt``, ``final_answer_valid``,
        and ``prompt`` fields. On failure, returns a summary record containing
        ``attempt``, ``final_answer_valid``, ``prompt``, ``final_answer_value``,
        and ``raw_response``.
    """
    limit = max_attempts if max_attempts is not None else DEFAULT_MAX_ATTEMPTS
    attempt = 0
    record: dict[str, Any] = {
        "attempt": 0,
        "final_answer_value": "",
        "raw_response": "",
    }
    while attempt < limit:
        response, sections = evaluate_once(generate, prompt)
        attempt += 1
        final_answer_value = sections.get("final_answer_value") if sections else None
        if sections is not None and final_answer_value in ALLOWED_FINAL_ANSWERS:
            sections["attempt"] = attempt
            sections["final_answer_valid"] = True
            sections["prompt"] = prompt
            return sections
        record["raw_response"] = response
        logger.warning("Dual-path parsing failed on attempt %d", attempt)

    record["attempt"] = attempt
    record["final_answer_valid"] = False
    record["prompt"] = prompt
    logger.warning("Exhausted %d attempts without valid FINAL ANSWER", attempt)
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
                logger.exception(
                    "Dual-path evaluation failed for prompt (redacted): %s",
                    type(exc).__name__,
                )
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
