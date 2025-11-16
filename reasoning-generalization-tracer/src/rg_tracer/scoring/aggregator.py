"""Parse tracer aggregation reports with a JSON-first then indent fallback strategy."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def parse_summary_block(text: str) -> dict[str, Any]:
    """Parse a summary text block into a nested structure.

    The tracer usually emits JSON objects. Non-object JSON triggers a ValueError so
    callers can surface formatting issues. When decoding fails entirely the helper
    falls back to the indentation parser so downstream scoring can treat both
    formats the same.
    """
    stripped = text.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return _parse_indented_pairs(stripped.splitlines())

    if not isinstance(parsed, dict):
        raise ValueError("Summary JSON must be an object")

    return parsed


def _parse_indented_pairs(lines: Iterable[str]) -> dict[str, Any]:
    """Parse indented key/value lines into nested dictionaries."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in lines:
        if not raw_line.strip():
            continue

        indent = _indent_width(raw_line)
        content = raw_line.strip()

        if content.endswith(":"):
            _pop_stack_to_parent(stack, indent)
            key = content[:-1].strip()
            parent = stack[-1][1]
            new_container: dict[str, Any] = {}
            parent[key] = new_container
            stack.append((indent + 1, new_container))
            continue

        if ":" not in content:
            continue

        _pop_stack_to_parent(stack, indent)
        key, value_text = _split_key_value(content)
        parent = stack[-1][1]
        parent[key] = _coerce_value(value_text)

    return root


def _pop_stack_to_parent(stack: list[tuple[int, dict[str, Any]]], indent: int) -> None:
    """Pop frames until indent is below the stored child threshold (parent indent + 1)."""
    while len(stack) > 1 and indent < stack[-1][0]:
        stack.pop()


def _indent_width(raw_line: str) -> int:
    """Return the indentation width treating tabs as four spaces."""
    expanded = raw_line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def _split_key_value(content: str) -> tuple[str, str]:
    """Split "key: value" content without breaking embedded colons."""
    key, value = content.split(":", 1)
    return key.strip(), value.strip()


def _coerce_value(text: str) -> Any:
    """Convert string tokens into native Python scalars when possible."""
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"

    if lower in {"null", "none"}:
        return None

    if len(text) > 1 and text[0] == "0" and text[1].isdigit():
        # Preserve leading-zero identifiers as strings instead of coercing.
        return text

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text
