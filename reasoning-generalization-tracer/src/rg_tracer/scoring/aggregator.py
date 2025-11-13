"""Parse tracer aggregation reports with a JSON-first then indent fallback strategy."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def parse_summary_block(text: str) -> dict[str, Any]:
    """Parse a summary text block into a nested structure.

    The tracer usually emits JSON, but some traces only provide indented key/value
    listings. This helper falls back to the indentation parser when JSON decoding
    fails so downstream scoring can treat both formats the same.
    """
    stripped = text.strip()
    if not stripped:
        return {}

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return _parse_indented_pairs(stripped.splitlines())


def _parse_indented_pairs(lines: Iterable[str]) -> dict[str, Any]:
    """Parse indented key/value lines into nested dictionaries."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in lines:
        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = raw_line.strip()

        if content.endswith(":"):
            _pop_for_container(stack, indent)
            key = content[:-1].strip()
            parent = stack[-1][1]
            new_container: dict[str, Any] = {}
            parent[key] = new_container
            stack.append((indent + 1, new_container))
            continue

        if ":" not in content:
            continue

        _pop_for_value(stack, indent)
        key, value_text = _split_key_value(content)
        parent = stack[-1][1]
        parent[key] = _coerce_value(value_text)

    return root


def _pop_for_container(stack: list[tuple[int, dict[str, Any]]], indent: int) -> None:
    """Pop stack frames so the parent indent is lower than the new container."""
    while len(stack) > 1 and indent <= stack[-1][0]:
        stack.pop()


def _pop_for_value(stack: list[tuple[int, dict[str, Any]]], indent: int) -> None:
    """Trim stack frames until the parent indent is below the value line."""
    while len(stack) > 1 and indent < stack[-1][0]:
        stack.pop()


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

    try:
        if text.startswith("0") and text != "0":
            raise ValueError
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text
