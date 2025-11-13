"""Tests for the reasoning generalization tracer aggregation parser."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RG_SRC = PROJECT_ROOT / "reasoning-generalization-tracer" / "src"
if str(RG_SRC) not in sys.path:
    sys.path.insert(0, str(RG_SRC))

parse_summary_block = importlib.import_module("rg_tracer.scoring.aggregator").parse_summary_block


def test_fallback_nested_dict_from_indent() -> None:
    """The fallback parser should build nested dicts for deeper indents."""
    text = """
config:
    attr:
        thresholds:
            entropy:
                min: 0.25
    dataset: eval
""".strip()

    parsed = parse_summary_block(text)

    assert parsed == {
        "config": {
            "attr": {
                "thresholds": {
                    "entropy": {
                        "min": 0.25,
                    }
                }
            },
            "dataset": "eval",
        }
    }


def test_fallback_supports_sibling_blocks() -> None:
    """Ensure colon-terminated siblings at the same depth create new dicts."""
    text = """
root:
    first:
        value: 1
    second:
        nested:
            enabled: true
""".strip()

    parsed = parse_summary_block(text)

    assert parsed["root"]["first"]["value"] == 1
    assert parsed["root"]["second"]["nested"]["enabled"] is True


def test_fallback_handles_tab_indentation() -> None:
    """Tabs should expand to spaces so nested blocks remain structured."""
    text = "root:\n\tchild:\n\t\tvalue: 1"

    parsed = parse_summary_block(text)

    assert parsed == {"root": {"child": {"value": 1}}}
