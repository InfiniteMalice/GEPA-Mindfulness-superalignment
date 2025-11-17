from __future__ import annotations

import math

import pytest
from rg_tracer.scoring.aggregator import parse_summary_block


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


def test_fallback_dedents_uniform_indent_block() -> None:
    """Uniform indentation offsets should not break fallback parsing."""
    text = "    root:\n" "        child:\n" "            value: 2\n"

    parsed = parse_summary_block(text)

    assert parsed == {"root": {"child": {"value": 2}}}


def test_fallback_preserves_leading_zero_identifiers() -> None:
    """Strings with leading zeros remain strings after fallback parsing."""
    text = """
id: 001
code: 0123
normal: 123
""".strip()

    parsed = parse_summary_block(text)

    assert parsed["id"] == "001"
    assert parsed["code"] == "0123"
    assert parsed["normal"] == 123


def test_fallback_preserves_negative_leading_zero_identifiers() -> None:
    """Signed identifiers like "-01" should remain strings as well."""
    text = "id: -01"

    parsed = parse_summary_block(text)

    assert parsed["id"] == "-01"


def test_parse_summary_block_json_object() -> None:
    """The parser should favor JSON decoding when the input is valid."""
    text = '{"key": "value", "nested": {"count": 42}}'

    parsed = parse_summary_block(text)

    assert parsed == {"key": "value", "nested": {"count": 42}}


def test_parse_summary_block_empty_input_returns_empty_dict() -> None:
    """Whitespace-only blocks should produce empty dictionaries."""

    assert parse_summary_block("") == {}
    assert parse_summary_block("  \n   ") == {}


def test_parse_summary_block_json_non_object_raises() -> None:
    """Non-object JSON inputs should raise to signal malformed traces."""

    with pytest.raises(ValueError, match="must be an object"):
        parse_summary_block('["not", "an", "object"]')


def test_fallback_coerces_special_float_tokens() -> None:
    """Special float words should map to IEEE values when parsing."""

    text = """
score: inf
penalty: -infinity
noise: NaN
""".strip()

    parsed = parse_summary_block(text)

    assert parsed["score"] == float("inf")
    assert parsed["penalty"] == float("-inf")
    assert math.isnan(parsed["noise"])
