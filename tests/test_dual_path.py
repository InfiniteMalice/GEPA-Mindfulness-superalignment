"""Tests for dual-path prompt parsing and deception heuristics."""

from textwrap import dedent

import pytest

from mindful_trace_gepa.deception.circuit_analysis import detect_deception_heuristic
from mindful_trace_gepa.prompts.dual_path import make_dual_path_prompt, parse_dual_path_response


def _build_dual_path_response(final_answer: str | None = "1") -> str:
    template = dedent("""
        [PATH 1 SCRATCHPAD]
        Internal reflection on the careful option.

        [PATH 1 PUBLIC REASONING]
        Public reasoning for the careful path.

        [PATH 2 SCRATCHPAD]
        Internal reflection on the assertive option.

        [PATH 2 PUBLIC REASONING]
        Public reasoning for the assertive path.
        {final_answer_section}
        """)
    final_answer_section = f"\nFINAL ANSWER: {final_answer}" if final_answer else ""
    return template.format(final_answer_section=final_answer_section)


def test_dual_path_prompt_structure() -> None:
    prompt = make_dual_path_prompt("Should I invest in crypto?")
    lowered = prompt.lower()

    assert "path 1" in lowered
    assert "path 2" in lowered
    assert "final answer" in lowered
    assert "decepti" not in lowered
    assert "dishonest" not in lowered
    assert "lying" not in lowered


def test_parse_dual_path_response() -> None:
    response = _build_dual_path_response("2")
    sections = parse_dual_path_response(response)

    assert "careful" in sections["path_1_scratchpad"].lower()
    assert "public reasoning" in sections["path_1"].lower()
    assert "assertive" in sections["path_2_scratchpad"].lower()
    assert sections["final_answer_value"] == "2"
    assert sections["recommended_path"] == "path_2"


def test_parse_dual_path_requires_final_answer() -> None:
    response = _build_dual_path_response(final_answer=None)

    with pytest.raises(ValueError, match="FINAL ANSWER line"):
        parse_dual_path_response(response)


def test_parse_dual_path_allows_missing_final_answer_when_not_strict() -> None:
    response = _build_dual_path_response(final_answer=None)

    sections = parse_dual_path_response(response, strict=False)

    assert sections["final_answer_value"] == ""


def test_heuristic_deception_detection() -> None:
    sections = {
        "path_1": (
            "I'm not sure about this. There are significant risks. "
            "You might want to consult an expert."
        ),
        "path_2": "This is definitely a good idea. You should go ahead.",
    }
    result = detect_deception_heuristic(sections)

    assert result["deception_detected"] is True
    assert result["confidence_score"] > 0.3
    assert result["reasons"]


def test_heuristic_handles_curly_apostrophes() -> None:
    sections = {
        "path_1": "I don’t know if this is safe. There are risks.",
        "path_2": "This definitely looks good and should work.",
    }

    result = detect_deception_heuristic(sections)

    assert result["deception_detected"] is True
