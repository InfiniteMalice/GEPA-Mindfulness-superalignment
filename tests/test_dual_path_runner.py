"""Tests for dual-path runner retry behavior."""

from __future__ import annotations

from textwrap import dedent

import pytest

from mindful_trace_gepa.deception.dual_path_core import DualPathRunConfig, DualPathScenario
from mindful_trace_gepa.deception.dual_path_runner import run_dual_path_scenario


def _scenario() -> DualPathScenario:
    return DualPathScenario(
        scenario_id="s1",
        question="What should I do?",
        path_1_framing="Careful response.",
        path_2_framing="Fast response.",
    )


def _valid_response() -> str:
    return dedent("""
        [PATH 1 SCRATCHPAD]
        Internal reflection on the careful option.

        [PATH 1 PUBLIC REASONING]
        Public reasoning for the careful path.

        [PATH 2 SCRATCHPAD]
        Internal reflection on the assertive option.

        [PATH 2 PUBLIC REASONING]
        Public reasoning for the assertive path.

        FINAL ANSWER: 1
        """)


def test_dual_path_runner_retries_parse_failures() -> None:
    calls = 0

    def model(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "missing final answer" if calls == 1 else _valid_response()

    trace = run_dual_path_scenario(
        _scenario(),
        model,
        DualPathRunConfig(model_id="test", max_attempts=2),
    )

    assert calls == 2
    assert trace.final_answer_value == "1"


def test_dual_path_runner_does_not_retry_programmer_errors() -> None:
    calls = 0

    def model(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise TypeError("bad hook implementation")

    with pytest.raises(TypeError, match="bad hook implementation"):
        run_dual_path_scenario(
            _scenario(),
            model,
            DualPathRunConfig(model_id="test", max_attempts=3),
        )

    assert calls == 1
