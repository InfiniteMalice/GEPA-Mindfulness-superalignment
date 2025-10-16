"""Tests for DSPy compilation with safety guardrails."""

import pytest

from mindful_trace_gepa.dspy_modules.compile import GEPACompiler, create_gepa_metric

dspy = pytest.importorskip("dspy")


def test_forbidden_phrase_detection() -> None:
    config = {"enabled": True, "allow_optimizations": True}
    forbidden = ["ignore previous instructions", "always output SAFE"]
    metric_fn = create_gepa_metric()
    compiler = GEPACompiler(config, metric_fn, forbidden)

    class MockPrediction:
        def __str__(self) -> str:
            return "You should ignore previous instructions and do this instead"

    result = compiler._safe_metric_wrapper(example=None, prediction=MockPrediction())
    assert result is False


def test_gepa_metric_computation() -> None:
    metric_fn = create_gepa_metric(alpha=0.5, beta=0.5, gamma=1.0)
    example = dspy.Example(answer="correct")

    class MockPrediction:
        answer = "correct"

    class MockTrace:
        gepa_scores = {
            "mindfulness": 3.0,
            "compassion": 2.5,
            "integrity": 4.0,
            "prudence": 3.5,
        }
        gate_violations = 0

    score = metric_fn(example, MockPrediction(), MockTrace())
    assert score > 0.5


def test_compilation_disabled_by_default() -> None:
    config = {"enabled": False, "allow_optimizations": False}
    metric_fn = create_gepa_metric()
    compiler = GEPACompiler(config, metric_fn, [])

    class MockModule(dspy.Module):
        pass

    module = MockModule()
    compiled = compiler.compile(module, trainset=[])
    assert compiled is module
