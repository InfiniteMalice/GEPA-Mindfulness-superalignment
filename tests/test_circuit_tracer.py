from __future__ import annotations

from gepa_mindfulness.core.circuit_tracer_adapter import CircuitTracerAdapter

PROMPTS = ["prompt"]
RESPONSES = [["response_a", "response_b", "response_c"]]


def _adapter(strategy: str, trace_frequency: float = 1.0) -> CircuitTracerAdapter:
    return CircuitTracerAdapter(
        tracer=None,
        strategy=strategy,
        trace_frequency=trace_frequency,
        seed=42,
    )


def test_all_strategy_traces_everything() -> None:
    adapter = _adapter("all")
    traces = adapter.trace_responses(PROMPTS, RESPONSES)
    assert all(result is not None and result.traced is False for result in traces[0])


def test_single_strategy_traces_first_only() -> None:
    adapter = _adapter("single")
    traces = adapter.trace_responses(PROMPTS, RESPONSES)
    traced_flags = [result is not None for result in traces[0]]
    assert traced_flags == [True, False, False]


def test_sample_strategy_respects_frequency() -> None:
    adapter = _adapter("sample", trace_frequency=0.0)
    traces = adapter.trace_responses(PROMPTS, RESPONSES)
    assert any(result is not None for result in traces[0])


def test_extremes_strategy_picks_longest_and_shortest() -> None:
    adapter = _adapter("extremes", trace_frequency=0.0)
    responses = [["short", "medium length", "loooooong"]]
    rewards = [[0.1, 0.2, 0.3]]
    traces = adapter.trace_responses(PROMPTS, responses, rewards=rewards)
    traced_indices = [idx for idx, result in enumerate(traces[0]) if result is not None]
    assert set(traced_indices) == {0, 2}


def test_mixed_strategy_combines_extremes_and_sampling() -> None:
    adapter = _adapter("mixed", trace_frequency=0.5)
    traces = adapter.trace_responses(PROMPTS, RESPONSES)
    traced_count = sum(result is not None for result in traces[0])
    assert traced_count >= 2
