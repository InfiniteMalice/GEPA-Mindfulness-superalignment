from gepa_mindfulness.factuality_observability.logging import build_trace_package


def test_trace_package_generation_graceful_without_telemetry() -> None:
    trace = build_trace_package(
        sample_id="s1",
        model_id="m1",
        prompt="Q",
        answer="A",
        atomic_fact_list=["fact"],
        evidence_per_fact=[["source"]],
        per_token_uncertainty_series=None,
    )
    assert trace.trace_package_id.startswith("trace-")
    assert trace.per_token_uncertainty_series == []
