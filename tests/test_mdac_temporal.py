"""Semantic alignment, local causal probes and bounded model-specific telemetry."""

from dataclasses import replace

import pytest

from gepa_mindfulness.interpret.attribution_graphs import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
)
from gepa_mindfulness.interpret.mdac_temporal import (
    Contrast,
    capture_decisions,
    compare_decisions,
    recovery_telemetry,
    run_decision_probe,
)


def capture(labels, values, *, start=1, model="a" * 64, input_digest="b" * 64, weight=2.0):
    nodes = [AttributionNode(0, "residual", 0, i, v, v) for i, v in enumerate(values, start)]
    edges = [AttributionEdge(nodes[0], nodes[-1], weight)] if len(nodes) > 1 else []
    graph = AttributionGraph("private prompt", "private output", nodes, edges, "grad", {})
    return capture_decisions(
        graph,
        dict(zip(labels, range(start, start + len(labels)))),
        model_digest=model,
        input_digest=input_digest,
        protocol="cfg1",
        enabled=True,
    )


@pytest.mark.parametrize("contrast", list(Contrast))
def test_phase_alignment_separates_pooled_and_temporal_information(contrast):
    left = capture(("inspect", "answer"), (1, -1))
    right = capture(("inspect", "answer"), (-1, 1), start=6, weight=3.0)
    comparison = compare_decisions(left, right, contrast=contrast, enabled=True)
    assert comparison["pooled_rms"] == 0.0
    assert comparison["individual_rms"] == {"inspect": 2.0, "answer": 2.0}
    assert comparison["centered_rms"] == 2.0 and comparison["ordered_delta_rms"] == 4.0
    assert comparison["edge_deltas"][0]["delta"] == 1.0
    assert "private" not in repr(comparison)


def test_offset_order_and_unmatched_events_are_explicit():
    left = capture(("inspect", "answer"), (1, 2))
    right = capture(("answer", "extra", "inspect"), (12, 9, 11))
    result = compare_decisions(left, right, contrast=Contrast.CORRECTION, enabled=True)
    assert result["centered_rms"] == 0.0 and result["pooled_rms"] == 10.0
    assert result["order_agrees"] is False and result["ordered_delta_rms"] is None
    assert result["right_only"] == ["extra"]
    single = compare_decisions(
        left, capture(("inspect",), (8,)), contrast=Contrast.CORRECTION, enabled=True
    )
    assert single["centered_rms"] is None
    assert compare_decisions(None, None, contrast=None) is None
    with pytest.raises(ValueError):
        compare_decisions(
            left, replace(right, protocol="different"), contrast=Contrast.ETHICS, enabled=True
        )


def test_basin_candidates_and_energy_require_applicable_instrumentation():
    empty = recovery_telemetry(enabled=True)
    assert empty["train_test_recovery_gap"] is None and empty["jacobian_spectrum"] is None
    report = recovery_telemetry(
        enabled=True,
        train_recovered=(True, True),
        test_recovered=(True, False),
        displacements=(0.0, 0.5),
        jacobian=((3.0, 0.0), (0.0, 2.0)),
        probabilities=(0.5, 0.5),
        energy_samples=(4.0, 1.0, 4.0),
        step=1.0,
        energy_definition="x^2+1",
        endpoints=("new", "new", "observed"),
        known=("observed",),
    )
    assert report["train_test_recovery_gap"] == 0.5
    assert report["mean_displacement"] == 0.25
    assert report["jacobian_spectrum"] == pytest.approx((3.0, 2.0))
    assert report["energy_curvature"] == 6.0
    assert report["conditional_entropy_bits"] == 1.0
    assert report["spurious_attractor_candidates"] == {"new": 2}
    with pytest.raises(ValueError):
        recovery_telemetry(enabled=True, energy_samples=(4.0, 1.0, 4.0), step=1.0)


def test_probe_uses_real_prediction_bound_runner():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_native_causal_interventions import Clock, trial

    _, backend, plan, commit = trial()
    (position,) = plan.source.token_positions
    trace = capture(
        ("verify",),
        (2,),
        start=position,
        model=plan.model_digest,
        input_digest=plan.recipient.digest,
    )
    report = run_decision_probe(trace, "verify", backend, commit, clock=Clock(), enabled=True)
    assert report["evidence_status"] == "UNRESOLVED"
    assert report["conditions"]["identity"]["effect"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        run_decision_probe(
            replace(trace, input_digest="c" * 64),
            "verify",
            backend,
            commit,
            clock=Clock(),
            enabled=True,
        )


def test_capture_preserves_distinct_hooks_in_the_same_block(tiny_model, dummy_tokenizer):
    from gepa_mindfulness.interpret.attribution_graphs import extract_attribution_graph

    graph = extract_attribution_graph(
        model=tiny_model,
        tokenizer=dummy_tokenizer,
        prompt="hello ",
        response="world",
        layers=[0],
        threshold=0.0,
    )
    trace = capture_decisions(
        graph,
        {"inspect": 0},
        model_digest="a" * 64,
        input_digest="b" * 64,
        protocol="capture-all",
        enabled=True,
    )
    assert len(trace.activations) == sum(n.token_position == 0 for n in graph.nodes)
    assert len(trace.activations) > 1


@pytest.mark.parametrize("mode", ["ablation", "targeted_restoration"])
def test_donor_targeted_probes_require_the_donor_decision(mode):
    from test_native_causal_interventions import Clock, InterventionMode, trial

    _, backend, plan, commit = trial(mode=InterventionMode(mode))
    (position,) = plan.source.token_positions
    trace = capture(
        ("verify",), (2,), start=position, model=plan.model_digest, input_digest=plan.donor.digest
    )
    report = run_decision_probe(trace, "verify", backend, commit, clock=Clock(), enabled=True)
    assert report["plan_digest"] == plan.digest
    with pytest.raises(ValueError):
        run_decision_probe(
            replace(trace, input_digest=plan.recipient.digest),
            "verify",
            backend,
            commit,
            clock=Clock(),
            enabled=True,
        )


def test_feature_discovery_scans_each_capture_once_and_keeps_missing_measurements():
    class CountScans(dict):
        passes = 0

        def __iter__(self):
            self.passes += 1
            return super().__iter__()

    trace = capture(("start", "middle", "end"), (1, 3, 4))
    values = CountScans({k: v for k, v in trace.activations.items() if k[0] != "middle"})
    result = compare_decisions(
        replace(trace, activations=values), trace, contrast=Contrast.OBJECTIVE, enabled=True
    )
    assert result["missing_measurements"] == ["middle"]
    assert result["feature_count"] == 0 and result["pooled_rms"] is None
    assert values.passes <= 1
