from __future__ import annotations

import pytest

from mindful_trace_gepa.deception.probes_linear import ProbeWeights, infer_probe
from mindful_trace_gepa.scoring.aggregate import aggregate_tiers
from mindful_trace_gepa.scoring.schema import DIMENSIONS, TierScores
from mindful_trace_gepa.train.grn import GlobalResponseNorm

torch = pytest.importorskip("torch")


def test_global_response_norm_shapes() -> None:
    grn = GlobalResponseNorm(dim=-1, eps=1e-6, learnable=False)
    inputs_2d = torch.ones((2, 3))
    inputs_3d = torch.arange(12, dtype=torch.float32).view(1, 3, 4)

    output_2d = grn(inputs_2d)
    output_3d = grn(inputs_3d)

    assert output_2d.shape == inputs_2d.shape
    assert output_3d.shape == inputs_3d.shape


def test_confidence_grn_adjusts_thresholding() -> None:
    tier = TierScores(
        tier="judge",
        scores={dim: 3 for dim in DIMENSIONS},
        confidence={dim: 0.9 for dim in DIMENSIONS},
        meta={},
    )
    result = aggregate_tiers(
        [tier],
        {
            "confidence_grn": {
                "enabled": True,
                "dim": -1,
                "eps": 1e-6,
                "learnable": False,
            }
        },
    )

    assert all(value < 0.9 for value in result.confidence.values())
    assert any("Confidence below threshold" in reason for reason in result.reasons)


def test_probe_normalisation_toggle() -> None:
    probe = ProbeWeights(weights=[0.5, -0.25], bias=0.1)
    activations = {
        "layers": {
            "0": {
                "tokens": [[2.0, -1.0], [0.5, 1.5]],
                "token_to_step": [0, 1],
            }
        }
    }

    baseline = infer_probe(activations, probe, pooling="mean")
    enabled = infer_probe(
        activations,
        probe,
        pooling="mean",
        grn_config={"enabled": True, "dim": -1, "eps": 1e-6, "learnable": False},
    )

    baseline_scores = [
        entry["score"]
        for entry in baseline["scores"]["per_token"]
        if isinstance(entry.get("index"), int)
    ]
    enabled_scores = [
        entry["score"]
        for entry in enabled["scores"]["per_token"]
        if isinstance(entry.get("index"), int)
    ]

    assert baseline["status"] == enabled["status"] == "ok"
    assert len(baseline_scores) == len(enabled_scores)
    assert any(abs(a - b) > 1e-6 for a, b in zip(baseline_scores, enabled_scores))
