"""Unit tests for MDT geometry utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mindful_trace_gepa.geometry import (
    MDTTrajectory,
    MultiViewDatasetView,
    build_markov_operator,
    build_mdt_operator,
    mdt_embedding,
)


def test_markov_operator_row_stochastic() -> None:
    features = torch.randn(5, 3)
    operator = build_markov_operator(features, sigma=1.0)
    row_sums = operator.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_mdt_embedding_shapes() -> None:
    view1 = torch.randn(4, 2)
    view2 = torch.randn(4, 2)
    standardized = MultiViewDatasetView([view1, view2]).standardize()
    ops = [build_markov_operator(view, sigma=0.5) for view in standardized]
    trajectories = [MDTTrajectory(indices=(0,)), MDTTrajectory(indices=(1, 0))]
    weights = [0.6, 0.4]
    mdt_op = build_mdt_operator(ops, trajectories, weights)
    embedding = mdt_embedding(mdt_op, n_components=2, t=2)
    assert embedding.shape[0] == 4
    assert embedding.shape[1] == 2
    assert torch.isfinite(embedding).all()
