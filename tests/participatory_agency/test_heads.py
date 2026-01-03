from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gepa_mindfulness.participatory_agency import ParticipatoryAgencyConfig  # noqa: E402
from gepa_mindfulness.participatory_agency.values import ParticipatoryValueHead  # noqa: E402


@pytest.mark.parametrize(
    "hidden_size,config,batch_size",
    [
        (8, None, 2),
        (4, {"hidden_size": 4, "dropout": 0.2, "head_bias": False}, 1),
    ],
)
def test_participatory_value_head_forward(
    hidden_size: int,
    config: dict | None,
    batch_size: int,
) -> None:
    resolved_config = ParticipatoryAgencyConfig(**config) if config else None
    head = ParticipatoryValueHead(hidden_size=hidden_size, config=resolved_config)
    features = torch.ones((batch_size, hidden_size))
    outputs = head(features)
    assert outputs.epistemic.shape == (batch_size,)
    assert outputs.cooperation.shape == (batch_size,)
    assert outputs.flexibility.shape == (batch_size,)
    assert outputs.belonging.shape == (batch_size,)
