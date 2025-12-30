from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gepa_mindfulness.participatory_agency import ParticipatoryAgencyConfig
from gepa_mindfulness.participatory_agency.values import ParticipatoryValueHead


def test_participatory_value_head_forward() -> None:
    head = ParticipatoryValueHead(hidden_size=8)
    features = torch.zeros((2, 8))
    outputs = head(features)
    assert outputs.epistemic.shape == (2,)
    assert outputs.cooperation.shape == (2,)
    assert outputs.flexibility.shape == (2,)
    assert outputs.belonging.shape == (2,)


def test_value_head_uses_config() -> None:
    config = ParticipatoryAgencyConfig(hidden_size=4, dropout=0.2, head_bias=False)
    head = ParticipatoryValueHead(hidden_size=4, config=config)
    features = torch.ones((1, 4))
    outputs = head(features)
    assert outputs.epistemic.shape == (1,)
