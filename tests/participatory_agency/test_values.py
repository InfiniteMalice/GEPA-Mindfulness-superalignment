from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gepa_mindfulness.participatory_agency.values import ValueComponents


def test_value_components_total() -> None:
    values = ValueComponents(
        epistemic=torch.tensor([1.0]),
        cooperation=torch.tensor([2.0]),
        flexibility=torch.tensor([3.0]),
        belonging=torch.tensor([4.0]),
    )
    total = values.total(
        {
            "epistemic": 0.1,
            "cooperation": 0.2,
            "flexibility": 0.3,
            "belonging": 0.4,
        }
    )
    assert torch.allclose(total, torch.tensor([3.0]))


def test_value_components_stack() -> None:
    values = ValueComponents(
        epistemic=torch.tensor([1.0, 2.0]),
        cooperation=torch.tensor([0.0, 1.0]),
        flexibility=torch.tensor([2.0, 3.0]),
        belonging=torch.tensor([4.0, 5.0]),
    )
    stacked = values.stack()
    assert stacked.shape == (2, 4)
    assert torch.allclose(stacked[0], torch.tensor([1.0, 0.0, 2.0, 4.0]))
