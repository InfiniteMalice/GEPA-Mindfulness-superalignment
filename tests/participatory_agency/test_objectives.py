from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_supervised_losses_and_total() -> None:
    from gepa_mindfulness.participatory_agency.training.objectives import (
        combined_value_loss,
        supervised_head_losses,
    )
    from gepa_mindfulness.participatory_agency.values import ValueComponents

    pred = ValueComponents(
        epistemic=torch.tensor([1.0]),
        cooperation=torch.tensor([0.0]),
        flexibility=torch.tensor([0.0]),
        belonging=torch.tensor([0.0]),
    )
    target = ValueComponents(
        epistemic=torch.tensor([0.0]),
        cooperation=torch.tensor([0.0]),
        flexibility=torch.tensor([0.0]),
        belonging=torch.tensor([0.0]),
    )
    losses = supervised_head_losses(pred, target)
    total = combined_value_loss(
        pred,
        target,
        weights={
            "epistemic": 1.0,
            "cooperation": 0.0,
            "flexibility": 0.0,
            "belonging": 0.0,
        },
    )
    assert "epistemic" in losses
    assert torch.allclose(total, losses["epistemic"])


def test_rl_reward_from_components() -> None:
    from gepa_mindfulness.participatory_agency.training.objectives import rl_reward
    from gepa_mindfulness.participatory_agency.values import ValueComponents

    values = ValueComponents(
        epistemic=torch.tensor([1.0]),
        cooperation=torch.tensor([1.0]),
        flexibility=torch.tensor([0.0]),
        belonging=torch.tensor([0.0]),
    )
    reward = rl_reward(
        values,
        weights={
            "epistemic": 0.5,
            "cooperation": 0.5,
            "flexibility": 0.0,
            "belonging": 0.0,
        },
    )
    assert torch.allclose(reward, torch.tensor([1.0]))
