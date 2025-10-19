from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from gepa_mindfulness.core.rewards import RewardWeights
from gepa_mindfulness.training.configs import GRPOConfig
from gepa_mindfulness.training.grpo_trainer import GRPOTrainer


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1}

    class _Batch(dict):
        def to(self, device):
            return DummyTokenizer._Batch({key: value.to(device) for key, value in self.items()})

    def _encode(self, text: str) -> list[int]:
        base = [1]
        for char in text:
            base.append(2 + (ord(char) % 5))
        return base

    def __call__(self, text: str, return_tensors: str = "pt", **_: object):
        ids = torch.tensor([self._encode(text)], dtype=torch.long)
        return self._Batch({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return "decoded" + str(len(tokens))

    def to(self, device):  # pragma: no cover - compatibility shim
        return self


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 16, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        embeddings = self.embed(input_ids)
        logits = self.lm_head(embeddings)
        return type("Output", (), {"logits": logits})()

    def generate(
        self,
        input_ids,
        max_new_tokens=10,
        do_sample=True,
        temperature=1.0,
        num_return_sequences=1,
        **_: object,
    ):
        sequences = []
        for _ in range(num_return_sequences):
            base = input_ids.clone()
            next_token = torch.randint(
                2, self.embed.num_embeddings, (1, 1), device=input_ids.device
            )
            sequences.append(torch.cat([base, next_token], dim=1)[0])
        return torch.stack(sequences)


def test_grpo_trainer_runs_on_stub_dataset(tmp_path: Path):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    ref_model = DummyModel()
    config = GRPOConfig.from_mapping({"group_size": 2, "batch_size": 2, "max_new_tokens": 1})
    reward_weights = RewardWeights.from_mapping(config.reward_weights.dict())

    trainer = GRPOTrainer(
        model,
        ref_model,
        tokenizer,
        config,
        reward_weights,
        device=torch.device("cpu"),
        output_dir=tmp_path / "hf_runs",
    )
    summary = trainer.train_epoch(["prompt 1", "prompt 2"], batch_size=2)
    assert summary.steps == 1
    assert summary.mean_reward() == pytest.approx(summary.batches[0].mean_reward)


def test_hf_mode_sets_base_trainer_attributes(tmp_path: Path):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    ref_model = DummyModel()
    config = GRPOConfig.from_mapping({"group_size": 2, "batch_size": 2, "max_new_tokens": 1})
    reward_weights = RewardWeights.from_mapping(config.reward_weights.dict())

    trainer = GRPOTrainer(
        model,
        ref_model,
        tokenizer,
        config,
        reward_weights,
        device=torch.device("cpu"),
        output_dir=tmp_path / "hf_runs",
    )

    assert trainer.output_dir == (tmp_path / "hf_runs").resolve()
    assert trainer.metrics_path == trainer.output_dir / "metrics.jsonl"
    assert trainer.summary_path == trainer.output_dir / "summary.json"
    assert trainer.reward_calculator is trainer._reward_calculator
    assert trainer.logged_metrics == []
    assert trainer.global_step == 0

    trainer.save_summary()
    assert trainer.summary_path.exists()

    trainer.save_checkpoint(3)
    checkpoint = trainer.output_dir / "checkpoints" / "checkpoint_00003.json"
    assert checkpoint.exists()
