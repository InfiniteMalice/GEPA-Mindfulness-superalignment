"""Pytest fixtures for Phase 0 modules."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
MINDFUL_TRACE_AVAILABLE = importlib.util.find_spec("mindful_trace_gepa") is not None

if not MINDFUL_TRACE_AVAILABLE:  # pragma: no cover - optional dependency
    collect_ignore = [
        "test_aggregate_confidence.py",
        "test_classifier_smoke.py",
        "test_cli_minimal.py",
        "test_cli_score_auto.py",
        "test_complete_integration.py",
        "test_corrected_integration.py",
        "test_deception_probe_synthetic.py",
        "test_deception_signals.py",
        "test_detectors.py",
        "test_dspy_compile.py",
        "test_dspy_disabled.py",
        "test_dual_path.py",
        "test_llm_judge_schema.py",
        "test_longctx_stream_score.py",
        "test_mm_deception_text_only.py",
        "test_shards_manifest.py",
        "test_smoke_datasets.py",
        "test_tier0_heuristics.py",
        "test_viewer_build.py",
    ]

if TORCH_AVAILABLE:  # pragma: no branch - import guard
    import torch  # type: ignore
else:  # pragma: no cover - executed only when torch missing
    torch = None  # type: ignore

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:  # pragma: no branch - import guard
    from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore
else:  # pragma: no cover - executed only when deps missing
    GPT2Config = GPT2LMHeadModel = Any  # type: ignore


@dataclass
class BatchEncoding(dict):
    """Minimal batch encoding mimicking Hugging Face outputs."""

    def to(self, device: "torch.device") -> "BatchEncoding":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to move BatchEncoding tensors.")
        return BatchEncoding({key: value.to(device) for key, value in self.items()})


class DummyTokenizer:
    """Whitespace tokenizer compatible with small GPT-2 models."""

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {
            "<pad>": 0,
            "<eos>": 1,
            "hello": 2,
            "world": 3,
            "good": 4,
            "model": 5,
            "response": 6,
            "test": 7,
            "context": 8,
            "value": 9,
        }
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [token for token in text.strip().split(" ") if token]
        if add_special_tokens:
            tokens.append(self.eos_token)
        return [self._token_id(token) for token in tokens]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if skip_special_tokens and token in {self.pad_token, self.eos_token}:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def __call__(self, text: str, return_tensors: str = "pt") -> BatchEncoding:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to tensorise dummy tokenizer outputs.")
        token_ids = self.encode(text, add_special_tokens=False)
        tensor = torch.tensor([token_ids], dtype=torch.long)
        attention = torch.ones_like(tensor)
        return BatchEncoding({"input_ids": tensor, "attention_mask": attention})

    def _token_id(self, token: str) -> int:
        if token not in self.token_to_id:
            raise ValueError(f"unknown token: {token}")
        return self.token_to_id[token]


@pytest.fixture()
def tiny_model() -> GPT2LMHeadModel:
    if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
        pytest.skip("PyTorch and transformers are required for the tiny_model fixture.")
    config = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.config._name_or_path = "tiny-gpt2"
    return model


@pytest.fixture()
def dummy_tokenizer() -> DummyTokenizer:
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch is required for the dummy_tokenizer fixture.")
    return DummyTokenizer()
