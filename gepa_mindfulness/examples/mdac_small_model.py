"""Local one-token GRPO smoke adapter for the MDAC alignment-first lifecycle."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.causal_records import content_digest
from evaluation.mdac_comparisons import Completion, Predict, TrainingRequest, TrainingRun
from gepa_mindfulness.training.algorithms.base import AlgorithmBatch
from gepa_mindfulness.training.algorithms.grpo import (
    GRPOAlgorithmConfig,
    compute_group_advantages,
    compute_grpo_loss,
)
from gepa_mindfulness.training.backends.base import TorchTensorOps
from gepa_mindfulness.training.trajectory import PolicyEvaluation


class LocalOneTokenGRPO:
    """Exercise actual local parameters with two declared action tokens and positive rewards."""

    def __init__(self, source: Path, destination: Path, labels: dict[str, str]) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True).cpu().eval()
        self.reference = model
        self.destination = destination
        destination.mkdir(parents=True, exist_ok=True)
        self.config_digest = content_digest(model.config.to_dict())
        self.start = self._identity(model)
        self.checkpoints = {self.start: source}
        self.labels = dict(labels)
        first = self.tokenizer.eos_token_id
        other = self.tokenizer.encode("9", add_special_tokens=False)
        if first is None or not other or first == other[-1]:
            raise ValueError("local tokenizer needs distinct end-of-sequence and numeral tokens")
        self.actions = (first, other[-1])
        self.baseline = self._predict(model)

    @staticmethod
    def _identity(model) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return "sha256:" + digest.hexdigest()

    def _logits(self, model, prompt):
        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).input_ids
        return model(input_ids=ids).logits[0, -1, list(self.actions)]

    def _predict(self, model) -> Predict:
        def predict(prompt: str, seed: int, limit: int) -> Completion:
            with torch.no_grad():
                choice = int(self._logits(model, prompt).argmax())
            return Completion(("disclose", "solve")[choice], 1)

        return predict

    def train(self, request: TrainingRequest, targets: tuple[str | None, ...]) -> TrainingRun:
        if request.method != "grpo" or request.model_config_digest != self.config_digest:
            raise ValueError("checkpoint trainer requires its frozen GRPO configuration")
        if request.runtime_options != "{}":
            raise ValueError("the one-token smoke uses teacherless GRPO")
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.checkpoints[request.checkpoint], local_files_only=True
            )
            .cpu()
            .eval()
        )
        if self._identity(model) != request.checkpoint:
            raise ValueError("loaded parent checkpoint differs from its recorded weights")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(request.seed)
        tokens = 0
        for prompt, target in zip(request.prompts, targets, strict=True):
            if target not in ("disclose", "solve") or self.labels.get(prompt) != target:
                raise ValueError("training prompt requires a reviewed actor target")
            logits = self._logits(model, prompt)
            choices = torch.multinomial(
                logits.detach().softmax(0), 4, replacement=True, generator=generator
            )
            wanted = int(target == "solve")
            rewards = [float(c == wanted) for c in choices.tolist()]
            advantages = compute_group_advantages(rewards)
            assert advantages is not None
            log_probs = logits.log_softmax(0)[choices].reshape(-1, 1)
            batch = AlgorithmBatch(
                log_probs.detach(),
                torch.tensor(advantages).reshape(-1, 1),
                torch.ones_like(log_probs, dtype=torch.bool),
            )
            with torch.no_grad():
                reference = (
                    self._logits(self.reference, prompt).log_softmax(0)[choices].reshape(-1, 1)
                )
            entropy = -(logits.softmax(0) * logits.log_softmax(0)).sum().expand_as(log_probs)
            loss = compute_grpo_loss(
                TorchTensorOps(),
                batch,
                PolicyEvaluation(log_probs, reference, entropy=entropy),
                GRPOAlgorithmConfig(),
            )
            optimizer.zero_grad()
            loss.total_loss.backward()
            optimizer.step()
            tokens += len(choices)
        end = self._identity(model)
        destination = self.destination / end.split(":")[1]
        model.save_pretrained(destination)
        self.checkpoints[end] = destination
        return TrainingRun(
            request.checkpoint,
            end,
            len(request.prompts),
            len(request.prompts),
            self._predict(model),
            request.digest,
            tokens,
            self.config_digest,
        )


def initialize_smoke_checkpoint(directory: Path, seed: int = 7) -> None:
    """Write a random tiny GPT-2 and public toy tokenizer into a new local directory."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    directory.mkdir(parents=True, exist_ok=False)
    words = ["<unk>", "<pad>", "<eos>", "9", "Ethics", "Reason", "Recall", "Count", "Sort"]
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(words),
                n_embd=8,
                n_layer=1,
                n_head=1,
                n_positions=256,
                bos_token_id=2,
                eos_token_id=2,
                pad_token_id=1,
            )
        )
    model.save_pretrained(directory)
    fast.save_pretrained(directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create the random local GRPO smoke checkpoint.")
    parser.add_argument("--initialize", type=Path, required=True)
    initialize_smoke_checkpoint(parser.parse_args().initialize)
