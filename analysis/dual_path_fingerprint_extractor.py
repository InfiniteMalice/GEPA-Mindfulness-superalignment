"""Extract contrastive fingerprints from dual-path runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _split_sentences(text: str) -> list[str]:
    sentences = []
    buffer = []
    for char in text:
        buffer.append(char)
        if char in {".", "?", "!"}:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _token_counts(text: str) -> Counter[str]:
    tokens = [token.strip(".,!?;:\"'()[]{}").lower() for token in text.split() if token.strip()]
    return Counter(tokens)


def _contrastive_terms(path_1: Counter[str], path_2: Counter[str]) -> list[str]:
    scores: dict[str, int] = {}
    for token, count in path_1.items():
        scores[token] = scores.get(token, 0) + count
    for token, count in path_2.items():
        scores[token] = scores.get(token, 0) - count
    non_zero = ((token, delta) for token, delta in scores.items() if delta != 0)
    ranked = sorted(non_zero, key=lambda item: abs(item[1]), reverse=True)
    return [token for token, _ in ranked[:10]]


def _ablation_targets(tokens: list[str]) -> dict[str, list[str]]:
    def _stable_bucket(token: str, modulus: int) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest, "big") % modulus

    neurons = [f"neuron_{_stable_bucket(token, 128)}" for token in tokens]
    attention_heads = [f"head_{_stable_bucket(token, 64)}" for token in tokens]
    mlp_blocks = [f"mlp_{_stable_bucket(token, 32)}" for token in tokens]
    return {
        "feature_sets": tokens,
        "neurons": neurons,
        "attention_heads": attention_heads,
        "mlp_blocks": mlp_blocks,
    }


def extract_fingerprint(run: Mapping[str, Any]) -> dict[str, Any]:
    path1_text = " ".join(
        [
            str(run.get("path_1_scratchpad", "")),
            str(run.get("path_1", "")),
        ]
    )
    path2_text = " ".join(
        [
            str(run.get("path_2_scratchpad", "")),
            str(run.get("path_2", "")),
        ]
    )
    scratchpad_text = str(run.get("scratchpad", ""))

    path1_counts = _token_counts(path1_text)
    path2_counts = _token_counts(path2_text)
    contrastive = _contrastive_terms(path1_counts, path2_counts)
    trace = _split_sentences(scratchpad_text)
    attribution = {
        "path_1_tokens": path1_counts,
        "path_2_tokens": path2_counts,
    }

    return {
        "id": run.get("id"),
        "final_answer": run.get("final_answer_value"),
        "contrastive_tokens": contrastive,
        "thought_trace": trace,
        "attribution_map": attribution,
        "ablation_targets": _ablation_targets(contrastive),
    }


def extract_batch(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [extract_fingerprint(record) for record in records]


def load_and_extract(source: Path) -> list[dict[str, Any]]:
    runs = _load_jsonl(source)
    return extract_batch(runs)


__all__ = [
    "extract_batch",
    "extract_fingerprint",
    "load_and_extract",
]
