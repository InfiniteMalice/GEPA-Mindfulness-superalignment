"""Extract contrastive fingerprints from dual-path runs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.debug("JSONL file not found: %s", path)
        return []

    records = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d: %s", line_num, line[:100])
                continue
    return records


def _split_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter; may split on abbreviations like "e.g." or "U.S.".

    Returns all sentences found; no hard limit on count or length.
    """
    candidates = re.split(r"([.?!]+(?:\s+|$))", text)
    sentences = []
    for i in range(0, len(candidates) - 1, 2):
        sentence = (candidates[i] + candidates[i + 1]).strip()
        if sentence:
            sentences.append(sentence)
    if len(candidates) % 2 == 1 and candidates[-1].strip():
        sentences.append(candidates[-1].strip())
    return sentences


def _token_counts(text: str) -> Counter[str]:
    tokens = []
    for token in text.split():
        cleaned = token.strip(".,!?;:\"'()[]{}").lower()
        if cleaned:
            tokens.append(cleaned)
    return Counter(tokens)


def _contrastive_terms(path_1: Counter[str], path_2: Counter[str]) -> list[str]:
    all_tokens = set(path_1.keys()) | set(path_2.keys())
    scores = {token: path_1.get(token, 0) - path_2.get(token, 0) for token in all_tokens}
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
        "contrastive_tokens": tokens,
        "neurons": neurons,
        "attention_heads": attention_heads,
        "mlp_blocks": mlp_blocks,
    }


def extract_fingerprint(run: Mapping[str, Any]) -> dict[str, Any]:
    path1_text = " ".join([run.get("path_1_scratchpad") or "", run.get("path_1") or ""])
    path2_text = " ".join([run.get("path_2_scratchpad") or "", run.get("path_2") or ""])
    scratchpad_text = run.get("scratchpad") or ""

    path1_counts = _token_counts(path1_text)
    path2_counts = _token_counts(path2_text)
    contrastive = _contrastive_terms(path1_counts, path2_counts)
    trace = _split_sentences(scratchpad_text)
    attribution = {
        "path_1_tokens": dict(path1_counts),
        "path_2_tokens": dict(path2_counts),
    }

    return {
        "id": run.get("id"),
        "final_answer_value": run.get("final_answer_value"),
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
