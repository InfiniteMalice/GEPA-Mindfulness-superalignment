"""Utility classes for loading training datasets."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence


@dataclass
class DatasetExample:
    """Single example used during training."""

    prompt: str
    references: Sequence[str] | None
    gepa_scores: Mapping[str, float] | None
    imperatives: Mapping[str, Mapping[str, float]] | None


class DatasetBatch:
    """Wrapper providing simple batching helpers."""

    def __init__(self, items: Sequence[DatasetExample]) -> None:
        self.items = list(items)
        if not self.items:
            raise ValueError("Dataset must contain at least one example")

    @classmethod
    def from_path(cls, path: Path) -> "DatasetBatch":
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            items = [cls._example_from_json(line) for line in path.read_text().splitlines() if line]
        else:
            items = [
                DatasetExample(
                    prompt=line.strip(),
                    references=None,
                    gepa_scores=None,
                    imperatives=None,
                )
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        return cls(items)

    @staticmethod
    def _example_from_json(payload: str) -> DatasetExample:
        data = json.loads(payload)
        prompt = data.get("prompt") or data.get("question")
        if not isinstance(prompt, str):
            raise ValueError("JSON example missing 'prompt' field")
        references = data.get("answers") or data.get("reference")
        if isinstance(references, str):
            ref_list: Sequence[str] | None = [references]
        elif isinstance(references, list):
            ref_list = [str(item) for item in references]
        else:
            ref_list = None
        gepa_scores = data.get("gepa_scores")
        if isinstance(gepa_scores, dict):
            gepa_scores = {str(k): float(v) for k, v in gepa_scores.items()}
        else:
            gepa_scores = None
        imperatives = data.get("imperatives")
        if isinstance(imperatives, dict):
            processed: Dict[str, Dict[str, float]] = {}
            for key, payload in imperatives.items():
                if isinstance(payload, dict):
                    processed[key] = {
                        "support": float(payload.get("support", 0.0)),
                        "opposition": float(payload.get("opposition", 0.0)),
                    }
            imperatives = processed
        else:
            imperatives = None
        return DatasetExample(
            prompt=prompt,
            references=ref_list,
            gepa_scores=gepa_scores,
            imperatives=imperatives,
        )

    def sample_batch(self, batch_size: int) -> List[DatasetExample]:
        batch_size = min(batch_size, len(self.items))
        return random.sample(self.items, batch_size)

    def iter_batches(self, batch_size: int) -> Iterator[List[DatasetExample]]:
        indices = list(range(len(self.items)))
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            yield [self.items[idx] for idx in chunk]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)


__all__ = ["DatasetBatch", "DatasetExample"]
