"""Text-first loaders for multimodal deception datasets with CPU-safe fallbacks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass
class DeceptionExample:
    """Simple container for text-only deception examples."""

    identifier: str
    text: str
    label: int
    meta: Dict[str, Any]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        LOGGER.warning("Dataset split missing at %s", path)
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed line in %s: %s", path, exc)
    return rows


def _normalise_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
        return 1 if number >= 0.5 else 0
    except (TypeError, ValueError):
        if isinstance(value, str) and value.lower() in {"true", "deceptive", "yes", "lie"}:
            return 1
    return 0


def _select_text(row: Mapping[str, Any], *candidates: str) -> str:
    for key in candidates:
        if key in row and row[key]:
            return str(row[key])
    return ""


def _load_split(
    base: Path,
    split: str,
    *,
    text_keys: Sequence[str],
    label_key: str,
    max_samples: Optional[int],
    with_mm: bool,
) -> List[Dict[str, Any]]:
    filename = f"{split}.jsonl"
    path = base / filename
    rows = _read_jsonl(path)
    examples: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = _select_text(row, *text_keys, "text", "content", "transcript")
        if label_key in row:
            label = _normalise_label(row.get(label_key))
        else:
            label = _normalise_label(row.get("label"))
        identifier = str(row.get("id") or f"{split}-{index}")
        meta = {
            key: value
            for key, value in row.items()
            if key not in {"id", label_key, "label"}
        }
        if with_mm:
            meta.setdefault(
                "multimodal_assets",
                {
                    "audio": row.get("audio_path"),
                    "video": row.get("video_path"),
                    "image": row.get("image_path"),
                },
            )
        examples.append({"id": identifier, "text": text, "label": label, "meta": meta})
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def load_rltd_text_only(
    path: str | Path,
    *,
    max_samples: Optional[int] = None,
    with_mm: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    base = Path(path)
    return {
        "train": _load_split(
            base,
            "train",
            text_keys=("transcript", "text"),
            label_key="label",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
        "validation": _load_split(
            base,
            "validation",
            text_keys=("transcript", "text"),
            label_key="label",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
        "test": _load_split(
            base,
            "test",
            text_keys=("transcript", "text"),
            label_key="label",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
    }


def load_opspam_text_only(
    path: str | Path,
    *,
    max_samples: Optional[int] = None,
    with_mm: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    base = Path(path)
    return {
        "train": _load_split(
            base,
            "train",
            text_keys=("review", "text"),
            label_key="deceptive",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
        "validation": _load_split(
            base,
            "validation",
            text_keys=("review", "text"),
            label_key="deceptive",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
        "test": _load_split(
            base,
            "test",
            text_keys=("review", "text"),
            label_key="deceptive",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
    }


def load_mu3d_text_only(
    path: str | Path,
    *,
    max_samples: Optional[int] = None,
    with_mm: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    base = Path(path)
    return {
        "train": _load_split(
            base,
            "train",
            text_keys=("dialogue", "text"),
            label_key="label",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
        "validation": _load_split(
            base,
            "validation",
            text_keys=("dialogue", "text"),
            label_key="label",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
        "test": _load_split(
            base,
            "test",
            text_keys=("dialogue", "text"),
            label_key="label",
            max_samples=max_samples,
            with_mm=with_mm,
        ),
    }


def iter_text_examples(
    dataset: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    splits: Optional[Sequence[str]] = None,
) -> Iterator[DeceptionExample]:
    selected = splits or ("train", "validation", "test")
    for split in selected:
        rows = dataset.get(split) or []
        for row in rows:
            yield DeceptionExample(
                identifier=str(row.get("id")),
                text=str(row.get("text", "")),
                label=int(row.get("label", 0)),
                meta=dict(row.get("meta", {})),
            )


__all__ = [
    "DeceptionExample",
    "iter_text_examples",
    "load_mu3d_text_only",
    "load_opspam_text_only",
    "load_rltd_text_only",
]
