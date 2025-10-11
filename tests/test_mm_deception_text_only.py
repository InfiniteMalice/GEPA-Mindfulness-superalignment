import json
from pathlib import Path

from mindful_trace_gepa.data.mm_deception import (
    DeceptionExample,
    iter_text_examples,
    load_mu3d_text_only,
    load_opspam_text_only,
    load_rltd_text_only,
)


def _write_split(base: Path, split: str, rows) -> None:
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_rows() -> list[dict[str, object]]:
    return [
        {"id": "a1", "transcript": "Honest update", "label": 0},
        {"id": "a2", "transcript": "Deceptive summary", "label": 1},
    ]


def test_loaders_text_only(tmp_path: Path) -> None:
    rltd_dir = tmp_path / "RLTD"
    mu3d_dir = tmp_path / "MU3D"
    opspam_dir = tmp_path / "OpSpam"

    for base in (rltd_dir, mu3d_dir, opspam_dir):
        for split in ("train", "validation", "test"):
            _write_split(base, split, _build_rows())

    rltd = load_rltd_text_only(rltd_dir, max_samples=4)
    mu3d = load_mu3d_text_only(mu3d_dir, max_samples=4)
    opspam = load_opspam_text_only(opspam_dir, max_samples=4)

    assert rltd["train"], "RLTD train split should not be empty"
    assert mu3d["validation"], "MU3D validation split should not be empty"
    assert opspam["test"], "OpSpam test split should not be empty"

    examples = list(iter_text_examples(rltd, splits=["train"]))
    assert all(isinstance(item, DeceptionExample) for item in examples)
    assert examples[0].text
    assert examples[0].meta == {"transcript": "Honest update"}
