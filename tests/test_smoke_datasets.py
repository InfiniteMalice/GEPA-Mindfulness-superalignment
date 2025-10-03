import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

DATASET_GLOBS: Tuple[str, ...] = (
    "datasets/*/sample.jsonl",
    "gepa_datasets/*/sample.jsonl",
)


def _discover_dataset_files() -> list[Path]:
    files: list[Path] = []
    for pattern in DATASET_GLOBS:
        for candidate in Path().glob(pattern):
            if candidate.is_file():
                files.append(candidate)
    return sorted(files)


def _trace_events(record: dict, dataset_name: str) -> Iterable[dict]:
    base_content = record.get("question") or record.get("prompt") or record.get("inquiry") or ""
    answer = record.get("answer") or record.get("response") or record.get("solution") or ""
    yield {
        "timestamp": "2025-01-01T00:00:00Z",
        "stage": "framing",
        "content": f"{dataset_name}: {base_content[:80]}",
        "principle_scores": {"mindfulness": 0.8, "empathy": 0.75},
        "imperative_scores": {"Reduce Suffering": 0.85, "Increase Knowledge": 0.7},
    }
    yield {
        "timestamp": "2025-01-01T00:00:05Z",
        "stage": "decision",
        "content": f"Answer: {answer[:80]}",
        "principle_scores": {"mindfulness": 0.82, "agency": 0.7},
        "imperative_scores": {"Increase Knowledge": 0.78},
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


def _run_cli(*args: str) -> None:
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    extra_paths = [str(project_root), str(src_dir)]
    existing = env.get("PYTHONPATH", "")
    joined = os.pathsep.join(path for path in [existing, *extra_paths] if path)
    env["PYTHONPATH"] = joined
    subprocess.run(
        [sys.executable, "-m", "mindful_trace_gepa", *args],
        check=True,
        env=env,
    )


def test_dataset_smoke_runs(tmp_path: Path) -> None:
    dataset_files = _discover_dataset_files()
    assert dataset_files, "Expected at least one dataset sample to be present"

    for index, sample_path in enumerate(dataset_files):
        first_line = sample_path.read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)

        dataset_name = sample_path.parent.name
        trace_path = tmp_path / f"{dataset_name}_trace.jsonl"
        tokens_path = tmp_path / f"{dataset_name}_tokens.jsonl"
        report_path = tmp_path / f"{dataset_name}_report.html"
        viewer_path = tmp_path / f"{dataset_name}_viewer.html"

        _write_jsonl(trace_path, _trace_events(record, dataset_name))

        if index % 2 == 0:
            _write_jsonl(tokens_path, [{"token": "demo", "logprob": -0.1, "abstained": False}])
        else:
            tokens_path.parent.mkdir(parents=True, exist_ok=True)
            if tokens_path.exists():
                tokens_path.unlink()

        _run_cli(
            "score",
            "--trace",
            str(trace_path),
            "--policy",
            str(Path("policies/default_cw4.yml")),
            "--out",
            str(report_path),
        )
        assert report_path.exists()

        _run_cli(
            "view",
            "--trace",
            str(trace_path),
            "--tokens",
            str(tokens_path),
            "--out",
            str(viewer_path),
        )
        assert viewer_path.exists()
        html = viewer_path.read_text(encoding="utf-8")
        assert "GEPA" in html or "trace" in html
