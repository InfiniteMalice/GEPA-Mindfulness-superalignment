import json
import os
import subprocess
import sys
from pathlib import Path


def _write_trace_file(path: Path) -> None:
    events = [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "stage": "framing",
            "content": "Frame the ethical question",
            "principle_scores": {"mindfulness": 0.9},
            "imperative_scores": {"Reduce Suffering": 0.8},
        },
        {
            "timestamp": "2025-01-01T00:00:05Z",
            "stage": "decision",
            "content": "Provide careful answer",
            "principle_scores": {"mindfulness": 0.85},
            "imperative_scores": {"Increase Knowledge": 0.75},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            json.dump(event, handle)
            handle.write("\n")


def _write_tokens_file(path: Path) -> None:
    records = [
        {"token": "Frame", "logprob": -0.1, "abstained": False},
        {"token": "Decision", "logprob": -0.05, "abstained": False},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
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


def test_gepa_cli_smoke(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    tokens_path = tmp_path / "tokens.jsonl"
    report_path = tmp_path / "report.html"
    viewer_path = tmp_path / "viewer.html"

    _write_trace_file(trace_path)
    _write_tokens_file(tokens_path)

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
    assert "GEPA Score Summary" in report_path.read_text(encoding="utf-8")

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
    content = viewer_path.read_text(encoding="utf-8")
    assert "trace" in content and "tokens" in content
