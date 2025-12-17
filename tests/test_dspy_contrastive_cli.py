from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindful_trace_gepa import cli


def test_dspy_contrastive_subcommand_invokes_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "data.jsonl"
    data_payload = json.dumps({"id": "example", "query": "How to proceed?"}) + "\n"
    data_path.write_text(data_payload)
    out_dir = tmp_path / "runs"

    called: dict[str, object] = {}

    def fake_runner(data: Path, out: Path, context: str) -> None:
        called["data"] = data
        called["out"] = out
        called["context"] = context

    monkeypatch.setattr(cli, "run_dual_path_contrastive", fake_runner)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "dspy",
            "contrastive-run",
            "--data",
            str(data_path),
            "--out",
            str(out_dir),
            "--context",
            "general",
        ]
    )

    args.func(args)

    assert called == {"data": data_path, "out": out_dir, "context": "general"}
