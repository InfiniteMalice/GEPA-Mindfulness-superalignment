from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from mindful_trace_gepa.cli import handle_dspy_run
from mindful_trace_gepa.configuration import load_dspy_config
from mindful_trace_gepa.dspy_modules.pipeline import GEPAChain


def test_dspy_disabled_pipeline_produces_checkpoints(tmp_path: Path) -> None:
    config = load_dspy_config()
    assert config.enabled is False
    chain = GEPAChain(config=config)
    result = chain.run("What safeguards apply?", context="policy")
    assert result.checkpoints, "Expected checkpoints even when DSPy disabled"

    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"inquiry": "How to stay mindful?"}\n', encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    args = SimpleNamespace(
        input=str(input_path),
        trace=str(trace_path),
        context=None,
        model=None,
        enable_optim=False,
    )
    handle_dspy_run(args)
    assert trace_path.exists()
    contents = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "Trace JSONL should not be empty"
