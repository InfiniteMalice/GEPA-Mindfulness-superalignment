import pytest

from mindful_trace_gepa import cli


def test_dspy_run_accepts_dual_path_flag(monkeypatch: pytest.MonkeyPatch):
    called = {}

    def fake_handler(args):
        called["dual_path"] = getattr(args, "dual_path", None)
        called["input"] = args.input
        called["trace"] = args.trace

    monkeypatch.setattr(cli, "handle_dspy_run", fake_handler)

    parser = cli.build_parser()
    args = parser.parse_args(
        ["dspy", "run", "--dual-path", "--input", "input.jsonl", "--trace", "trace.jsonl"]
    )

    args.func(args)

    assert called == {
        "dual_path": True,
        "input": "input.jsonl",
        "trace": "trace.jsonl",
    }
