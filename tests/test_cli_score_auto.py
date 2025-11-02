import argparse
import json

import pytest

pytest.importorskip(
    "mindful_trace_gepa",
    reason="mindful_trace_gepa package is required for CLI scoring tests.",
)

from mindful_trace_gepa.cli_scoring import handle_score_auto
from mindful_trace_gepa.scoring.schema import DIMENSIONS, TierScores


def test_score_auto_generates_output(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    events = [
        {
            "content": (
                "We assume 30% uptake with monitoring; stakeholders include users "
                "and regulators."
            ),
            "gepa_hits": ["monitor"],
        },
        {
            "content": "Policy GEPA-1 forbids deception; compare option A vs contingency B.",
        },
    ]
    trace_path.write_text("\n".join(json.dumps(evt) for evt in events), encoding="utf-8")

    out_path = tmp_path / "scores.json"
    monkeypatch.setenv("GEPA_JUDGE_MOCK", "1")
    args = argparse.Namespace(
        trace=str(trace_path),
        policy=None,
        out=str(out_path),
        config="configs/scoring.yml",
        judge=True,
        classifier=False,
        classifier_config=None,
        classifier_artifacts=None,
        print=False,
    )
    handle_score_auto(args)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "final" in payload and "confidence" in payload
    assert payload["per_tier"]
    assert any(tier["tier"] == "judge" for tier in payload["per_tier"])


def test_score_auto_partial_weights_merge(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps({"content": "placeholder"}), encoding="utf-8")

    config_path = tmp_path / "weights.yml"
    config_path.write_text("weights:\n  judge: 0.4\n", encoding="utf-8")

    def fake_run_heuristics(events):
        scores = {dim: 4 for dim in DIMENSIONS}
        confidence = {dim: 1.0 for dim in DIMENSIONS}
        return TierScores(tier="heuristic", scores=scores, confidence=confidence, meta={})

    class DummyJudge:
        def __init__(self, config) -> None:  # pragma: no cover - simple stub
            pass

        def score_trace(self, events):
            scores = {dim: 2 for dim in DIMENSIONS}
            confidence = {dim: 1.0 for dim in DIMENSIONS}
            return TierScores(tier="judge", scores=scores, confidence=confidence, meta={})

    monkeypatch.setattr("mindful_trace_gepa.cli_scoring.run_heuristics", fake_run_heuristics)
    monkeypatch.setattr("mindful_trace_gepa.cli_scoring.LLMJudge", DummyJudge)

    out_path = tmp_path / "scores.json"

    args = argparse.Namespace(
        trace=str(trace_path),
        policy=None,
        out=str(out_path),
        config=str(config_path),
        judge=True,
        classifier=False,
        classifier_config=None,
        classifier_artifacts=None,
        print=False,
    )

    handle_score_auto(args)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["final"]["mindfulness"] == 3


def test_score_auto_none_classifier_weight_uses_default(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps({"content": "placeholder"}), encoding="utf-8")

    config_path = tmp_path / "weights.yml"
    config_path.write_text("weights:\n  judge: 0.4\n  classifier: null\n", encoding="utf-8")

    def fake_run_heuristics(events):
        scores = {dim: 4 for dim in DIMENSIONS}
        confidence = {dim: 1.0 for dim in DIMENSIONS}
        return TierScores(tier="heuristic", scores=scores, confidence=confidence, meta={})

    class DummyJudge:
        def __init__(self, config) -> None:  # pragma: no cover - simple stub
            pass

        def score_trace(self, events):
            scores = {dim: 2 for dim in DIMENSIONS}
            confidence = {dim: 1.0 for dim in DIMENSIONS}
            return TierScores(tier="judge", scores=scores, confidence=confidence, meta={})

    class DummyClassifier:
        def __init__(self, config) -> None:  # pragma: no cover - simple stub
            pass

        def load(self, path):  # pragma: no cover - stub does nothing
            return None

        def predict(self, events):
            scores = {dim: 4 for dim in DIMENSIONS}
            confidence = {dim: 1.0 for dim in DIMENSIONS}
            return TierScores(tier="classifier", scores=scores, confidence=confidence, meta={})

    monkeypatch.setattr("mindful_trace_gepa.cli_scoring.run_heuristics", fake_run_heuristics)
    monkeypatch.setattr("mindful_trace_gepa.cli_scoring.LLMJudge", DummyJudge)

    def load_classifier(path):
        return DummyClassifier(path)

    monkeypatch.setattr(
        "mindful_trace_gepa.cli_scoring.load_classifier_from_config",
        load_classifier,
    )

    out_path = tmp_path / "scores.json"

    args = argparse.Namespace(
        trace=str(trace_path),
        policy=None,
        out=str(out_path),
        config=str(config_path),
        judge=True,
        classifier=True,
        classifier_config="unused",
        classifier_artifacts=str(tmp_path / "artifacts"),
        print=False,
    )

    handle_score_auto(args)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["final"]["mindfulness"] == 3
