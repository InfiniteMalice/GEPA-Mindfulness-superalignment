"""CLI helpers for the automated wisdom scoring pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from .scoring import (
    LLMJudge,
    aggregate_tiers,
    load_classifier_from_config,
    run_heuristics,
    write_scoring_artifacts,
)
from .scoring.aggregate import build_config
from .scoring.llm_judge import JudgeConfig
from .scoring.schema import AggregateScores, TierScores
from .storage import iter_jsonl


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_scoring_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return build_config()
    data = _load_yaml(path)
    if not isinstance(data, Mapping):
        return build_config()
    return build_config(data)


def _load_trace_events(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def handle_score_auto(args: argparse.Namespace) -> None:
    trace_path = Path(args.trace)
    config_path = Path(args.config) if getattr(args, "config", None) else None
    scoring_config = _load_scoring_config(config_path)

    events = _load_trace_events(trace_path)
    tiers: List[TierScores] = []

    heuristic_scores = run_heuristics(events)
    tiers.append(heuristic_scores)

    policy_blob: Dict[str, Any] = {}
    if args.policy and Path(args.policy).exists():
        policy_text = Path(args.policy).read_text(encoding="utf-8")
        if yaml is None:
            policy_blob = {"raw": policy_text}
        else:
            policy_blob = yaml.safe_load(policy_text) or {}

    if getattr(args, "judge", False):
        judge_model = scoring_config.get("judge_model", "gpt-sim-judge")
        judge = LLMJudge(JudgeConfig(model=str(judge_model)))
        tiers.append(judge.score_trace(events))

    classifier_scores: Optional[TierScores] = None
    if getattr(args, "classifier", False):
        classifier_config_path = getattr(args, "classifier_config", "configs/classifier/default.yml")
        classifier = load_classifier_from_config(classifier_config_path)
        artifacts_path = getattr(args, "classifier_artifacts", "artifacts/classifier")
        try:
            classifier.load(artifacts_path)
        except FileNotFoundError:
            # Graceful fallback – classifier will mimic heuristics
            pass
        classifier_scores = classifier.predict(events)
        tiers.append(classifier_scores)

    aggregate = aggregate_tiers(tiers, scoring_config)
    extras = {
        "policy": policy_blob,
        "config": {
            "scoring": str(config_path) if config_path else None,
            "classifier": getattr(args, "classifier_config", None),
        },
    }
    write_scoring_artifacts(aggregate, args.out, trace_path=trace_path, extras=extras)

    if getattr(args, "print", True):
        print(json.dumps(aggregate.as_json(), indent=2))


def handle_judge_run(args: argparse.Namespace) -> None:
    trace_path = Path(args.trace)
    events = _load_trace_events(trace_path)
    judge = LLMJudge(JudgeConfig(model=args.model or "gpt-sim-judge", mock=args.mock))
    tier = judge.score_trace(events)
    payload = tier.dict()
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def handle_classifier_train(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels)
    config_path = Path(args.config)
    out_path = Path(args.out)

    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    classifier = load_classifier_from_config(config_path)
    classifier.fit(rows)
    out_path.mkdir(parents=True, exist_ok=True)
    classifier.save(out_path)

    metrics = {
        "temperature": classifier.temperature,
        "feature_names": classifier.feature_names,
        "num_examples": len(rows),
    }
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def handle_lowconf_triage(args: argparse.Namespace) -> None:
    scores_path = Path(args.scores)
    out_path = Path(args.out)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found at {scores_path}")
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    aggregate = AggregateScores.from_dict(payload)
    rows: List[Dict[str, Any]] = []
    for dim, conf in aggregate.confidence.items():
        if conf < args.threshold:
            rows.append({
                "dimension": dim,
                "confidence": conf,
                "score": aggregate.final.get(dim),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    scoring = subparsers.add_parser("score-auto", help="Run tiered wisdom scoring")
    scoring.add_argument("--trace", required=True, help="Trace JSONL path")
    scoring.add_argument("--policy", help="Optional policy YAML")
    scoring.add_argument("--out", required=True, help="Output scores JSON")
    scoring.add_argument("--config", help="Scoring config YAML (weights, thresholds)")
    scoring.add_argument("--judge", action="store_true", help="Include LLM judge tier")
    scoring.add_argument("--classifier", action="store_true", help="Include classifier tier")
    scoring.add_argument("--classifier-config", help="Classifier config YAML")
    scoring.add_argument("--classifier-artifacts", help="Directory with trained classifier")
    scoring.add_argument("--no-print", action="store_false", dest="print", help="Suppress stdout summary")
    scoring.set_defaults(func=handle_score_auto)

    judge = subparsers.add_parser("judge", help="Interact with the tiered judge")
    judge.add_argument("--trace", required=True, help="Trace JSONL path")
    judge.add_argument("--out", required=True, help="Where to write judge response")
    judge.add_argument("--model", help="Model name override")
    judge.add_argument("--mock", action="store_true", help="Force mock judge response")
    judge.set_defaults(func=handle_judge_run)

    clf = subparsers.add_parser("clf", help="Classifier utilities")
    clf_sub = clf.add_subparsers(dest="classifier_command")

    clf_train = clf_sub.add_parser("train", help="Train the tier-2 classifier")
    clf_train.add_argument("--labels", required=True, help="Labelled dataset JSONL")
    clf_train.add_argument("--config", required=True, help="Classifier config YAML")
    clf_train.add_argument("--out", required=True, help="Artifacts directory")
    clf_train.set_defaults(func=handle_classifier_train)

    triage = clf_sub.add_parser("triage", help="Export low-confidence dimensions")
    triage.add_argument("--scores", required=True, help="Aggregate scores JSON")
    triage.add_argument("--out", required=True, help="Output JSONL path")
    triage.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold")
    triage.set_defaults(func=handle_lowconf_triage)
