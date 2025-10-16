"""CLI helpers for the automated wisdom scoring pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

from .scoring import (
    LLMJudge,
    aggregate_tiers,
    build_config,
    load_classifier_from_config,
    run_heuristics,
    write_scoring_artifacts,
)
from .scoring.llm_judge import JudgeConfig
from .scoring.schema import AggregateScores, TierScores, DIMENSIONS
from .storage import iter_jsonl
from .utils.imports import optional_import

yaml_module = optional_import("yaml")
click_module = optional_import("click")


def _echo(message: str, *, err: bool = False) -> None:
    """Emit CLI output using click if available, falling back to print."""

    if click_module is not None:
        click_module.echo(message, err=err)
    else:
        stream = sys.stderr if err else sys.stdout
        print(message, file=stream)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml_module is None:
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    loader = cast(Any, yaml_module)
    return loader.safe_load(path.read_text(encoding="utf-8")) or {}


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
        if yaml_module is None:
            policy_blob = {"raw": policy_text}
        else:
            loader = cast(Any, yaml_module)
            policy_blob = loader.safe_load(policy_text) or {}

    if getattr(args, "judge", False):
        judge_model = scoring_config.get("judge_model", "gpt-sim-judge")
        judge = LLMJudge(JudgeConfig(model=str(judge_model)))
        tiers.append(judge.score_trace(events))

    classifier_scores: Optional[TierScores] = None
    if getattr(args, "classifier", False):
        classifier_config_path = getattr(
            args,
            "classifier_config",
            "configs/classifier/default.yml",
        )
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
    payload = tier.to_dict()
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def handle_classifier_train(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels)
    config_path = Path(args.config)
    out_path = Path(args.out)

    rows = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
            rows.append(
                {
                    "dimension": dim,
                    "confidence": conf,
                    "score": aggregate.final.get(dim),
                }
            )
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
    scoring.add_argument(
        "--no-print",
        action="store_false",
        dest="print",
        help="Suppress stdout summary",
    )
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


if click_module is not None:

    @click_module.command()
    @click_module.option("--trace", required=True, help="Path to trace JSONL")
    @click_module.option("--out", required=True, help="Output scores JSON")
    @click_module.option(
        "--config",
        default="configs/scoring.yml",
        show_default=True,
        help="Scoring config",
    )
    @click_module.option("--mock-judge", is_flag=True, help="Use mock judge for testing")
    def score_auto(trace: str, out: str, config: str, mock_judge: bool) -> None:
        """Run automated scoring with all tiers."""

        config_path = Path(config) if config else None
        scoring_config = _load_scoring_config(config_path)
        events = _load_trace_events(Path(trace))
        tiers: List[TierScores] = [run_heuristics(events)]

        judge_cfg = scoring_config.get("judge", {}) if isinstance(scoring_config, Mapping) else {}
        judge_enabled = True
        if isinstance(judge_cfg, Mapping) and judge_cfg.get("enabled") is False:
            judge_enabled = False
        judge_scores: Optional[TierScores] = None
        if judge_enabled:
            model_name = str(
                (judge_cfg or {}).get("model", scoring_config.get("judge_model", "gpt-sim-judge"))
            )
            temperature = float((judge_cfg or {}).get("temperature", 0.0))
            max_tokens = int((judge_cfg or {}).get("max_tokens", 2000))
            retries = int((judge_cfg or {}).get("retries", 3))
            mock = mock_judge or bool((judge_cfg or {}).get("mock_mode"))
            if mock:
                os.environ["GEPA_JUDGE_MOCK"] = "1"
            try:
                judge = LLMJudge(
                    JudgeConfig(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retries=retries,
                        mock=mock,
                    )
                )
                judge_scores = judge.score_trace(events)
            except Exception as exc:  # pragma: no cover - resilience guard
                fallback_conf = {
                    dim: min(0.2, tiers[0].confidence.get(dim, 0.1)) for dim in DIMENSIONS
                }
                judge_scores = TierScores(
                    tier="judge",
                    scores=dict(tiers[0].scores),
                    confidence=fallback_conf,
                    meta={"fallback": "heuristic", "error": str(exc)},
                )
                _echo("Judge tier unavailable; using heuristic fallback.", err=True)
        if judge_scores is not None:
            tiers.append(judge_scores)

        classifier_cfg = (
            scoring_config.get("classifier", {}) if isinstance(scoring_config, Mapping) else {}
        )
        classifier_scores: Optional[TierScores] = None
        classifier_enabled = False
        if isinstance(classifier_cfg, Mapping):
            classifier_enabled = bool(classifier_cfg.get("enabled", False))
        if classifier_enabled:
            classifier_config_path = (
                classifier_cfg.get("config_path") or "configs/classifier/default.yml"
            )
            classifier = load_classifier_from_config(classifier_config_path)
            artifacts_hint = classifier_cfg.get("artifacts_dir")
            if artifacts_hint:
                artifacts_path = Path(artifacts_hint)
            else:
                model_path = classifier_cfg.get("model_path")
                artifacts_path = (
                    Path(model_path).parent if model_path else Path("artifacts/classifier")
                )
            try:
                classifier.load(artifacts_path)
                classifier_scores = classifier.predict(events)
            except FileNotFoundError:
                fallback_conf = {dim: 0.1 for dim in DIMENSIONS}
                classifier_scores = TierScores(
                    tier="classifier",
                    scores=dict(tiers[0].scores),
                    confidence=fallback_conf,
                    meta={"fallback": "untrained"},
                )
                _echo("Classifier artifacts not found; falling back to heuristics.", err=True)
            except Exception as exc:  # pragma: no cover - resilience guard
                fallback_conf = {dim: 0.1 for dim in DIMENSIONS}
                classifier_scores = TierScores(
                    tier="classifier",
                    scores=dict(tiers[0].scores),
                    confidence=fallback_conf,
                    meta={"fallback": "error", "error": str(exc)},
                )
                _echo("Classifier tier errored; using heuristic fallback.", err=True)
        if classifier_scores is not None:
            tiers.append(classifier_scores)

        aggregate = aggregate_tiers(tiers, scoring_config)
        extras = {
            "config": {
                "scoring": str(config_path) if config_path else None,
                "classifier": classifier_cfg,
            },
        }
        write_scoring_artifacts(aggregate, out, trace_path=trace, extras=extras)
        _echo(json.dumps(aggregate.as_json(), indent=2))

    @click_module.command()
    @click_module.option("--scores", required=True, help="Scores JSON file")
    @click_module.option(
        "--threshold", default=0.75, type=float, show_default=True, help="Confidence threshold"
    )
    @click_module.option("--out", required=True, help="Output triage JSONL")
    def triage_lowconf(scores: str, threshold: float, out: str) -> None:
        """Export low-confidence items for human labeling."""

        scores_path = Path(scores)
        if not scores_path.exists():
            raise FileNotFoundError(f"Scores file not found at {scores_path}")
        payload = json.loads(scores_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        triage_rows: List[Dict[str, Any]] = []
        for item in items:
            confidence = item.get("confidence", {}) or {}
            escalate = bool(item.get("escalate"))
            should_export = escalate or any(
                float(confidence.get(dim, 1.0)) < threshold for dim in confidence
            )
            if should_export:
                triage_rows.append(
                    {
                        "id": item.get("id") or item.get("trace"),
                        "trace": item.get("trace"),
                        "tier_scores": item.get("per_tier"),
                        "reason": item.get("reasons", []),
                    }
                )

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in triage_rows:
                handle.write(json.dumps(row) + "\n")
        _echo(f"Exported {len(triage_rows)} items to {out_path}")

    @click_module.command()
    @click_module.option("--labels", required=True, help="Training labels JSONL")
    @click_module.option("--config", required=True, help="Classifier config")
    @click_module.option("--out", required=True, help="Output directory")
    def clf_train(labels: str, config: str, out: str) -> None:
        """Train Tier-2 classifier."""

        try:
            handle_classifier_train(argparse.Namespace(labels=labels, config=config, out=out))
        except Exception as exc:  # pragma: no cover - resilience guard
            out_path = Path(out)
            out_path.mkdir(parents=True, exist_ok=True)
            (out_path / "model.pt").write_text("# Placeholder model", encoding="utf-8")
            (out_path / "calibration.json").write_text(
                json.dumps({"temperature": 1.0}), encoding="utf-8"
            )
            _echo(
                "Classifier training failed; emitted placeholder artifacts instead.",
                err=True,
            )
            _echo(str(exc), err=True)

else:  # pragma: no cover - click optional dependency missing

    def score_auto(*_: object, **__: object) -> None:
        raise RuntimeError("click is required to use the score_auto command")

    def triage_lowconf(*_: object, **__: object) -> None:
        raise RuntimeError("click is required to use the triage_lowconf command")

    def clf_train(*_: object, **__: object) -> None:
        raise RuntimeError("click is required to use the clf_train command")
