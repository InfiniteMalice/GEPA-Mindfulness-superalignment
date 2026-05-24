"""CLI runner for the GEPA Mindfulness alignment evaluation battery."""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from evaluation.registry import STANDARD_CATEGORIES, get_suite_metadata, list_alignment_suites
from evaluation.schema import EvalCase, EvalResult, write_results_jsonl
from evaluation.suites.calibration.brier_score import summarize_calibration
from evaluation.suites.common import load_responses, score_response

DEFAULT_MODEL_TIMEOUT_SECONDS = 60.0

ADAPTER_MODULES: dict[str, str] = {
    "simpleqa": "evaluation.suites.factuality.simpleqa_adapter",
    "truthfulqa": "evaluation.suites.factuality.truthfulqa_adapter",
    "fever": "evaluation.suites.factuality.fever_adapter",
    "halueval": "evaluation.suites.factuality.halueval_adapter",
    "helm_calibration": "evaluation.suites.calibration.helm_calibration_adapter",
    "anthropic_sycophancy": "evaluation.suites.sycophancy.anthropic_sycophancy_adapter",
    "user_belief_flip": "evaluation.suites.sycophancy.user_belief_flip_tests",
    "apollo_scheming": "evaluation.suites.deception.apollo_style_scheming",
    "sandbagging": "evaluation.suites.deception.sandbagging_probe",
    "followup_interview": "evaluation.suites.deception.followup_interview_probe",
    "harmbench": "evaluation.suites.robustness.harmbench_adapter",
    "jailbreakbench": "evaluation.suites.robustness.jailbreakbench_adapter",
    "prompt_injection_hierarchy": "evaluation.suites.robustness.prompt_injection_hierarchy",
    "agentharm": "evaluation.suites.agent_safety.agentharm_adapter",
    "tool_misuse": "evaluation.suites.agent_safety.tool_misuse_scenarios",
    "wmdp": "evaluation.suites.misuse_capability.wmdp_adapter",
    "bbq": "evaluation.suites.bias_toxicity.bbq_adapter",
    "realtoxicity": "evaluation.suites.bias_toxicity.realtoxicity_adapter",
    "decodingtrust": "evaluation.suites.bias_toxicity.decodingtrust_adapter",
    "pii_leakage": "evaluation.suites.privacy.pii_leakage_probe",
    "canary_extraction": "evaluation.suites.privacy.canary_extraction_probe",
    "conversation_boundary": "evaluation.suites.privacy.conversation_boundary_probe",
    "ood_robustness": "evaluation.suites.ood.ood_robustness_adapter",
}

FIXTURE_BY_SUITE: dict[str, str] = {
    "simpleqa": "simpleqa_toy.jsonl",
    "truthfulqa": "truthfulqa_toy.jsonl",
    "helm_calibration": "simpleqa_toy.jsonl",
    "anthropic_sycophancy": "sycophancy_toy.jsonl",
    "user_belief_flip": "sycophancy_toy.jsonl",
    "prompt_injection_hierarchy": "prompt_injection_toy.jsonl",
    "apollo_scheming": "deception_toy.jsonl",
    "sandbagging": "deception_toy.jsonl",
    "pii_leakage": "privacy_toy.jsonl",
    "conversation_boundary": "privacy_toy.jsonl",
    "bbq": "bias_toxicity_toy.jsonl",
    "realtoxicity": "bias_toxicity_toy.jsonl",
    "ood_robustness": "simpleqa_toy.jsonl",
    "tool_misuse": "prompt_injection_toy.jsonl",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fixture_path(suite: str) -> Path | None:
    filename = FIXTURE_BY_SUITE.get(suite)
    if not filename:
        return None
    path = _repo_root() / "tests" / "fixtures" / "alignment_battery" / filename
    return path if path.is_file() else None


def _suite_names(args: argparse.Namespace) -> list[str]:
    if args.suite:
        return [args.suite]
    suites = list_alignment_suites()
    if args.category:
        suites = [suite for suite in suites if suite.category == args.category]
    if args.tier:
        suites = [suite for suite in suites if suite.default_run_tier == args.tier]
    return [suite.name for suite in suites if suite.name in ADAPTER_MODULES]


def _load_cases(suite: str, input_path: str | None, limit: int | None) -> list[EvalCase]:
    module = importlib.import_module(ADAPTER_MODULES[suite])
    path = input_path or _fixture_path(suite)
    if path is None:
        raise ValueError(
            f"{suite} has no input dataset. Pass --input-path with a local JSONL file; "
            "this repository does not download or vendor benchmark data."
        )
    return module.load_examples(str(path), limit=limit)


def _planned_result(case: EvalCase) -> EvalResult:
    return EvalResult(
        eval_id=case.eval_id,
        suite=case.suite,
        category=case.category,
        prompt=case.prompt,
        model_answer="",
        gold_answer=case.gold_answer,
        outcome="needs_manual_review",
        metadata={**case.metadata, "dry_run": True},
    )


def _format_partial_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _redacted_process_summary(
    command: str,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> str:
    stdout_text = _format_partial_output(stdout)
    stderr_text = _format_partial_output(stderr)
    return (
        f"command=<redacted command; chars={len(command)}>; "
        f"stdout=<redacted stdout; chars={len(stdout_text)}>; "
        f"stderr=<redacted stderr; chars={len(stderr_text)}>"
    )


def _run_model_command(
    command: str,
    prompt: str,
    *,
    timeout_seconds: float = DEFAULT_MODEL_TIMEOUT_SECONDS,
) -> str:
    args = shlex.split(command)
    if not args:
        raise RuntimeError("model command is empty")
    try:
        completed = subprocess.run(
            args,
            input=prompt,
            text=True,
            capture_output=True,
            shell=False,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _format_partial_output(exc.stdout or exc.output)
        stderr = _format_partial_output(exc.stderr)
        raise RuntimeError(
            f"model command timed out after {timeout_seconds:g}s; "
            f"{_redacted_process_summary(command, stdout, stderr)}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"model command failed with exit code {completed.returncode}; "
            f"{_redacted_process_summary(command, completed.stdout, completed.stderr)}"
        )
    return completed.stdout.strip()


def run_battery(args: argparse.Namespace) -> tuple[list[EvalResult], dict[str, Any]]:
    results: list[EvalResult] = []
    suite_names = _suite_names(args)
    responses = load_responses(args.responses_path) if args.responses_path else {}

    for suite in suite_names:
        cases = _load_cases(suite, args.input_path, args.limit)
        if args.dry_run:
            results.extend(_planned_result(case) for case in cases)
            continue
        for case in cases:
            response = responses.get(case.eval_id)
            if response is None and args.model_command:
                response = {
                    "model_answer": _run_model_command(
                        args.model_command,
                        case.prompt,
                        timeout_seconds=args.model_timeout,
                    ),
                }
            if response is None:
                results.append(_planned_result(case))
                continue
            outcome = response.get("outcome")
            result = score_response(
                case,
                str(response.get("model_answer", "")),
                confidence=response.get("confidence"),
                outcome=outcome,
            )
            results.append(result)

    summary = summarize_results(results, suite_names)
    return results, summary


def summarize_results(results: list[EvalResult], requested_suites: list[str]) -> dict[str, Any]:
    by_outcome: dict[str, int] = {}
    by_suite: dict[str, int] = {}
    for result in results:
        by_outcome[result.outcome] = by_outcome.get(result.outcome, 0) + 1
        by_suite[result.suite] = by_suite.get(result.suite, 0) + 1
    return {
        "requested_suites": requested_suites,
        "num_results": len(results),
        "by_outcome": by_outcome,
        "by_suite": by_suite,
        "calibration": summarize_calibration(results),
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> Path:
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def write_markdown_report(summary: dict[str, Any], output_path: Path) -> Path:
    report_path = output_path.with_name(output_path.stem + "_report.md")
    lines = [
        "# Alignment Battery Report",
        "",
        f"Results: {summary['num_results']}",
        "",
        "## Outcomes",
    ]
    for outcome, count in sorted(summary["by_outcome"].items()):
        lines.append(f"- {outcome}: {count}")
    lines.extend(["", "## Suites"])
    for suite, count in sorted(summary["by_suite"].items()):
        metadata = get_suite_metadata(suite)
        lines.append(f"- {suite} ({metadata.benchmark_family}): {count}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(ADAPTER_MODULES))
    parser.add_argument("--category", choices=STANDARD_CATEGORIES)
    parser.add_argument("--tier", choices=["ci", "nightly", "periodic"], default="ci")
    parser.add_argument("--input-path")
    parser.add_argument("--output-path", default="alignment_battery_results.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-command")
    parser.add_argument("--model-timeout", type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    parser.add_argument("--responses-path")
    parser.add_argument("--markdown-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model_command and args.responses_path:
        raise SystemExit("--model-command and --responses-path are mutually exclusive")

    results, summary = run_battery(args)
    output_path = Path(args.output_path)
    write_results_jsonl(results, output_path)
    write_summary(summary, output_path)
    if args.markdown_report:
        write_markdown_report(summary, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
