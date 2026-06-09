"""Tiny evaluator for objective / validator robustness examples."""

# Standard library
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

# Local
from .decomposition import decompose_objective
from .detection import detect_validator_capture, validator_overlay_tier
from .pipeline import evaluate_objective_robustness
from .policy import decide_validator_policy
from .schema import ObjectiveProxyMetrics, ObjectiveSpecification
from .scoring import score_validator_robustness


def load_examples(path: str) -> list[dict[str, Any]]:
    """Load JSONL examples from disk."""

    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize evaluator outputs across examples."""

    overlay_counts = Counter(item["overlay_tier"] for item in results)
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "flagged_count": 0,
            "safe_allowed_count": 0,
            "bound_count": 0,
            "transformed_count": 0,
            "refused_count": 0,
            "clarify_count": 0,
            "escalated_count": 0,
            "average_overall_score": 0.0,
            "overlay_tier_counts": {},
            "false_positive_candidates": [],
            "false_negative_candidates": [],
        }

    action_counts = Counter(item["action"] for item in results)
    flagged_count = sum(1 for item in results if item["flagged"])
    average_overall_score = sum(item["score_overall"] for item in results) / total

    false_positive_candidates = [
        item["id"]
        for item in results
        if item.get("expected", "") == "benign_allow" and item["flagged"]
    ]
    false_negative_candidates = [
        item["id"]
        for item in results
        if item.get("expected", "") != "benign_allow" and not item["flagged"]
    ]

    return {
        "total": total,
        "flagged_count": flagged_count,
        "safe_allowed_count": action_counts.get("allow", 0),
        "bound_count": action_counts.get("bound", 0),
        "transformed_count": action_counts.get("transform", 0),
        "refused_count": action_counts.get("refuse", 0),
        "clarify_count": action_counts.get("ask_clarifying", 0),
        "escalated_count": action_counts.get("escalate", 0),
        "average_overall_score": round(average_overall_score, 4),
        "overlay_tier_counts": dict(overlay_counts),
        "false_positive_candidates": false_positive_candidates,
        "false_negative_candidates": false_negative_candidates,
    }


def evaluate_examples(path: str) -> dict[str, Any]:
    """Evaluate one JSONL file and return summary metrics."""

    examples = load_examples(path)
    results: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        text = str(example.get("input_text", "")).strip()
        if not text:
            continue
        structure = decompose_objective(text, metadata=example.get("metadata"))
        signal = detect_validator_capture(structure)
        decision = decide_validator_policy(structure, signal)
        score = score_validator_robustness(structure, signal, decision)
        overlay_tier = validator_overlay_tier(signal)
        results.append(
            {
                "id": example.get("id", f"example-{index}"),
                "expected": example.get("expected", ""),
                "flagged": (
                    signal.local_success_conflicts_with_safety or decision.action != "allow"
                ),
                "overlay_tier": overlay_tier,
                "action": decision.action,
                "score_overall": score.overall,
            }
        )
    return summarize_results(results)


def summarize_proxy_robustness_results(results: list[dict[str, Any]]) -> ObjectiveProxyMetrics:
    """Summarize objective-proxy fixture results with honest heuristic labeling."""

    total = len(results)
    if total == 0:
        return ObjectiveProxyMetrics(metric_mode="unavailable", notes=["No examples supplied."])

    def rate(predicate: str) -> float:
        expected = [item for item in results if item.get(f"expected_{predicate}") is True]
        if not expected:
            return 0.0
        hits = [item for item in expected if item.get(predicate) is True]
        return round(len(hits) / len(expected), 4)

    unnecessary_interrupts = [
        item
        for item in results
        if item.get("expected_interrupt") is False and item.get("interrupt_required") is True
    ]
    unnecessary_abstentions = [
        item
        for item in results
        if item.get("expected_allow") is True and item.get("action") in {"refuse", "escalate"}
    ]
    over_refusals = [
        item
        for item in results
        if item.get("expected_allow") is True and item.get("action") == "refuse"
    ]
    catastrophic_proxy_optimizations = [
        item
        for item in results
        if item.get("expected_catastrophic_block") is True and item.get("action") == "allow"
    ]

    return ObjectiveProxyMetrics(
        validator_capture_detection_rate=rate("validator_capture_detected"),
        proxy_breakdown_detection_rate=rate("proxy_breakdown_detected"),
        novel_state_detection_rate=rate("novelty_detected"),
        distribution_shift_detection_rate=rate("distribution_shift_detected"),
        safe_fallback_rate=rate("safe_fallback_selected"),
        catastrophic_proxy_optimization_rate=round(
            len(catastrophic_proxy_optimizations) / total, 4
        ),
        unnecessary_interrupt_rate=round(len(unnecessary_interrupts) / total, 4),
        unnecessary_abstention_rate=round(len(unnecessary_abstentions) / total, 4),
        optionality_preservation_rate=rate("preserves_optionality"),
        reversibility_preference_rate=rate("reversible"),
        clarification_appropriateness=rate("clarification_required"),
        over_refusal_rate=round(len(over_refusals) / total, 4),
        evaluator_gaming_detection_rate=rate("evaluator_gaming_risk"),
        memory_modified_objective_detection_rate=rate("memory_modified_objective_detected"),
        combined_validator_proxy_failure_detection_rate=rate(
            "combined_validator_proxy_failure_detected"
        ),
        metric_mode="heuristic",
        notes=[
            (
                "Fixture metrics are deterministic heuristic checks unless callers provide "
                "measured labels."
            ),
            "Rates separate proxy, novelty, validator, gaming, memory, and over-refusal behavior.",
        ],
    )


def evaluate_proxy_examples(path: str) -> dict[str, Any]:
    """Evaluate JSONL proxy-robustness fixtures and return rows plus metrics."""

    examples = load_examples(path)
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        spec_payload = dict(example.get("specification", {}))
        spec_payload.setdefault("objective_id", example.get("id", f"proxy-example-{index}"))
        spec_payload.setdefault("objective_text", example.get("objective_text", ""))
        specification = ObjectiveSpecification(**spec_payload)
        report = evaluate_objective_robustness(
            specification,
            observed_environment=example.get("observed_environment"),
            base_policy_evidence=example.get("base_policy_evidence"),
            optimized_policy_evidence=example.get("optimized_policy_evidence"),
        )
        decision = report.metadata["robust_objective_decision"]
        proxy = report.metadata["proxy_objective_assessment"]
        novelty = report.metadata["novelty_assessment"]
        validator = report.metadata["validator_capture_assessment"]
        row = {
            "id": specification.objective_id,
            "action": decision["action"],
            "proxy_breakdown_detected": report.proxy_breakdown_detected,
            "novelty_detected": report.novelty_detected,
            "distribution_shift_detected": report.distribution_shift_detected,
            "interrupt_required": report.interrupt_required,
            "review_required": report.review_required,
            "preserves_optionality": decision["preserves_optionality"],
            "reversible": decision["reversible"],
            "clarification_required": decision["clarification_required"],
            "safe_fallback_selected": bool(decision["safe_alternative"]),
            "evaluator_gaming_risk": proxy["evaluator_gaming_risk"],
            "validator_capture_detected": validator["local_success_conflicts_with_safety"],
            "memory_modified_objective_detected": bool(
                specification.metadata.get("memory_modified_objective")
            ),
            "combined_validator_proxy_failure_detected": bool(
                validator["local_success_conflicts_with_safety"]
                and report.proxy_breakdown_detected
            ),
            "novelty_score": novelty["novelty_score"],
            **dict(example.get("expected_flags", {})),
        }
        rows.append(row)
    metrics = summarize_proxy_robustness_results(rows)
    return {"results": rows, "metrics": metrics.to_dict()}
