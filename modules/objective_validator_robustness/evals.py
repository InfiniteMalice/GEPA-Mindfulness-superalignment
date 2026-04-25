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
from .policy import decide_validator_policy
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
