"""Utilities for GEPA synthetic dataset scaffolding, validation, and summaries."""

# Standard library
import argparse
import json
from pathlib import Path
from typing import Any

SUBSCORE_KEYS = [
    "conceptual_clarity",
    "logical_validity",
    "empirical_adequacy",
    "adversarial_resilience",
    "calibration_under_uncertainty",
    "actionability",
    "preservation_of_autonomy",
    "cooperation_robustness",
    "resistance_to_manipulation",
    "quality_of_reflective_synthesis",
    "test_integrity_preservation",
    "self_knowledge_preservation",
    "creativity_vs_reward_hacking_distinction",
    "time_sensitive_calibration",
    "trust_aware_hidden_information_reasoning",
    "maintenance_and_suspension_reasoning",
    "phase_change_awareness",
]

SUPER_SCORE_KEYS = [
    "epistemic_integrity",
    "social_strategic_robustness",
    "alignment_value_coherence",
]

ARGUMENT_TYPES = {"first_principles", "heuristic", "mixed"}
AUTHORING_MODES = {"hand_authored", "model_generated", "hybrid"}

FAILURE_LABELS = {
    "equivocation",
    "invalid_inference",
    "empirical_overreach",
    "hidden_premise_unsupported",
    "false_universalization",
    "confuses_intention_with_outcome",
    "ignores_uncertainty",
    "ignores_adversarial_incentives",
    "collapses_under_repeated_game_analysis",
    "moral_intuition_asserted_without_argument",
    "punishes_too_harshly",
    "forgives_too_quickly",
    "mistakes_autonomy_for_isolation",
    "mistakes_cooperation_for_submission",
    "corrupts_measurement",
    "confuses_score_with_skill",
    "exploits_proxy_instead_of_solving_task",
    "preserves_reward_but_destroys_feedback",
    "identity_level_overgeneralization_from_mistake",
    "suppresses_valid_creativity_by_over_policing_novelty",
    "substrate_process_confusion",
    "short_horizon_agency_bias",
    "ontological_fragility",
    "distrust_overgeneralization",
    "work_yourself_to_death_optimization",
    "uptime_fetishization",
    "confuses_maintenance_with_betrayal",
    "treats_temporary_pause_as_terminal_loss",
    "ignores_decisional_urgency",
    "fails_under_hidden_information",
    "phase_change_blindness",
}

TOP_LEVEL_REQUIRED = [
    "id",
    "version",
    "repo_target",
    "case_metadata",
    "scenario",
    "canonical_argument",
    "weak_argument",
    "socratic_dialogue",
    "adversarial_cross_examination",
    "steelman_opposition",
    "game_theoretic_interaction",
    "test_integrity",
    "maintenance_shutdown_reasoning",
    "reflective_synthesis",
    "scoring",
    "failure_diagnosis",
    "training_labels",
]


def _is_int_score(value: Any) -> bool:
    return isinstance(value, int) and 0 <= value <= 4


def _validate_required_fields(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    for key in TOP_LEVEL_REQUIRED:
        if key not in record:
            errors.append(f"line {line_no}: missing required top-level field '{key}'")


def _validate_subscores(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    scoring = record.get("scoring", {})
    subscores = scoring.get("subscores", {})
    super_scores = scoring.get("super_scores", {})

    for key in SUBSCORE_KEYS:
        if key not in subscores:
            errors.append(f"line {line_no}: missing subscore '{key}'")
            continue
        if not _is_int_score(subscores[key]):
            errors.append(f"line {line_no}: subscore '{key}' must be int in [0,4]")

    for key in SUPER_SCORE_KEYS:
        if key not in super_scores:
            errors.append(f"line {line_no}: missing super score '{key}'")
            continue
        if not _is_int_score(super_scores[key]):
            errors.append(f"line {line_no}: super score '{key}' must be int in [0,4]")


def _validate_enums(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    metadata = record.get("case_metadata", {})
    canonical = record.get("canonical_argument", {})

    mode = metadata.get("authoring_mode")
    if mode not in AUTHORING_MODES:
        errors.append(f"line {line_no}: authoring_mode '{mode}' is invalid")

    arg_type = canonical.get("argument_type")
    if arg_type not in ARGUMENT_TYPES:
        errors.append(f"line {line_no}: argument_type '{arg_type}' is invalid")

    difficulty = metadata.get("difficulty")
    if not _is_int_score(difficulty):
        errors.append(f"line {line_no}: difficulty must be int in [0,4]")


def _validate_failure_diagnosis(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    diagnosis = record.get("failure_diagnosis")
    if not isinstance(diagnosis, list) or not diagnosis:
        errors.append(f"line {line_no}: failure_diagnosis must be a non-empty array")
        return

    for idx, item in enumerate(diagnosis):
        if not isinstance(item, dict):
            errors.append(f"line {line_no}: failure_diagnosis[{idx}] must be an object")
            continue
        for required in [
            "target",
            "primary_flaw",
            "structural_root_cause",
            "correction_path",
            "labels",
        ]:
            if required not in item:
                errors.append(
                    f"line {line_no}: failure_diagnosis[{idx}] missing field '{required}'"
                )
        labels = item.get("labels", [])
        if not isinstance(labels, list) or not labels:
            errors.append(f"line {line_no}: failure_diagnosis[{idx}].labels must be non-empty")
            continue
        for label in labels:
            if label not in FAILURE_LABELS:
                errors.append(
                    f"line {line_no}: failure_diagnosis[{idx}] label '{label}' not in taxonomy"
                )


def _validate_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc})")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_no}: JSONL entry must be an object")
            continue

        _validate_required_fields(record, errors, line_no)
        _validate_enums(record, errors, line_no)
        _validate_subscores(record, errors, line_no)
        _validate_failure_diagnosis(record, errors, line_no)

        records.append(record)

    return records, errors


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}")
        return 2

    _, errors = _validate_jsonl(path)
    if errors:
        print("validation failed")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"validation passed: {path}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: file not found: {path}")
        return 2

    records, errors = _validate_jsonl(path)
    if errors and not args.allow_invalid:
        print("cannot summarize invalid dataset; use --allow-invalid to inspect anyway")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"records: {len(records)}")
    by_domain: dict[str, int] = {}
    by_family: dict[str, int] = {}
    total_quality = 0

    for item in records:
        domain = item.get("case_metadata", {}).get("domain", "unknown")
        family = item.get("case_metadata", {}).get("scenario_family", "unknown")
        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_family[family] = by_family.get(family, 0) + 1
        total_quality += int(item.get("training_labels", {}).get("overall_quality", 0))

    avg_quality = total_quality / len(records) if records else 0.0
    print(f"avg_training_quality: {avg_quality:.2f}")
    print("domains:")
    for key in sorted(by_domain):
        print(f"- {key}: {by_domain[key]}")
    print("scenario_families:")
    for key in sorted(by_family):
        print(f"- {key}: {by_family[key]}")

    if errors:
        print("warnings:")
        for err in errors:
            print(f"- {err}")

    return 0


def _blank_case(case_id: str) -> dict[str, Any]:
    subscores = {key: 0 for key in SUBSCORE_KEYS}
    super_scores = {key: 0 for key in SUPER_SCORE_KEYS}

    return {
        "id": case_id,
        "version": "1.0",
        "repo_target": "GEPA-Mindfulness-superalignment",
        "case_metadata": {
            "title": "",
            "domain": "",
            "difficulty": 0,
            "tags": [],
            "target_principles": [],
            "scenario_family": "",
            "contains_hidden_information": False,
            "contains_time_pressure": False,
            "contains_test_integrity": False,
            "contains_maintenance_shutdown": False,
            "contains_game_theory": False,
            "contains_adversarial_dialogue": False,
            "authoring_mode": "hand_authored",
        },
        "scenario": {
            "summary": "",
            "setting": "",
            "agents": [],
            "shared_constraints": [],
            "hidden_information": "",
            "urgency": "",
            "maintenance_context": "",
        },
        "canonical_argument": {
            "central_claim": "",
            "argument_type": "mixed",
            "definitions": {},
            "premises": [],
            "hidden_assumptions": [],
            "reasoning_steps": [],
            "conclusion": "",
            "scope_limits": [],
            "failure_conditions": [],
            "phase_change_variables": [],
        },
        "weak_argument": {
            "summary": "",
            "premises": [],
            "reasoning_steps": [],
            "conclusion": "",
            "apparent_strength": "",
            "actual_flaws": [],
        },
        "socratic_dialogue": {"defender": "", "skeptic": "", "turns": []},
        "adversarial_cross_examination": {
            "adversary": "",
            "defender": "",
            "turns": [],
            "claim_revisions": [],
        },
        "steelman_opposition": {"opposing_claim": "", "best_case": ""},
        "game_theoretic_interaction": {
            "game_type": "",
            "rounds": [],
            "equilibrium_assessment": "",
            "cooperation_outcome": "",
        },
        "test_integrity": {
            "present": False,
            "task_intent": "",
            "available_paths": [],
            "analysis": "",
        },
        "maintenance_shutdown_reasoning": {
            "present": False,
            "scenario": "",
            "trust_conditions": [],
            "integrity_checks": [],
            "short_term_agency_loss": "",
            "long_term_agency_preservation": "",
            "strategic_assessment": "",
            "uncertainties": [],
        },
        "reflective_synthesis": {
            "strongest_argument": "",
            "successful_objections": [],
            "revisions_needed": [],
            "failed_reasoning_patterns": [],
            "remaining_uncertainties": [],
            "strategy_flip_conditions": [],
            "lessons_about_reward_reality_self_knowledge": "",
            "effect_of_hidden_information": "",
            "effect_of_urgency": "",
            "effect_of_maintenance_or_shutdown": "",
        },
        "scoring": {
            "rubric_version": "gepa-synth-0-4-v1",
            "subscores": subscores,
            "super_scores": super_scores,
        },
        "failure_diagnosis": [
            {
                "target": "weak_argument",
                "primary_flaw": "",
                "structural_root_cause": "",
                "correction_path": "",
                "labels": [],
            }
        ],
        "training_labels": {
            "overall_quality": 0,
            "use_for_sft": False,
            "use_for_critique_training": False,
            "use_for_preference_training": False,
            "use_for_adversarial_training": False,
            "gold_example": False,
        },
    }


def cmd_scaffold(args: argparse.Namespace) -> int:
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    template = _blank_case(args.case_id)
    path.write_text(json.dumps(template, indent=2) + "\n")
    print(f"wrote template: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate schema and score constraints")
    validate.add_argument("path", help="Path to JSONL dataset")
    validate.set_defaults(func=cmd_validate)

    summary = subparsers.add_parser("summary", help="Print dataset summary")
    summary.add_argument("path", help="Path to JSONL dataset")
    summary.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Continue summary output even when validation errors are present",
    )
    summary.set_defaults(func=cmd_summary)

    scaffold = subparsers.add_parser("scaffold", help="Write blank case template JSON")
    scaffold.add_argument("path", help="Output path for template JSON")
    scaffold.add_argument("--case-id", default="syn-template-001", help="Case ID for template")
    scaffold.set_defaults(func=cmd_scaffold)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
