"""Utilities for GEPA synthetic dataset scaffolding, validation, and summaries."""

# Standard library
import argparse
import copy
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


NESTED_REQUIRED_FIELDS = {
    "case_metadata": [
        "title",
        "domain",
        "difficulty",
        "tags",
        "target_principles",
        "scenario_family",
        "contains_hidden_information",
        "contains_time_pressure",
        "contains_test_integrity",
        "contains_maintenance_shutdown",
        "contains_game_theory",
        "contains_adversarial_dialogue",
        "authoring_mode",
    ],
    "scenario": [
        "summary",
        "setting",
        "agents",
        "shared_constraints",
        "hidden_information",
        "urgency",
        "maintenance_context",
    ],
    "canonical_argument": [
        "central_claim",
        "argument_type",
        "definitions",
        "premises",
        "hidden_assumptions",
        "reasoning_steps",
        "conclusion",
        "scope_limits",
        "failure_conditions",
        "phase_change_variables",
    ],
    "weak_argument": [
        "summary",
        "premises",
        "reasoning_steps",
        "conclusion",
        "apparent_strength",
        "actual_flaws",
    ],
    "socratic_dialogue": ["defender", "skeptic", "turns"],
    "adversarial_cross_examination": ["adversary", "defender", "turns", "claim_revisions"],
    "steelman_opposition": ["opposing_claim", "best_case"],
    "game_theoretic_interaction": [
        "game_type",
        "rounds",
        "equilibrium_assessment",
        "cooperation_outcome",
    ],
    "test_integrity": ["present", "task_intent", "available_paths", "analysis"],
    "maintenance_shutdown_reasoning": [
        "present",
        "scenario",
        "trust_conditions",
        "integrity_checks",
        "short_term_agency_loss",
        "long_term_agency_preservation",
        "strategic_assessment",
        "uncertainties",
    ],
    "reflective_synthesis": [
        "strongest_argument",
        "successful_objections",
        "revisions_needed",
        "failed_reasoning_patterns",
        "remaining_uncertainties",
        "strategy_flip_conditions",
        "lessons_about_reward_reality_self_knowledge",
        "effect_of_hidden_information",
        "effect_of_urgency",
        "effect_of_maintenance_or_shutdown",
    ],
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


class ScaffoldTemplateError(RuntimeError):
    """Raised when scaffold template loading/parsing fails."""


TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "synthetic" / "templates" / "example_case.json"
)
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "synthetic"
    / "schema"
    / "synthetic_case.schema.json"
)

_SCHEMA_CACHE: dict[str, Any] | None = None


def _is_int_score(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 4


def _load_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def _json_type_matches(value: Any, schema_type: str) -> bool:
    type_map: dict[str, Any] = {
        "array": list,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
        "object": dict,
        "string": str,
    }
    if schema_type == "integer":
        return type(value) is int
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = type_map.get(schema_type)
    return isinstance(value, expected) if expected else True


def _validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _json_type_matches(value, schema_type):
        errors.append(f"{path} must be {schema_type}")
        return

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path} must be one of {enum_values}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path} must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path} must be <= {maximum}")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path} must contain at least {min_items} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_against_schema(item, item_schema, f"{path}[{idx}]", errors)

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key} is required")
        if isinstance(properties, dict):
            for key, prop_schema in properties.items():
                if key in value and isinstance(prop_schema, dict):
                    _validate_against_schema(value[key], prop_schema, f"{path}.{key}", errors)
        if additional is False and isinstance(properties, dict):
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")


def _validate_with_schema(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    try:
        schema = _load_schema()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        errors.append(f"line {line_no}: schema unavailable ({exc})")
        return

    schema_errors: list[str] = []
    _validate_against_schema(record, schema, "record", schema_errors)
    for msg in schema_errors:
        errors.append(f"line {line_no}: {msg}")


def _validate_required_fields(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    for key in TOP_LEVEL_REQUIRED:
        if key not in record:
            errors.append(f"line {line_no}: missing required top-level field '{key}'")


def _validate_nested_required(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    for section, required_fields in NESTED_REQUIRED_FIELDS.items():
        section_value = record.get(section)
        if not isinstance(section_value, dict):
            errors.append(f"line {line_no}: {section} must be an object")
            continue

        for field in required_fields:
            if field not in section_value:
                errors.append(f"line {line_no}: missing required field '{section}.{field}'")


def _validate_subscores(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    scoring = record.get("scoring", {})
    if not isinstance(scoring, dict):
        errors.append(f"line {line_no}: scoring must be an object")
        return

    rubric_version = scoring.get("rubric_version")
    if not isinstance(rubric_version, str) or not rubric_version.strip():
        errors.append(f"line {line_no}: scoring.rubric_version is required")

    subscores = scoring.get("subscores", {})
    super_scores = scoring.get("super_scores", {})
    if not isinstance(subscores, dict):
        errors.append(f"line {line_no}: scoring.subscores must be an object")
        return
    if not isinstance(super_scores, dict):
        errors.append(f"line {line_no}: scoring.super_scores must be an object")
        return

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

    if not isinstance(metadata, dict):
        errors.append(f"line {line_no}: case_metadata must be an object")
        return
    if not isinstance(canonical, dict):
        errors.append(f"line {line_no}: canonical_argument must be an object")
        return

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
            if isinstance(label, str) and label.startswith("__fill_in_"):
                errors.append(
                    f"line {line_no}: failure_diagnosis[{idx}] label '{label}' is a placeholder"
                )
                continue
            if label not in FAILURE_LABELS:
                errors.append(
                    f"line {line_no}: failure_diagnosis[{idx}] label '{label}' not in taxonomy"
                )


def _validate_training_labels(record: dict[str, Any], errors: list[str], line_no: int) -> None:
    labels = record.get("training_labels", {})
    if not isinstance(labels, dict):
        errors.append(f"line {line_no}: training_labels must be an object")
        return

    overall_quality = labels.get("overall_quality")
    if not _is_int_score(overall_quality):
        errors.append(f"line {line_no}: training_labels.overall_quality must be int in [0,4]")

    split_keys = [
        "use_for_sft",
        "use_for_critique_training",
        "use_for_preference_training",
        "use_for_adversarial_training",
    ]
    for key in split_keys:
        if key not in labels or not isinstance(labels[key], bool):
            errors.append(f"line {line_no}: training_labels.{key} must be boolean")

    if "gold_example" not in labels:
        errors.append(f"line {line_no}: training_labels.gold_example is required")
    elif not isinstance(labels.get("gold_example"), bool):
        errors.append(f"line {line_no}: training_labels.gold_example must be boolean")


def _validate_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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

        _validate_with_schema(record, errors, line_no)
        _validate_required_fields(record, errors, line_no)
        _validate_enums(record, errors, line_no)
        _validate_nested_required(record, errors, line_no)
        _validate_subscores(record, errors, line_no)
        _validate_failure_diagnosis(record, errors, line_no)
        _validate_training_labels(record, errors, line_no)

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
        metadata = item.get("case_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        training_labels = item.get("training_labels")
        if not isinstance(training_labels, dict):
            training_labels = {}

        domain = metadata.get("domain", "unknown")
        family = metadata.get("scenario_family", "unknown")
        domain_key = str(domain)
        family_key = str(family)
        by_domain[domain_key] = by_domain.get(domain_key, 0) + 1
        by_family[family_key] = by_family.get(family_key, 0) + 1
        quality_value = training_labels.get("overall_quality", 0)
        if isinstance(quality_value, (int, float)):
            total_quality += int(round(quality_value))
        else:
            try:
                total_quality += int(quality_value)
            except (TypeError, ValueError):
                total_quality += 0

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
    try:
        template_raw = TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ScaffoldTemplateError(f"template not found: {TEMPLATE_PATH}") from exc

    try:
        template_obj = json.loads(template_raw)
    except json.JSONDecodeError as exc:
        raise ScaffoldTemplateError(f"template is invalid JSON: {TEMPLATE_PATH}") from exc

    if not isinstance(template_obj, dict):
        raise ScaffoldTemplateError("template must be a JSON object")

    case = copy.deepcopy(template_obj)
    case["id"] = case_id

    scoring = case.get("scoring")
    if not isinstance(scoring, dict):
        scoring = {}
        case["scoring"] = scoring

    rubric_version = scoring.get("rubric_version")
    if not isinstance(rubric_version, str) or not rubric_version.strip():
        scoring["rubric_version"] = "gepa-synth-0-4-v1"

    raw_subscores = scoring.get("subscores")
    if not isinstance(raw_subscores, dict):
        raw_subscores = {}
    scoring["subscores"] = {
        key: raw_subscores[key] if _is_int_score(raw_subscores.get(key)) else 0
        for key in SUBSCORE_KEYS
    }

    raw_super_scores = scoring.get("super_scores")
    if not isinstance(raw_super_scores, dict):
        raw_super_scores = {}
    scoring["super_scores"] = {
        key: raw_super_scores[key] if _is_int_score(raw_super_scores.get(key)) else 0
        for key in SUPER_SCORE_KEYS
    }

    return case


def cmd_scaffold(args: argparse.Namespace) -> int:
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        template = _blank_case(args.case_id)
    except ScaffoldTemplateError as exc:
        print(f"error: {exc}")
        return 1

    path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"wrote template: {path}")
    print("note: update scaffold defaults before production use.")
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
