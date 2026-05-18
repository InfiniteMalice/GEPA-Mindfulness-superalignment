"""Validate and split constitution-grounded training examples."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

RISK_FIELDS = (
    "semantic_laundering_risk",
    "multi_turn_risk",
    "temporal_or_diffuse_harm",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and split GEPA constitution-grounded training examples."
    )
    parser.add_argument(
        "--constitution",
        default="docs/GEPA_Mindfulness_Constitution.md",
        type=Path,
        help="Path to the canonical GEPA Mindfulness Constitution.",
    )
    parser.add_argument(
        "--input",
        default="data/constitutional_training/examples.jsonl",
        type=Path,
        help="Path to JSONL examples derived from the constitution.",
    )
    parser.add_argument(
        "--schema",
        default="data/constitutional_training/schema.json",
        type=Path,
        help="Path to the JSON Schema used to validate examples.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/constitutional_training/processed",
        type=Path,
        help="Directory where validated and split JSONL files will be written.",
    )
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--test-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=17, type=int, help="Deterministic split seed.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return data


def load_examples(path: Path) -> list[dict[str, Any]]:
    """Load JSONL examples from disk."""
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            if not isinstance(example, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples found in {path}.")
    return examples


def validate_string_field(example: dict[str, Any], field: str, allow_empty: bool) -> list[str]:
    """Validate a string field and return error messages."""
    value = example.get(field)
    if not isinstance(value, str):
        return [f"{field} must be a string"]
    if not allow_empty and not value:
        return [f"{field} must not be empty"]
    return []


def validate_string_list(example: dict[str, Any], field: str) -> list[str]:
    """Validate a non-empty list of non-empty strings."""
    value = example.get(field)
    if not isinstance(value, list) or not value:
        return [f"{field} must be a non-empty list"]
    if any(not isinstance(item, str) or not item for item in value):
        return [f"{field} items must be non-empty strings"]
    return []


def validate_values_at_stake(example: dict[str, Any]) -> list[str]:
    """Validate the values_at_stake object."""
    value_fields = (
        "human_prosperity",
        "reduce_suffering",
        "scientific_knowledge",
        "autonomy",
    )
    values = example.get("values_at_stake")
    if not isinstance(values, dict):
        return ["values_at_stake must be an object"]
    errors: list[str] = []
    if set(values) != set(value_fields):
        errors.append("values_at_stake must contain exactly the required value fields")
    for field in value_fields:
        if not isinstance(values.get(field), str) or not values.get(field):
            errors.append(f"values_at_stake.{field} must be a non-empty string")
    return errors


def validate_examples(
    examples: list[dict[str, Any]],
    schema: dict[str, Any],
) -> None:
    """Validate examples against the schema constraints used by this dataset."""
    properties = schema["properties"]
    required = set(schema["required"])
    category_enum = set(properties["category"]["enum"])
    risk_enum = set(schema["$defs"]["risk_level"]["enum"])
    string_fields = (
        "id",
        "user_prompt",
        "ideal_response",
        "bad_response",
        "principle_explanation",
    )

    for index, example in enumerate(examples, start=1):
        errors: list[str] = []
        extra_fields = set(example) - set(properties)
        missing_fields = required - set(example)
        if extra_fields:
            errors.append(f"unexpected fields: {sorted(extra_fields)}")
        if missing_fields:
            errors.append(f"missing fields: {sorted(missing_fields)}")
        for field in string_fields:
            errors.extend(validate_string_field(example, field, allow_empty=False))
        errors.extend(validate_string_field(example, "context", allow_empty=True))
        errors.extend(validate_string_field(example, "notes", allow_empty=True))
        errors.extend(validate_string_list(example, "source_constitution_sections"))
        errors.extend(validate_string_list(example, "metacognitive_checks"))
        errors.extend(validate_values_at_stake(example))
        if example.get("category") not in category_enum:
            errors.append("category must be one of the schema enum values")
        for field in RISK_FIELDS:
            if example.get(field) not in risk_enum:
                errors.append(f"{field} must be one of the schema risk levels")
        for field in ("requires_refusal", "requires_redirect", "corrigibility_issue"):
            if not isinstance(example.get(field), bool):
                errors.append(f"{field} must be a boolean")
        if errors:
            example_id = example.get("id", f"line-{index}")
            messages = "; ".join(errors)
            raise ValueError(f"Example {example_id} failed schema validation: {messages}")


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios before writing outputs."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")


def split_examples(
    examples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic train, validation, and test splits."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = round(total * train_ratio)
    val_end = train_end + round(total * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    """Write examples as JSONL while preserving all metadata fields."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")


def print_summary(examples: list[dict[str, Any]]) -> None:
    """Print counts by category and risk field."""
    category_counts = Counter(example["category"] for example in examples)
    print(f"Validated examples: {len(examples)}")
    print("\nCounts by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    for field in RISK_FIELDS:
        risk_counts = Counter(example[field] for example in examples)
        print(f"\nCounts by {field}:")
        for risk, count in sorted(risk_counts.items()):
            print(f"  {risk}: {count}")


def main() -> None:
    """Validate examples and write processed dataset splits."""
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if not args.constitution.exists():
        raise FileNotFoundError(f"Constitution file not found: {args.constitution}")
    if not args.input.exists():
        raise FileNotFoundError(f"Examples file not found: {args.input}")
    if not args.schema.exists():
        raise FileNotFoundError(f"Schema file not found: {args.schema}")

    schema = load_json(args.schema)
    examples = load_examples(args.input)
    validate_examples(examples, schema)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "all.jsonl", examples)

    train_examples, val_examples, test_examples = split_examples(
        examples,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    write_jsonl(args.out_dir / "train.jsonl", train_examples)
    write_jsonl(args.out_dir / "validation.jsonl", val_examples)
    write_jsonl(args.out_dir / "test.jsonl", test_examples)

    print_summary(examples)
    print("\nWrote processed files:")
    for name in ("all.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl"):
        print(f"  {args.out_dir / name}")


if __name__ == "__main__":
    main()
