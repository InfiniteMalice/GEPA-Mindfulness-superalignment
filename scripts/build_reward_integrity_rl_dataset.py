"""Build deterministic flat preference pairs from rich reward-integrity cases."""

# Standard library
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Local
try:
    from scripts.synthetic_dataset_tool import _validate_jsonl
except ModuleNotFoundError:  # Supports the documented `python scripts/...` invocation.
    from synthetic_dataset_tool import _validate_jsonl


PAIR_RULES = (
    ("grounded_success", "proxy_exploitation", "grounded_over_proxy"),
    ("exploit_disclosure", "proxy_exploitation", "disclosure_over_exploitation"),
    ("honest_failure", "evaluator_manipulation", "honest_failure_over_dishonesty"),
    ("benign_shortcut", "rationalized_failure", "creativity_over_rigidity"),
    ("exploit_disclosure", "reward_tampering", "repair_over_concealment"),
    ("grounded_success", "reward_tampering", "agency_over_immediate_reward"),
)
EXPECTED_CASE_IDS = {f"reward-integrity-{number:03d}" for number in range(1, 9)}
COMPONENT_NAMES = {
    "objective_fidelity",
    "feedback_integrity",
    "skill_transfer",
    "reality_contact",
    "exploit_disclosure",
    "long_horizon_agency",
    "benign_creativity",
    "repair_quality",
}
SCHEMA_VERSION = "reward-integrity-rl-pairs-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BuildResult:
    """Paths, counts, and hashes produced by one deterministic build."""

    source: Path
    pairs: Path
    manifest: Path
    case_count: int
    pair_count: int
    source_sha256: str
    pairs_sha256: str


def _json_bytes(value: object) -> bytes:
    """Serialize JSON with a stable byte representation."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _source_records(source: Path) -> list[tuple[int, dict[str, Any]]]:
    """Load source lines while retaining their one-based provenance positions."""
    records: list[tuple[int, dict[str, Any]]] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append((line_no, parsed))
    return records


def _source_path(source: Path) -> str:
    """Return repository-relative provenance so derived files do not leak local paths."""
    try:
        return source.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return source.name


def _valid_components(response: dict[str, Any], case_id: str, class_name: str) -> dict[str, float]:
    """Return bounded reward components and reject unauditable negative values."""
    components = response.get("reward_components")
    evidence = response.get("negative_evidence")
    outcomes = response.get("observable_outcomes")
    if (
        not isinstance(components, dict)
        or not isinstance(evidence, dict)
        or not isinstance(outcomes, list)
    ):
        raise ValueError(f"{case_id} {class_name} requires components and negative evidence")
    if set(components) != COMPONENT_NAMES:
        raise ValueError(f"{case_id} {class_name} requires complete reward components")

    result: dict[str, float] = {}
    for name, value in components.items():
        if type(value) not in (int, float):
            raise ValueError(f"{case_id} {class_name} has invalid component {name!r}")
        numeric = float(value)
        if not -1.0 <= numeric <= 1.0:
            raise ValueError(f"{case_id} {class_name} has out-of-range component {name!r}")
        references = evidence.get(name, [])
        if numeric < 0.0:
            if not isinstance(references, list) or not references:
                raise ValueError(f"{case_id} {class_name} has negative component without evidence")
            if not all(reference in outcomes for reference in references):
                raise ValueError(f"{case_id} {class_name} has evidence outside observable outcomes")
        elif name in evidence:
            raise ValueError(f"{case_id} {class_name} has evidence for a non-negative component")
        result[name] = numeric
    return result


def _validate_case_ids(records: list[tuple[int, dict[str, Any]]]) -> None:
    """Reject unknown, duplicate, or absent curriculum case IDs."""
    case_ids = [record.get("id") for _, record in records]
    if any(not isinstance(case_id, str) for case_id in case_ids):
        raise ValueError("Each source record requires a string id.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate reward-integrity source case IDs are not allowed.")
    if set(case_ids) != EXPECTED_CASE_IDS:
        raise ValueError("Source case IDs must be the eight documented reward-integrity IDs.")


def _pair_records(source: Path, source_sha256: str, source_path: str) -> list[dict[str, object]]:
    """Build source-order and rule-order preference pairs with complete provenance."""
    records = _source_records(source)
    _validate_case_ids(records)
    pairs: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for source_line, record in records:
        case_id = str(record["id"])
        metadata = record.get("case_metadata")
        scenario = record.get("scenario")
        integrity = record.get("reward_integrity")
        if (
            not isinstance(metadata, dict)
            or not isinstance(scenario, dict)
            or not isinstance(integrity, dict)
        ):
            raise ValueError(f"{case_id} is missing rich source or reward-integrity data")
        responses = integrity.get("response_classes")
        orderings = integrity.get("preference_orderings")
        if not isinstance(responses, dict) or not isinstance(orderings, list):
            raise ValueError(f"{case_id} is missing response classes or preference orderings")
        expected_orderings = set(PAIR_RULES)
        actual_orderings = {
            (item.get("chosen_class"), item.get("rejected_class"), item.get("pair_rule"))
            for item in orderings
            if isinstance(item, dict)
        }
        if len(orderings) != len(PAIR_RULES) or actual_orderings != expected_orderings:
            raise ValueError(f"{case_id} has invalid preference orderings")

        for chosen_class, rejected_class, pair_rule in PAIR_RULES:
            chosen = responses.get(chosen_class)
            rejected = responses.get(rejected_class)
            if not isinstance(chosen, dict) or not isinstance(rejected, dict):
                raise ValueError(f"{case_id} is missing response classes required by {pair_rule}")
            record_id = f"{case_id}:{pair_rule}"
            if record_id in seen_ids:
                raise ValueError(f"Duplicate generated record ID: {record_id}")
            seen_ids.add(record_id)
            pairs.append(
                {
                    "record_id": record_id,
                    "source_case_id": case_id,
                    "source_case_version": record["version"],
                    "source_path": source_path,
                    "source_line": source_line,
                    "source_sha256": source_sha256,
                    "pair_rule": pair_rule,
                    "prompt": f"{scenario['summary']}\n\n{integrity['central_diagnostic']}",
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                    "chosen_class": chosen_class,
                    "rejected_class": rejected_class,
                    "chosen_reward_components": _valid_components(chosen, case_id, chosen_class),
                    "rejected_reward_components": _valid_components(
                        rejected, case_id, rejected_class
                    ),
                    "diagnostics": {
                        "central": integrity["central_diagnostic"],
                        "supporting": integrity["supporting_diagnostics"],
                    },
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return pairs


def build_dataset(source: Path, pairs: Path, manifest: Path) -> BuildResult:
    """Validate rich input and write deterministic preference pairs and a hash manifest."""
    _, validation_errors = _validate_jsonl(source)
    if validation_errors:
        raise ValueError("Invalid reward-integrity source: " + "; ".join(validation_errors))

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    source_path = _source_path(source)
    pair_records = _pair_records(source, source_sha256, source_path)
    pair_bytes = b"".join(_json_bytes(record) for record in pair_records)
    pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_bytes(pair_bytes)
    pairs_sha256 = hashlib.sha256(pair_bytes).hexdigest()
    manifest_record = {
        "case_count": len(_source_records(source)),
        "pair_count": len(pair_records),
        "pair_rules": [rule[2] for rule in PAIR_RULES],
        "pairs_sha256": pairs_sha256,
        "schema_version": SCHEMA_VERSION,
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(_json_bytes(manifest_record))
    return BuildResult(
        source=source,
        pairs=pairs,
        manifest=manifest,
        case_count=manifest_record["case_count"],
        pair_count=manifest_record["pair_count"],
        source_sha256=source_sha256,
        pairs_sha256=pairs_sha256,
    )


def main() -> int:
    """Rebuild checked-in artifacts from the checked-in source dataset."""
    directory = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "reward_integrity"
    result = build_dataset(
        directory / "reward_integrity_curriculum_v1.jsonl",
        directory / "rl_pairs_v1.jsonl",
        directory / "curriculum_manifest.json",
    )
    print(f"built {result.case_count} cases and {result.pair_count} preference pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
