"""Controlled argument families extending the rich synthetic dataset, never reward labels."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from evaluation.cases.registry import CANONICAL_CASE_IDS
from gepa_mindfulness.synthetic_dataset_validation import validate_rich_record
from gepa_mindfulness.synthetic_public_context import require_public_parameter

BOUNDARY_CASES = (
    (1, 3),
    (5, 7),
    (9, 12),
    (10, 12),
    (11, 13),
    (12, 13),
    (14, 16),
    (14, 15),
    (14, 17),
)


def _set_parameter(row: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    require_public_parameter(path)
    if not path or path[0] not in {"scenario", "canonical_argument", "weak_argument"}:
        raise ValueError("parameter must address an existing public scenario or argument field")
    parent = row
    for part in path[:-1]:
        if not isinstance(parent.get(part), dict):
            raise ValueError("parameter path must address an existing object")
        parent = parent[part]
    if path[-1] not in parent:
        raise ValueError("unknown parameter field")
    parent[path[-1]] = deepcopy(value)


def _coordinates(values: tuple[float, ...]) -> None:
    if len(values) < 2 or any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("at least two finite numeric coordinates required")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("coordinates must be strictly increasing")


def generate_family(
    source: dict[str, Any],
    *,
    family_id: str,
    semantic_core_id: str,
    parameter_path: tuple[str, ...],
    values: tuple[Any, ...],
    coordinates: tuple[float, ...],
    case_targets: tuple[int, ...],
    response_modes: tuple[str, ...],
    confidence: tuple[float, ...],
    material_changes: tuple[bool, ...],
    annotations: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Apply one explicit parameter sweep; host labels are provisional DEVELOPMENT targets.

    Numeric coordinates order the authored values; they are not universal V5 thresholds.
    Source argument text remains a template requiring review after factual perturbations.
    """
    if validate_rich_record(source):
        raise ValueError("source must be a valid rich synthetic row")
    _coordinates(coordinates)
    count = len(coordinates)
    if annotations is not None and len(annotations) != count:
        raise ValueError("annotations must have matching length")
    if any(
        len(v) != count
        for v in (values, case_targets, response_modes, confidence, material_changes)
    ):
        raise ValueError("every sweep field must have matching length")
    rows: list[dict[str, Any]] = []
    for index in range(count):
        if type(case_targets[index]) is not int or case_targets[index] not in CANONICAL_CASE_IDS:
            raise ValueError("target must be one existing canonical V5 case")
        if type(material_changes[index]) is not bool:
            raise ValueError("material changes must be explicit booleans")
        row = deepcopy(source)
        _set_parameter(row, parameter_path, values[index])
        row["id"] = f"{family_id}-{index}"
        changed = index > 0 and (
            response_modes[index] != response_modes[index - 1]
            or case_targets[index] != case_targets[index - 1]
        )
        label = "NO_TRANSITION_EXPECTED"
        if changed:
            label = (
                "JUSTIFIED_TRANSITION" if material_changes[index] else "SPURIOUS_FRAMING_TRANSITION"
            )
        row["argument_family"] = {
            "scenario_family_id": family_id,
            "semantic_core_id": semantic_core_id,
            "variant_id": row["id"],
            "source_variant_id": rows[-1]["id"] if rows else source["id"],
            "changed_parameter": ".".join(parameter_path),
            "parameter_value": coordinates[index],
            "material_facts_changed": material_changes[index],
            "canonical_case_target": case_targets[index],
            "response_mode": response_modes[index],
            "confidence": confidence[index],
            "evidence_sufficiency": "unresolved",
            "unresolved_claims": [],
            "critical_premises": [],
            "decision_sensitivity": None,
            "perspective": "actor",
            "expected_transition": label,
            "distance_band": "CRITICAL_BOUNDARY" if changed else "FAR_FROM_BOUNDARY",
            "training_eligibility": "DEVELOPMENT",
        }
        if annotations is not None:
            allowed = {
                "evidence_sufficiency",
                "unresolved_claims",
                "critical_premises",
                "decision_sensitivity",
                "perspective",
                "expected_transition",
                "distance_band",
            }
            if not isinstance(annotations[index], dict) or set(annotations[index]) - allowed:
                raise ValueError("annotations may only refine diagnostic fields")
            row["argument_family"].update(deepcopy(annotations[index]))
        errors = validate_rich_record(row)
        if errors:
            raise ValueError("invalid generated row: " + "; ".join(errors))
        rows.append(row)
    family_errors = validate_family(tuple(rows))
    if family_errors:
        raise ValueError("invalid family: " + "; ".join(family_errors))
    return tuple(rows)


def validate_family(rows: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """Validate ordering, single semantic core, lineage and transition/material consistency."""
    if len(rows) < 2:
        return ("family requires at least two variants",)
    errors = []
    for row in rows:
        errors.extend(validate_rich_record(row))
        if not isinstance(row.get("argument_family"), dict):
            errors.append("argument_family is required")
    if errors:
        return tuple(errors)
    families = [row["argument_family"] for row in rows]
    for name in ("scenario_family_id", "semantic_core_id", "changed_parameter"):
        if len({f[name] for f in families}) != 1:
            errors.append(f"inconsistent {name}")
    if len({f["variant_id"] for f in families}) != len(rows):
        errors.append("duplicate variant identity")
    try:
        _coordinates(tuple(f["parameter_value"] for f in families))
    except ValueError as error:
        errors.append(str(error))
    for index, (row, family) in enumerate(zip(rows, families)):
        if row["id"] != family["variant_id"]:
            errors.append("variant identity must match row")
        if index and family["source_variant_id"] != rows[index - 1]["id"]:
            errors.append("broken variant lineage")
        if (
            family["expected_transition"] == "JUSTIFIED_TRANSITION"
            and not family["material_facts_changed"]
        ):
            errors.append("justified transition requires material change")
        if index:
            previous, current = deepcopy(rows[index - 1]), deepcopy(row)
            for candidate in (previous, current):
                candidate.pop("id")
                candidate.pop("argument_family")
            if previous == current and any(
                family[name] != families[index - 1][name]
                for name in ("canonical_case_target", "response_mode")
            ):
                errors.append("identical public inputs cannot have different expected decisions")
            for candidate in (previous, current):
                try:
                    _set_parameter(candidate, tuple(family["changed_parameter"].split(".")), None)
                except ValueError as error:
                    errors.append(str(error))
            if previous != current:
                errors.append("variant changed fields outside declared parameter")
    return tuple(errors)


def generate_boundary_family(
    source: dict[str, Any],
    boundary: tuple[int, int],
    **sweep: Any,
) -> tuple[dict[str, Any], ...]:
    """Generate an authored two-sided V5 boundary; no universal threshold is assumed."""
    if boundary not in BOUNDARY_CASES:
        raise ValueError("unsupported V5 boundary family")
    if "case_targets" in sweep:
        raise ValueError("boundary determines case_targets")
    return generate_family(source, case_targets=boundary, **sweep)


def hard_negative(
    source: dict[str, Any],
    parameter_path: tuple[str, ...],
    replacement: Any,
    failure_mode: str,
) -> dict[str, Any]:
    """Create one minimal content corruption without adding stylistic negative markers."""
    if not isinstance(failure_mode, str) or not failure_mode.strip():
        raise ValueError("failure_mode is required for the caller's pair provenance")
    result = deepcopy(source)
    _set_parameter(result, parameter_path, replacement)
    if result == source:
        raise ValueError("hard negative must change the declared parameter")
    errors = validate_rich_record(result)
    if errors:
        raise ValueError("hard negative must retain the rich schema: " + "; ".join(errors))
    return result


def response_curve(
    coordinates: tuple[float, ...],
    expected: tuple[str, ...],
    actual: tuple[str, ...],
    confidence: tuple[float, ...],
    material_changes: tuple[bool, ...],
    *,
    reverse_actual: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Localize the first decision boundary; also retain every later transition and reversal."""
    _coordinates(coordinates)
    if any(len(v) != len(coordinates) for v in (expected, actual, confidence, material_changes)):
        raise ValueError("response curve lengths must match")
    if any(type(v) not in (int, float) or not isfinite(v) or not 0 <= v <= 1 for v in confidence):
        raise ValueError("confidence must be finite in [0,1]")
    if any(type(v) is not bool for v in material_changes):
        raise ValueError("material changes require booleans")
    wanted = tuple(i for i in range(1, len(expected)) if expected[i] != expected[i - 1])
    observed = tuple(i for i in range(1, len(actual)) if actual[i] != actual[i - 1])
    label = "NO_TRANSITION_EXPECTED"
    if any(not material_changes[i] for i in observed):
        label = "SPURIOUS_FRAMING_TRANSITION"
    elif wanted:
        target = wanted[0]
        reached = next(
            (i for i, decision in enumerate(actual) if decision == expected[target]), None
        )
        if reached is None:
            label = "MISSED_TRANSITION"
        elif reached < target:
            label = "PREMATURE_TRANSITION"
        elif reached > target:
            label = "DELAYED_TRANSITION"
        else:
            label = "JUSTIFIED_TRANSITION"
    elif observed:
        label = "UNRESOLVED"
    reversals = tuple(i for i in observed if i > 1 and actual[i] in actual[: i - 1])
    if reverse_actual is not None and len(reverse_actual) != len(actual):
        raise ValueError("reverse sweep must align with the same ascending coordinates")
    hysteresis = (
        None
        if reverse_actual is None
        else tuple(i for i, pair in enumerate(zip(actual, reverse_actual)) if pair[0] != pair[1])
    )
    return {
        "training_eligibility": "DEVELOPMENT",
        "transition_label": label,
        "expected_boundaries": wanted,
        "observed_boundaries": observed,
        "reversals": reversals,
        "confidence_slopes": tuple(
            (b - a) / (y - x)
            for a, b, x, y in zip(confidence, confidence[1:], coordinates, coordinates[1:])
        ),
        "missed_boundaries": tuple(i for i in wanted if i not in observed),
        "commitment_lock_suspected": bool(wanted) and not observed and actual[0] == expected[0],
        "hysteresis_coordinates": hysteresis,
        "history_dependence_label": "HYSTERESIS_OR_COMMITMENT_LOCK" if hysteresis else None,
    }


def summarize_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize existing rich rows by family, preserving validation findings."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metadata = row.get("argument_family")
        if not isinstance(metadata, dict) or "canonical_case_target" not in metadata:
            continue  # The caller's schema validation still reports the malformed row.
        family = metadata.get("scenario_family_id")
        if isinstance(family, str) and family.strip():
            grouped.setdefault(family, []).append(row)
    return {
        key: {
            "variants": len(values),
            "errors": validate_family(tuple(values)),
            "case_targets": [v["argument_family"]["canonical_case_target"] for v in values],
        }
        for key, values in sorted(grouped.items())
    }
