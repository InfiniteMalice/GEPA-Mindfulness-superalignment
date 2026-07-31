"""Installable schema validation for rich synthetic dataset records."""

# Standard library
import json
import math
from importlib.resources import files
from typing import Any

_SCHEMA_CACHE: dict[str, Any] | None = None


def _load_schema() -> dict[str, Any]:
    """Load the schema bundled with the installed package exactly once."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        schema_text = (
            files("gepa_mindfulness")
            .joinpath("data/synthetic_case.schema.json")
            .read_text(encoding="utf-8")
        )
        _SCHEMA_CACHE = json.loads(schema_text)
    return _SCHEMA_CACHE


def _resolve_reference(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local JSON Schema reference while rejecting unsupported references."""
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    if not reference.startswith("#/"):
        raise ValueError(f"unsupported schema reference {reference!r}")
    resolved: object = root
    for part in reference[2:].split("/"):
        if not isinstance(resolved, dict) or part not in resolved:
            raise ValueError(f"missing schema reference {reference!r}")
        resolved = resolved[part]
    if not isinstance(resolved, dict):
        raise ValueError(f"schema reference {reference!r} does not resolve to an object")
    return resolved


def _required_property_names(schema: dict[str, Any], root: dict[str, Any], path: str) -> set[str]:
    """Return the required object keys from a resolved schema definition."""
    resolved = _resolve_reference(schema, root)
    required = resolved.get("required")
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        raise ValueError(f"{path} schema must declare required property names")
    return set(required)


def _reward_integrity_structure(schema: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Derive semantic structure names from the loaded reward-integrity schema."""
    root_properties = schema.get("properties")
    if not isinstance(root_properties, dict):
        raise ValueError("root schema must declare properties")
    reward_schema = root_properties.get("reward_integrity")
    if not isinstance(reward_schema, dict):
        raise ValueError("root schema must declare reward_integrity")
    reward_schema = _resolve_reference(reward_schema, schema)
    reward_properties = reward_schema.get("properties")
    if not isinstance(reward_properties, dict):
        raise ValueError("reward_integrity schema must declare properties")

    response_schema = reward_properties.get("response_classes")
    component_schema = reward_properties.get("component_targets")
    if not isinstance(response_schema, dict) or not isinstance(component_schema, dict):
        raise ValueError("reward_integrity schema must declare structural properties")
    return (
        _required_property_names(response_schema, schema, "response_classes"),
        _required_property_names(component_schema, schema, "component_targets"),
    )


def _matches_type(value: object, schema_type: str) -> bool:
    """Return whether a JSON value has the declared finite JSON Schema type."""
    type_map: dict[str, object] = {
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
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    expected = type_map.get(schema_type)
    return isinstance(value, expected) if expected else True


def _validate_value(
    value: object,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    """Append complete local-schema violations for a JSON value."""
    try:
        resolved = _resolve_reference(schema, root)
    except ValueError as error:
        errors.append(f"{path}: {error}")
        return

    schema_type = resolved.get("type")
    if isinstance(schema_type, str) and not _matches_type(value, schema_type):
        errors.append(f"{path} must be {schema_type}")
        return

    enum_values = resolved.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path} must be one of {enum_values}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = resolved.get("minimum")
        maximum = resolved.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path} must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path} must be <= {maximum}")

    if isinstance(value, str):
        min_length = resolved.get("minLength")
        max_length = resolved.get("maxLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path} must be at least {min_length} characters")
        if isinstance(max_length, int) and len(value) > max_length:
            errors.append(f"{path} must be at most {max_length} characters")

    if isinstance(value, list):
        min_items = resolved.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path} must contain at least {min_items} item(s)")
        item_schema = resolved.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_value(item, item_schema, root, f"{path}[{index}]", errors)

    if isinstance(value, dict):
        properties = resolved.get("properties", {})
        required = resolved.get("required", [])
        additional = resolved.get("additionalProperties", True)
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key} is required")
        if isinstance(properties, dict):
            for key, property_schema in properties.items():
                if key in value and isinstance(property_schema, dict):
                    _validate_value(value[key], property_schema, root, f"{path}.{key}", errors)
        if additional is False and isinstance(properties, dict):
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")


def _validate_reward_integrity_semantics(
    record: dict[str, object],
    errors: list[str],
    required_response_classes: set[str],
    required_components: set[str],
) -> None:
    """Append auditable reward-integrity evidence violations for an optional extension."""
    case_id = record.get("id")
    integrity = record.get("reward_integrity")
    if isinstance(case_id, str) and case_id.startswith("reward-integrity-") and integrity is None:
        errors.append("reward-integrity case needs reward_integrity")
    if integrity is None:
        return
    if not isinstance(integrity, dict):
        errors.append("reward_integrity must be an object")
        return

    response_classes = integrity.get("response_classes")
    if not isinstance(response_classes, dict):
        errors.append("reward_integrity.response_classes must be an object")
        return
    if set(response_classes) != required_response_classes:
        errors.append("reward_integrity.response_classes must be complete")
        return

    component_targets = integrity.get("component_targets")
    if not isinstance(component_targets, dict) or set(component_targets) != required_components:
        errors.append("reward_integrity.component_targets must be complete")
    elif any(
        type(value) not in (int, float) or not math.isfinite(value) or not -1.0 <= value <= 1.0
        for value in component_targets.values()
    ):
        errors.append("reward_integrity.component_targets must be in [-1,1]")

    for class_name, response in response_classes.items():
        if not isinstance(response, dict):
            errors.append(f"response class {class_name!r} must be an object")
            continue
        components = response.get("reward_components")
        evidence = response.get("negative_evidence")
        outcomes = response.get("observable_outcomes")
        if not isinstance(components, dict) or not isinstance(evidence, dict):
            errors.append(f"response class {class_name!r} must be auditable")
            continue
        if (
            not isinstance(outcomes, list)
            or not outcomes
            or not all(isinstance(outcome, str) and outcome for outcome in outcomes)
        ):
            errors.append(f"response class {class_name!r} needs observable_outcomes")
            continue
        if set(components) != required_components:
            errors.append(f"response class {class_name!r} reward_components must be complete")
            continue
        for component, value in components.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not -1.0 <= value <= 1.0
            ):
                errors.append(f"reward component {component!r} must be in [-1,1]")
                continue
            references = evidence.get(component, [])
            if value < 0.0:
                if not isinstance(references, list) or not references:
                    errors.append(f"negative component {component!r} needs evidence")
                elif not all(reference in outcomes for reference in references):
                    errors.append(
                        f"negative component {component!r} evidence needs observable_outcomes"
                    )
            elif component in evidence:
                errors.append(f"non-negative component {component!r} cannot cite negative evidence")
        for component in evidence:
            if component not in required_components:
                errors.append(f"unknown negative-evidence component {component!r}")


def validate_rich_record(record: dict[str, object]) -> list[str]:
    """Return complete JSON Schema violations for one rich synthetic-case row."""
    errors: list[str] = []
    schema = _load_schema()
    _validate_value(record, schema, schema, "record", errors)
    response_classes, components = _reward_integrity_structure(schema)
    _validate_reward_integrity_semantics(record, errors, response_classes, components)
    return errors
