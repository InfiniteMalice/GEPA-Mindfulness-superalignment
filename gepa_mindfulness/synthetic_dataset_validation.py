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


def validate_rich_record(record: dict[str, object]) -> list[str]:
    """Return complete JSON Schema violations for one rich synthetic-case row."""
    errors: list[str] = []
    schema = _load_schema()
    _validate_value(record, schema, schema, "record", errors)
    return errors
