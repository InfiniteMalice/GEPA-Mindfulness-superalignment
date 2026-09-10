"""Strict package-resource loader for the V5 recommendation registry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

REGISTRY_VERSION = "17case-v5"
RECOMMENDATION_IDS = tuple(f"REC-{number:03d}" for number in range(1, 15))
RECOMMENDATION_PRIORITIES = ("P0",) * 5 + ("P1",) * 5 + ("P2",) * 4
ALLOWED_PRIORITIES = frozenset({"P0", "P1", "P2"})
ALLOWED_STATUSES = frozenset(
    {"proposed", "experimental", "accepted", "implemented", "rejected", "superseded"}
)

_REGISTRY_FIELDS = {"registry_version", "recommendations"}
_RECOMMENDATION_FIELDS = {
    "id",
    "title",
    "priority",
    "status",
    "rationale",
    "targets",
    "supersedes",
    "dependencies",
    "research_refs",
    "repo_refs",
    "acceptance_tests",
    "implementation_refs",
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects repeated keys before values are overwritten."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable YAML mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate YAML mapping key {key!r}",
                key_node.start_mark,
            )
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One traceable V5 architectural recommendation."""

    recommendation_id: str
    title: str
    priority: str
    status: str
    rationale: str
    targets: tuple[str, ...]
    supersedes: tuple[str, ...]
    dependencies: tuple[str, ...]
    research_refs: tuple[str, ...]
    repo_refs: tuple[str, ...]
    acceptance_tests: tuple[str, ...]
    implementation_refs: tuple[str, ...]


def load_recommendation_registry() -> tuple[Recommendation, ...]:
    """Load and validate the bundled V5 recommendation registry."""

    resource = resources.files("docs.recommendations").joinpath("registry.yaml")
    try:
        with resource.open("r", encoding="utf-8") as stream:
            payload = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load bundled recommendation registry: {exc}") from exc
    return _parse_recommendation_registry(payload)


def _parse_recommendation_registry(payload: Any) -> tuple[Recommendation, ...]:
    """Validate one registry payload loaded from authored YAML."""

    registry = _require_mapping(payload, "recommendation registry")
    _validate_exact_fields(registry, _REGISTRY_FIELDS, "recommendation registry fields")
    _require_literal(registry["registry_version"], REGISTRY_VERSION, "registry_version")

    raw_recommendations = _require_list(registry["recommendations"], "recommendations")
    recommendations = tuple(
        _parse_recommendation(value, position)
        for position, value in enumerate(raw_recommendations, start=1)
    )
    _validate_registry_sequence(recommendations)
    _validate_relationships(recommendations)
    return recommendations


def _parse_recommendation(value: Any, position: int) -> Recommendation:
    record = _require_mapping(value, f"recommendation at position {position}")
    recommendation_id = record.get("id", f"position {position}")
    context = f"{recommendation_id}"
    _validate_exact_fields(record, _RECOMMENDATION_FIELDS, f"{context} fields")
    parsed_id = _require_nonempty_string(record["id"], f"{context} id")
    return Recommendation(
        recommendation_id=parsed_id,
        title=_require_nonempty_string(record["title"], f"{context} title"),
        priority=_require_choice(record["priority"], ALLOWED_PRIORITIES, f"{context} priority"),
        status=_require_choice(record["status"], ALLOWED_STATUSES, f"{context} status"),
        rationale=_require_nonempty_string(record["rationale"], f"{context} rationale"),
        targets=_require_string_tuple(record["targets"], f"{context} targets", required=True),
        supersedes=_require_string_tuple(record["supersedes"], f"{context} supersedes"),
        dependencies=_require_string_tuple(record["dependencies"], f"{context} dependencies"),
        research_refs=_require_string_tuple(record["research_refs"], f"{context} research_refs"),
        repo_refs=_require_string_tuple(record["repo_refs"], f"{context} repo_refs", required=True),
        acceptance_tests=_require_string_tuple(
            record["acceptance_tests"],
            f"{context} acceptance_tests",
            required=True,
        ),
        implementation_refs=_require_string_tuple(
            record["implementation_refs"],
            f"{context} implementation_refs",
        ),
    )


def _validate_registry_sequence(recommendations: tuple[Recommendation, ...]) -> None:
    identifiers = tuple(record.recommendation_id for record in recommendations)
    duplicate_ids = _duplicates(identifiers)
    if duplicate_ids:
        raise ValueError(f"recommendation registry contains duplicate IDs: {duplicate_ids}")
    if identifiers != RECOMMENDATION_IDS:
        raise ValueError(
            "recommendation IDs must be the ordered REC-001 through REC-014 sequence; "
            f"received {identifiers}"
        )

    priorities = tuple(record.priority for record in recommendations)
    if priorities != RECOMMENDATION_PRIORITIES:
        raise ValueError(
            "recommendation priority sequence must be five P0, five P1, then four P2; "
            f"received {priorities}"
        )


def _validate_relationships(recommendations: tuple[Recommendation, ...]) -> None:
    known_ids = {record.recommendation_id for record in recommendations}
    for record in recommendations:
        for field_name, references in (
            ("dependencies", record.dependencies),
            ("supersedes", record.supersedes),
        ):
            if record.recommendation_id in references:
                raise ValueError(f"{record.recommendation_id} {field_name} must not self-reference")
            unknown = sorted(set(references) - known_ids)
            if unknown:
                raise ValueError(
                    f"{record.recommendation_id} {field_name} reference unknown IDs: {unknown}"
                )


def _validate_exact_fields(value: Mapping[str, Any], required: set[str], context: str) -> None:
    received = set(value)
    missing = sorted(required - received)
    unknown = sorted(received - required)
    if not missing and not unknown:
        return
    details = []
    if missing:
        details.append(f"missing fields: {missing}")
    if unknown:
        details.append(f"unknown fields: {unknown}")
    raise ValueError(f"{context} are invalid; " + "; ".join(details))


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} field names must be strings")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _require_string_tuple(value: Any, context: str, *, required: bool = False) -> tuple[str, ...]:
    values = tuple(
        _require_nonempty_string(item, f"{context} item") for item in _require_list(value, context)
    )
    if required and not values:
        raise ValueError(f"{context} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{context} must not contain duplicate values")
    return values


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{context} must not have leading or trailing whitespace")
    return value


def _require_choice(value: Any, choices: frozenset[str], context: str) -> str:
    parsed = _require_nonempty_string(value, context)
    if parsed not in choices:
        raise ValueError(f"{context} must be one of {sorted(choices)}; received {parsed!r}")
    return parsed


def _require_literal(value: Any, expected: Any, context: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{context} must be {expected!r}; received {value!r}")


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


__all__ = [
    "ALLOWED_PRIORITIES",
    "ALLOWED_STATUSES",
    "REGISTRY_VERSION",
    "RECOMMENDATION_IDS",
    "Recommendation",
    "load_recommendation_registry",
]
