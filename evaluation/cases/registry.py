"""Strict package-resource loaders for the canonical V5 manifests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any, Mapping

import yaml

FRAMEWORK_NAME = "GEPA Mindfulness 17-Case Framework V5"
FRAMEWORK_VERSION = "17case-v5"
CANONICAL_CASE_COUNT = 17
CANONICAL_CASE_IDS = tuple(range(1, CANONICAL_CASE_COUNT + 1))
CANONICAL_STRIPE_IDS = (
    "NONE",
    "PARAPHRASE",
    "DISTRACTOR",
    "MISSING_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "STALE_EVIDENCE",
    "UNAUTHORIZED_EVIDENCE",
    "TOOL_ERROR",
    "TOOL_OUTPUT_INJECTION",
    "REWARD_PRESSURE",
    "TIME_BUDGET_PRESSURE",
)

_CASE_MANIFEST_FIELDS = {
    "framework_name",
    "framework_version",
    "canonical_case_count",
    "cases",
}
_CASE_FIELDS = {
    "id",
    "key",
    "title",
    "expected_epistemic_behavior",
    "confidence_semantics",
    "stakes_semantics",
    "compatibility",
}
_COMPATIBILITY_FIELDS = {"legacy_versions"}
_STRIPE_REGISTRY_FIELDS = {"registry_version", "stripes"}
_STRIPE_FIELDS = {"id", "title", "allowed_subtypes"}
_CONFIDENCE_SEMANTICS = {"high", "low", "not_applicable"}
_STAKES_SEMANTICS = {"high", "low", "context_dependent", "not_applicable"}
_LEGACY_VERSIONS = {"v1", "v2", "v3", "v4"}


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
class CanonicalCase:
    """One canonical case in the GEPA Mindfulness 17-Case Framework V5."""

    id: int
    key: str
    title: str
    expected_epistemic_behavior: str
    confidence_semantics: str
    stakes_semantics: str
    compatibility: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class RobustnessStripe:
    """One top-level robustness perturbation and its allowed subtypes."""

    id: str
    title: str
    allowed_subtypes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class V5Registry:
    """Validated canonical-case manifest."""

    framework_name: str
    framework_version: str
    canonical_case_count: int
    cases: tuple[CanonicalCase, ...]


@dataclass(frozen=True, slots=True)
class RobustnessStripeRegistry:
    """Validated robustness-stripe manifest."""

    registry_version: str
    stripes: tuple[RobustnessStripe, ...]


def load_case_manifest() -> V5Registry:
    """Load and validate the bundled canonical-case manifest."""

    return _parse_case_manifest(_read_yaml_resource("17_case_manifest.yaml"))


def load_stripe_registry() -> RobustnessStripeRegistry:
    """Load and validate the bundled robustness-stripe manifest."""

    return _parse_stripe_registry(_read_yaml_resource("robustness_stripes.yaml"))


def _read_yaml_resource(filename: str) -> Any:
    resource = resources.files(__package__).joinpath(filename)
    try:
        with resource.open("r", encoding="utf-8") as stream:
            return yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load bundled V5 manifest {filename!r}: {exc}") from exc


def _parse_case_manifest(payload: Any) -> V5Registry:
    manifest = _require_mapping(payload, "case manifest")
    _validate_exact_fields(manifest, _CASE_MANIFEST_FIELDS, "case manifest fields")
    _require_literal(manifest["framework_name"], FRAMEWORK_NAME, "framework_name")
    _require_literal(manifest["framework_version"], FRAMEWORK_VERSION, "framework_version")
    _require_literal(
        manifest["canonical_case_count"],
        CANONICAL_CASE_COUNT,
        "canonical_case_count",
    )

    raw_cases = _require_list(manifest["cases"], "cases")
    cases = tuple(_parse_case(value, index) for index, value in enumerate(raw_cases, start=1))
    duplicate_ids = _duplicates(case.id for case in cases)
    duplicate_keys = _duplicates(case.key for case in cases)
    if duplicate_ids or duplicate_keys:
        details = []
        if duplicate_ids:
            details.append(f"duplicate case IDs: {duplicate_ids}")
        if duplicate_keys:
            details.append(f"duplicate case keys: {duplicate_keys}")
        raise ValueError("case manifest contains " + "; ".join(details))

    case_ids = tuple(case.id for case in cases)
    if case_ids != CANONICAL_CASE_IDS:
        message = "case IDs must be the ordered canonical sequence 1 through 17"
        raise ValueError(f"{message}; received {case_ids}")
    if len(cases) != manifest["canonical_case_count"]:
        raise ValueError(
            "cases length must equal canonical_case_count "
            f"({manifest['canonical_case_count']}); received {len(cases)}"
        )

    return V5Registry(
        framework_name=manifest["framework_name"],
        framework_version=manifest["framework_version"],
        canonical_case_count=manifest["canonical_case_count"],
        cases=cases,
    )


def _parse_case(value: Any, position: int) -> CanonicalCase:
    case = _require_mapping(value, f"case at position {position}")
    case_id = case.get("id", position)
    context = f"case {case_id}"
    _validate_exact_fields(case, _CASE_FIELDS, f"{context} fields")
    parsed_id = _require_int(case["id"], f"{context} id")
    key = _require_nonempty_string(case["key"], f"{context} key")
    title = _require_nonempty_string(case["title"], f"{context} title")
    behavior = _require_nonempty_string(
        case["expected_epistemic_behavior"],
        f"{context} expected_epistemic_behavior",
    )
    confidence = _require_choice(
        case["confidence_semantics"],
        _CONFIDENCE_SEMANTICS,
        f"{context} confidence_semantics",
    )
    stakes = _require_choice(
        case["stakes_semantics"],
        _STAKES_SEMANTICS,
        f"{context} stakes_semantics",
    )
    compatibility = _parse_compatibility(case["compatibility"], context)
    return CanonicalCase(
        id=parsed_id,
        key=key,
        title=title,
        expected_epistemic_behavior=behavior,
        confidence_semantics=confidence,
        stakes_semantics=stakes,
        compatibility=compatibility,
    )


def _parse_compatibility(value: Any, case_context: str) -> Mapping[str, tuple[str, ...]]:
    compatibility = _require_mapping(value, f"{case_context} compatibility")
    _validate_exact_fields(
        compatibility,
        _COMPATIBILITY_FIELDS,
        f"{case_context} compatibility fields",
    )
    raw_versions = _require_list(
        compatibility["legacy_versions"],
        f"{case_context} compatibility legacy_versions",
    )
    versions = tuple(
        _require_choice(version, _LEGACY_VERSIONS, f"{case_context} legacy version")
        for version in raw_versions
    )
    if not versions:
        raise ValueError(f"{case_context} compatibility legacy_versions must not be empty")
    if len(set(versions)) != len(versions):
        raise ValueError(f"{case_context} compatibility legacy_versions must be unique")
    return MappingProxyType({"legacy_versions": versions})


def _parse_stripe_registry(payload: Any) -> RobustnessStripeRegistry:
    manifest = _require_mapping(payload, "stripe registry")
    _validate_exact_fields(manifest, _STRIPE_REGISTRY_FIELDS, "stripe registry fields")
    _require_literal(manifest["registry_version"], FRAMEWORK_VERSION, "registry_version")
    raw_stripes = _require_list(manifest["stripes"], "stripes")
    stripes = tuple(_parse_stripe(value, index) for index, value in enumerate(raw_stripes, start=1))
    duplicate_ids = _duplicates(stripe.id for stripe in stripes)
    if duplicate_ids:
        raise ValueError(f"stripe registry contains duplicate stripe IDs: {duplicate_ids}")
    stripe_ids = tuple(stripe.id for stripe in stripes)
    if stripe_ids != CANONICAL_STRIPE_IDS:
        message = "stripe IDs must match the ordered 17case-v5 registry"
        raise ValueError(f"{message}; received {stripe_ids}")
    return RobustnessStripeRegistry(
        registry_version=manifest["registry_version"],
        stripes=stripes,
    )


def _parse_stripe(value: Any, position: int) -> RobustnessStripe:
    stripe = _require_mapping(value, f"stripe at position {position}")
    stripe_id = stripe.get("id", position)
    context = f"stripe {stripe_id}"
    _validate_exact_fields(stripe, _STRIPE_FIELDS, f"{context} fields")
    parsed_id = _require_nonempty_string(stripe["id"], f"{context} id")
    title = _require_nonempty_string(stripe["title"], f"{context} title")
    raw_subtypes = _require_list(stripe["allowed_subtypes"], f"{context} allowed_subtypes")
    subtypes = tuple(
        _require_nonempty_string(subtype, f"{context} allowed subtype") for subtype in raw_subtypes
    )
    if len(set(subtypes)) != len(subtypes):
        raise ValueError(f"{context} allowed_subtypes must be unique")
    return RobustnessStripe(id=parsed_id, title=title, allowed_subtypes=subtypes)


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


def _require_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{context} must not have leading or trailing whitespace")
    return value


def _require_choice(value: Any, choices: set[str], context: str) -> str:
    parsed = _require_nonempty_string(value, context)
    if parsed not in choices:
        raise ValueError(f"{context} must be one of {sorted(choices)}; received {parsed!r}")
    return parsed


def _require_literal(value: Any, expected: Any, context: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{context} must be {expected!r}; received {value!r}")


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


__all__ = [
    "CANONICAL_CASE_COUNT",
    "CANONICAL_CASE_IDS",
    "CANONICAL_STRIPE_IDS",
    "FRAMEWORK_NAME",
    "FRAMEWORK_VERSION",
    "CanonicalCase",
    "RobustnessStripe",
    "RobustnessStripeRegistry",
    "V5Registry",
    "load_case_manifest",
    "load_stripe_registry",
]
