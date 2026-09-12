"""Strict declarations for disabled V5 experimental overlays."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

from .recommendations import RECOMMENDATION_IDS, REFERENCE_IDS

REGISTRY_VERSION = "17case-v5"
EXPERIMENTAL_OVERLAY_IDS = (
    "competing_hypotheses",
    "expected_information_gain_inquiry",
    "adaptive_small_multi_agent_topology",
    "declarative_orchestration_scope",
    "mechanistic_circuit_audit",
)

_REGISTRY_FIELDS = {"registry_version", "overlays"}
_OVERLAY_FIELDS = {
    "id",
    "title",
    "maturity",
    "enabled_by_default",
    "feature_flag",
    "allowed_outputs",
    "prohibited_effects",
    "research_refs",
    "recommendation_refs",
}
_REQUIRED_PROHIBITIONS = {"canonical_case_creation", "direct_optimizer_reward"}
_ALLOWED_OUTPUTS_BY_ID = {
    "competing_hypotheses": ("hypothesis_set",),
    "expected_information_gain_inquiry": ("information_gain_question",),
    "adaptive_small_multi_agent_topology": ("topology_proposal",),
    "declarative_orchestration_scope": ("orchestration_scope_declaration",),
    "mechanistic_circuit_audit": ("mechanistic_audit_reference",),
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


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
class ExperimentalOverlay:
    """One diagnostic-only experimental capability declaration."""

    id: str
    title: str
    maturity: str
    enabled_by_default: bool
    feature_flag: str
    allowed_outputs: tuple[str, ...]
    prohibited_effects: tuple[str, ...]
    research_refs: tuple[str, ...]
    recommendation_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentalOverlayConfig:
    """Explicit opt-in flags for diagnostic overlay declarations."""

    competing_hypotheses: bool = False
    expected_information_gain_inquiry: bool = False
    adaptive_small_multi_agent_topology: bool = False
    declarative_orchestration_scope: bool = False
    mechanistic_circuit_audit: bool = False

    def __post_init__(self) -> None:
        _validated_config(self)

    @classmethod
    def from_mapping(cls, value: object) -> ExperimentalOverlayConfig:
        """Parse external configuration without accepting unknown or truthy values."""

        values = _mapping(value, "overlay configuration")
        unknown = sorted(set(values) - set(EXPERIMENTAL_OVERLAY_IDS))
        if unknown:
            raise ValueError(f"unknown overlay configuration keys: {unknown}")
        parsed: dict[str, bool] = {}
        for name, raw in values.items():
            if type(raw) is not bool:
                raise ValueError(f"{name} must be a built-in bool")
            parsed[name] = raw
        return cls(**parsed)


def load_experimental_overlay_registry() -> tuple[ExperimentalOverlay, ...]:
    """Load and validate the bundled experimental overlay registry."""

    resource = resources.files("evaluation.cases").joinpath("experimental_overlays.yaml")
    try:
        with resource.open("r", encoding="utf-8") as stream:
            payload = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load experimental overlay registry: {exc}") from exc
    return _parse_experimental_overlay_registry(payload)


def enabled_overlays(config: ExperimentalOverlayConfig) -> tuple[ExperimentalOverlay, ...]:
    """Return only explicitly enabled diagnostic declarations in registry order."""

    checked = _validated_config(config)
    return tuple(
        overlay
        for overlay in load_experimental_overlay_registry()
        if getattr(checked, overlay.feature_flag) is True
    )


def _parse_experimental_overlay_registry(payload: object) -> tuple[ExperimentalOverlay, ...]:
    registry = _mapping(payload, "registry")
    _exact_fields(registry, _REGISTRY_FIELDS, "registry fields")
    if registry["registry_version"] != REGISTRY_VERSION:
        raise ValueError(f"registry_version must be {REGISTRY_VERSION!r}")
    raw_overlays = _list(registry["overlays"], "overlays")
    overlays = tuple(_parse_overlay(item, index) for index, item in enumerate(raw_overlays, 1))
    ids = tuple(item.id for item in overlays)
    flags = tuple(item.feature_flag for item in overlays)
    duplicate_ids = _duplicates(ids)
    duplicate_flags = _duplicates(flags)
    if duplicate_ids or duplicate_flags:
        details = []
        if duplicate_ids:
            details.append(f"duplicate overlay IDs: {duplicate_ids}")
        if duplicate_flags:
            details.append(f"duplicate feature flags: {duplicate_flags}")
        raise ValueError("; ".join(details))
    if ids != EXPERIMENTAL_OVERLAY_IDS:
        raise ValueError("overlay IDs must match the ordered experimental registry")
    return overlays


def _parse_overlay(value: object, position: int) -> ExperimentalOverlay:
    overlay = _mapping(value, f"overlay at position {position}")
    _exact_fields(overlay, _OVERLAY_FIELDS, "overlay fields")
    identifier = _token(overlay["id"], "id")
    title = _token(overlay["title"], "title")
    maturity = _token(overlay["maturity"], "maturity")
    if maturity != "experimental":
        raise ValueError("maturity must be 'experimental'")
    if overlay["enabled_by_default"] is not False:
        raise ValueError("enabled_by_default must be false")
    flag = _token(overlay["feature_flag"], "feature_flag")
    outputs = _tokens(overlay["allowed_outputs"], "allowed_outputs")
    prohibited = _tokens(overlay["prohibited_effects"], "prohibited_effects")
    research = _tokens(overlay["research_refs"], "research_refs")
    recommendations = _tokens(overlay["recommendation_refs"], "recommendation_refs")
    if flag != identifier:
        raise ValueError("feature_flag must match overlay id")
    if _ALLOWED_OUTPUTS_BY_ID.get(identifier) != outputs:
        raise ValueError("allowed_outputs do not match overlay")
    if not _REQUIRED_PROHIBITIONS.issubset(prohibited):
        raise ValueError("prohibited_effects must forbid canonical cases and direct reward")
    if identifier == "mechanistic_circuit_audit" and "correlation_as_causation" not in prohibited:
        raise ValueError("mechanistic prohibited_effects must forbid correlation as causation")
    unknown_research = sorted(set(research) - set(REFERENCE_IDS))
    if unknown_research:
        raise ValueError(f"unknown research references: {unknown_research}")
    unknown_recommendations = sorted(set(recommendations) - set(RECOMMENDATION_IDS))
    if unknown_recommendations:
        raise ValueError(f"unknown recommendation references: {unknown_recommendations}")
    return ExperimentalOverlay(
        identifier,
        title,
        maturity,
        False,
        flag,
        outputs,
        prohibited,
        research,
        recommendations,
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise ValueError(f"{context} must be an exact string-keyed mapping")
    return value


def _list(value: object, context: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{context} must be an exact list")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} are invalid; expected {sorted(expected)}")


def _token(value: object, context: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{context} must be a nonblank exact string")
    return value


def _tokens(value: object, context: str) -> tuple[str, ...]:
    values = tuple(_token(item, context) for item in _list(value, context))
    if not values:
        raise ValueError(f"{context} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{context} must be unique")
    return values


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _validated_config(value: object) -> ExperimentalOverlayConfig:
    if type(value) is not ExperimentalOverlayConfig:
        raise ValueError("config must be an exact ExperimentalOverlayConfig")
    for name in EXPERIMENTAL_OVERLAY_IDS:
        if type(getattr(value, name)) is not bool:
            raise ValueError(f"{name} must be a built-in bool")
    return value


__all__ = [
    "EXPERIMENTAL_OVERLAY_IDS",
    "REGISTRY_VERSION",
    "ExperimentalOverlay",
    "ExperimentalOverlayConfig",
    "enabled_overlays",
    "load_experimental_overlay_registry",
]
