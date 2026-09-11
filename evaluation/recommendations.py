"""Strict package-resource loaders for V5 recommendation and research registries."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import PurePosixPath
from typing import Any

import yaml

REGISTRY_VERSION = "17case-v5"
RECOMMENDATION_IDS = tuple(f"REC-{number:03d}" for number in range(1, 15))
RECOMMENDATION_PRIORITIES = ("P0",) * 5 + ("P1",) * 5 + ("P2",) * 4
ALLOWED_PRIORITIES = frozenset({"P0", "P1", "P2"})
ALLOWED_STATUSES = frozenset(
    {"proposed", "experimental", "accepted", "implemented", "rejected", "superseded"}
)
REFERENCE_IDS = (
    "REF-HEART",
    "REF-PEARL",
    "REF-SEGOS",
    "REF-COEVOLVE",
    "REF-CONSISTENCY",
    "REF-DSR",
    "REF-EDGEMEM",
    "REF-GRAPHMEM",
    "REF-SHEAVES",
    "REF-HERO",
    "REF-SAE",
    "REF-BIOMETRIC-MEM",
    "REF-HOH",
    "REF-AGENTSCOPE",
    "REF-REPOTOSKILL",
    "REF-SKILLGLOW",
    "REF-MASKILLS",
    "REF-WMLLM",
    "REF-DWM",
    "REF-LEXICAL-PERTURB",
    "REF-TOKENIZER-BETRAYAL",
)
REFERENCE_ARXIV_IDS = (
    "2609.01736",
    "2609.02216",
    "2609.08228",
    "2609.09134",
    "2609.08832",
    "2609.05824",
    "2609.05553",
    "2609.08599",
    "2609.09056",
    "2609.08189",
    "2609.09113",
    "2609.08558",
    "2609.01481",
    "2609.02371",
    "2609.02749",
    "2609.02217",
    "2609.02094",
    "2609.01608",
    "2609.02885",
    "2608.22140",
    "2601.14658",
)
ALLOWED_METADATA_STATUSES = frozenset({"resolved", "unresolved"})

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
_RESEARCH_REGISTRY_FIELDS = {"registry_version", "references"}
_RESEARCH_REFERENCE_FIELDS = {
    "reference_id",
    "metadata_status",
    "supplied_title",
    "title",
    "authors",
    "year",
    "arxiv_id",
    "doi",
    "venue_status",
    "canonical_url",
    "recommendation_ids",
    "source_demonstrates",
    "repository_inference",
    "maturity",
    "local_repo_notes",
}
_ARXIV_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}\Z")
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+\Z")


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


@dataclass(frozen=True, slots=True)
class ResearchReference:
    """One primary-source research record and its restrained repository connection."""

    reference_id: str
    metadata_status: str
    supplied_title: str
    title: str | None
    authors: tuple[str, ...]
    year: int | None
    arxiv_id: str
    doi: str | None
    venue_status: str | None
    canonical_url: str
    recommendation_ids: tuple[str, ...]
    source_demonstrates: str
    repository_inference: str
    maturity: str
    local_repo_notes: tuple[str, ...]


def load_recommendation_registry() -> tuple[Recommendation, ...]:
    """Load and validate the bundled V5 recommendation registry."""

    resource = resources.files("docs.recommendations").joinpath("registry.yaml")
    try:
        with resource.open("r", encoding="utf-8") as stream:
            payload = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load bundled recommendation registry: {exc}") from exc
    return _parse_recommendation_registry(payload)


def load_research_reference_registry() -> tuple[ResearchReference, ...]:
    """Load and validate the bundled V5 primary-source reference registry."""

    resource = resources.files("docs.recommendations").joinpath("references.yaml")
    try:
        with resource.open("r", encoding="utf-8") as stream:
            payload = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load bundled research reference registry: {exc}") from exc
    return _parse_research_reference_registry(payload)


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


def _parse_research_reference_registry(payload: Any) -> tuple[ResearchReference, ...]:
    """Validate one primary-source registry payload loaded from authored YAML."""

    registry = _require_mapping(payload, "research reference registry")
    _validate_exact_fields(
        registry,
        _RESEARCH_REGISTRY_FIELDS,
        "research reference registry fields",
    )
    _require_literal(registry["registry_version"], REGISTRY_VERSION, "registry_version")

    raw_references = _require_list(registry["references"], "references")
    references = tuple(
        _parse_research_reference(value, position)
        for position, value in enumerate(raw_references, start=1)
    )
    _validate_reference_sequence(references)
    _validate_reference_relationships(references)
    return references


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


def _parse_research_reference(value: Any, position: int) -> ResearchReference:
    record = _require_mapping(value, f"research reference at position {position}")
    reference_id = record.get("reference_id", f"position {position}")
    context = f"{reference_id}"
    _validate_exact_fields(record, _RESEARCH_REFERENCE_FIELDS, f"{context} fields")

    parsed_id = _require_nonempty_string(record["reference_id"], f"{context} reference_id")
    metadata_status = _require_choice(
        record["metadata_status"],
        ALLOWED_METADATA_STATUSES,
        f"{context} metadata_status",
    )
    arxiv_id = _require_arxiv_id(record["arxiv_id"], f"{context} arxiv_id")
    title = _require_optional_string(record["title"], f"{context} title")
    authors = _require_string_tuple(record["authors"], f"{context} authors")
    year = _require_optional_year(record["year"], f"{context} year")
    doi = _require_optional_doi(record["doi"], f"{context} doi")
    venue_status = _require_optional_string(
        record["venue_status"],
        f"{context} venue_status",
    )
    source_demonstrates = _require_nonempty_string(
        record["source_demonstrates"],
        f"{context} source_demonstrates",
    )
    _validate_metadata_by_status(
        reference_id=parsed_id,
        metadata_status=metadata_status,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        venue_status=venue_status,
        source_demonstrates=source_demonstrates,
    )

    expected_url = f"https://arxiv.org/abs/{arxiv_id}"
    canonical_url = _require_nonempty_string(
        record["canonical_url"],
        f"{context} canonical_url",
    )
    _require_literal(canonical_url, expected_url, f"{context} canonical_url")
    local_repo_notes = _require_string_tuple(
        record["local_repo_notes"],
        f"{context} local_repo_notes",
        required=True,
    )
    for note in local_repo_notes:
        _validate_repo_relative_path(note, f"{context} local_repo_notes item")

    return ResearchReference(
        reference_id=parsed_id,
        metadata_status=metadata_status,
        supplied_title=_require_nonempty_string(
            record["supplied_title"],
            f"{context} supplied_title",
        ),
        title=title,
        authors=authors,
        year=year,
        arxiv_id=arxiv_id,
        doi=doi,
        venue_status=venue_status,
        canonical_url=canonical_url,
        recommendation_ids=_require_string_tuple(
            record["recommendation_ids"],
            f"{context} recommendation_ids",
            required=True,
        ),
        source_demonstrates=source_demonstrates,
        repository_inference=_require_nonempty_string(
            record["repository_inference"],
            f"{context} repository_inference",
        ),
        maturity=_require_nonempty_string(record["maturity"], f"{context} maturity"),
        local_repo_notes=local_repo_notes,
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


def _validate_reference_sequence(references: tuple[ResearchReference, ...]) -> None:
    identifiers = tuple(record.reference_id for record in references)
    duplicate_ids = _duplicates(identifiers)
    if duplicate_ids:
        raise ValueError(f"research reference registry contains duplicate IDs: {duplicate_ids}")
    if identifiers != REFERENCE_IDS:
        raise ValueError(
            "research reference IDs must be the ordered requested sequence; "
            f"received {identifiers}"
        )

    arxiv_ids = tuple(record.arxiv_id for record in references)
    duplicate_arxiv_ids = _duplicates(arxiv_ids)
    if duplicate_arxiv_ids:
        raise ValueError(
            "research reference registry contains duplicate arXiv IDs: " f"{duplicate_arxiv_ids}"
        )
    if arxiv_ids != REFERENCE_ARXIV_IDS:
        raise ValueError(
            "research reference arXiv IDs must be the ordered supplied arXiv sequence; "
            f"received {arxiv_ids}"
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


def _validate_reference_relationships(references: tuple[ResearchReference, ...]) -> None:
    known_ids = {record.recommendation_id for record in load_recommendation_registry()}
    for record in references:
        unknown = sorted(set(record.recommendation_ids) - known_ids)
        if unknown:
            raise ValueError(
                f"{record.reference_id} recommendation_ids reference unknown IDs: {unknown}"
            )


def _validate_metadata_by_status(
    *,
    reference_id: str,
    metadata_status: str,
    title: str | None,
    authors: tuple[str, ...],
    year: int | None,
    doi: str | None,
    venue_status: str | None,
    source_demonstrates: str,
) -> None:
    if metadata_status == "resolved":
        missing = []
        if title is None:
            missing.append("title")
        if not authors:
            missing.append("authors")
        if year is None:
            missing.append("year")
        if missing:
            raise ValueError(f"{reference_id} resolved metadata must include fields: {missing}")
        return

    unsupported = []
    if title is not None:
        unsupported.append("title")
    if authors:
        unsupported.append("authors")
    if year is not None:
        unsupported.append("year")
    if doi is not None:
        unsupported.append("doi")
    if venue_status is not None:
        unsupported.append("venue_status")
    if unsupported:
        raise ValueError(
            f"{reference_id} unresolved metadata must omit unsupported fields: {unsupported}"
        )
    if not source_demonstrates.startswith("No source claim has been verified"):
        raise ValueError(
            f"{reference_id} unresolved metadata source_demonstrates must state that no "
            "source claim has been verified"
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


def _require_optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, context)


def _require_optional_year(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer year or null")
    if not 1900 <= value <= 9999:
        raise ValueError(f"{context} must be between 1900 and 9999; received {value!r}")
    return value


def _require_arxiv_id(value: Any, context: str) -> str:
    parsed = _require_nonempty_string(value, context)
    if _ARXIV_ID_PATTERN.fullmatch(parsed) is None:
        raise ValueError(f"{context} must use the canonical numeric arXiv ID format")
    return parsed


def _require_optional_doi(value: Any, context: str) -> str | None:
    parsed = _require_optional_string(value, context)
    if parsed is not None and _DOI_PATTERN.fullmatch(parsed) is None:
        raise ValueError(f"{context} must use DOI syntax beginning with '10.'")
    return parsed


def _validate_repo_relative_path(value: str, context: str) -> None:
    path = PurePosixPath(value)
    if "\\" in value or path.is_absolute() or ".." in path.parts or value.startswith("./"):
        raise ValueError(f"{context} must be a normalized repository-relative path")


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
    "ALLOWED_METADATA_STATUSES",
    "ALLOWED_PRIORITIES",
    "ALLOWED_STATUSES",
    "REFERENCE_ARXIV_IDS",
    "REFERENCE_IDS",
    "REGISTRY_VERSION",
    "RECOMMENDATION_IDS",
    "Recommendation",
    "ResearchReference",
    "load_recommendation_registry",
    "load_research_reference_registry",
]
