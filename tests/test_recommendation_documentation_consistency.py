"""Mechanical consistency checks for the V5 recommendation reader."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from os.path import relpath
from pathlib import Path

import pytest

from evaluation.recommendations import Recommendation, load_recommendation_registry

REPOSITORY_ROOT = Path(__file__).parents[1]
READER_PATH = REPOSITORY_ROOT / "docs" / "recommendations" / "UNIFIED_RECOMMENDATIONS.md"
RESEARCH_READER = "RESEARCH_TRACEABILITY.md"


def test_recommendation_reader_links_every_record_to_its_traceability_evidence() -> None:
    reader = READER_PATH.read_text(encoding="utf-8")

    for recommendation in load_recommendation_registry():
        section = _recommendation_section(reader, recommendation.recommendation_id)
        for repo_ref in recommendation.repo_refs:
            link_target = Path(relpath(REPOSITORY_ROOT / repo_ref, READER_PATH.parent)).as_posix()
            assert f"]({link_target})" in section
        for acceptance_test in recommendation.acceptance_tests:
            test_path = REPOSITORY_ROOT / acceptance_test
            if test_path.is_file():
                link_target = Path(relpath(test_path, READER_PATH.parent)).as_posix()
                assert f"]({link_target})" in section
            else:
                assert "Planned acceptance checks:" in section
                assert f"`{acceptance_test}`" in section
        for research_ref in recommendation.research_refs:
            anchor = research_ref.lower()
            assert f"[`{research_ref}`]({RESEARCH_READER}#{anchor})" in section


def test_recommendation_reader_has_exact_registry_derived_identity_inventory() -> None:
    reader = READER_PATH.read_text(encoding="utf-8")
    recommendations = load_recommendation_registry()
    expected = tuple(
        (item.recommendation_id, item.title)
        for item in _recommendations_in_reader_group_order(recommendations)
    )

    assert _reader_identity_inventory(reader) == expected


def test_recommendation_reader_has_exact_registry_derived_evidence_order() -> None:
    reader = READER_PATH.read_text(encoding="utf-8")

    for recommendation in load_recommendation_registry():
        section = _recommendation_section(reader, recommendation.recommendation_id)
        expected_refs = (
            recommendation.implementation_refs
            if recommendation.status == "implemented"
            else recommendation.repo_refs
        )
        expected_targets = tuple(
            Path(relpath(REPOSITORY_ROOT / ref, READER_PATH.parent)).as_posix()
            for ref in expected_refs
        )

        assert _repository_evidence_targets(section) == expected_targets


@pytest.mark.parametrize("mutation", ["duplicate", "extra"])
def test_recommendation_reader_rejects_additional_repository_evidence_lines(
    mutation: str,
) -> None:
    reader = READER_PATH.read_text(encoding="utf-8")
    recommendation = load_recommendation_registry()[0]
    section = _recommendation_section(reader, recommendation.recommendation_id)
    evidence_line = next(
        line for line in section.splitlines() if line.startswith("- Repository evidence:")
    )
    added_line = (
        evidence_line
        if mutation == "duplicate"
        else "- Repository evidence: [`extra.py`](../../tests/test_schema_v3.py)."
    )
    mutated_section = section.replace(evidence_line, f"{evidence_line}\n{added_line}", 1)

    with pytest.raises(AssertionError, match="exactly one repository evidence line"):
        _repository_evidence_targets(mutated_section)


def test_recommendation_reader_relative_links_resolve() -> None:
    reader = READER_PATH.read_text(encoding="utf-8")

    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", reader):
        if target.startswith("#"):
            continue
        path_target = target.partition("#")[0]
        assert (READER_PATH.parent / path_target).resolve().is_file(), target


def test_recommendation_reader_groups_records_by_priority_and_status() -> None:
    reader = READER_PATH.read_text(encoding="utf-8")

    for recommendation in load_recommendation_registry():
        group = _priority_status_section(reader, recommendation.priority, recommendation.status)
        assert f"#### {recommendation.recommendation_id} —" in group


def _recommendation_section(reader: str, recommendation_id: str) -> str:
    match = re.search(
        rf"^#### {re.escape(recommendation_id)}\b.*?(?=^#### |\Z)",
        reader,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing reader section for {recommendation_id}"
    return match.group(0)


def _priority_status_section(reader: str, priority: str, status: str) -> str:
    heading = f"{priority} — {status.title()}"
    match = re.search(
        rf"^### {re.escape(heading)}\b.*?(?=^### |\Z)",
        reader,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing priority/status group {heading}"
    return match.group(0)


def _recommendations_in_reader_group_order(
    recommendations: Iterable[Recommendation],
) -> Iterator[Recommendation]:
    recommendations = tuple(recommendations)
    priorities = dict.fromkeys(item.priority for item in recommendations)
    for priority in priorities:
        priority_records = tuple(item for item in recommendations if item.priority == priority)
        statuses = dict.fromkeys(item.status for item in priority_records)
        for status in statuses:
            yield from (item for item in priority_records if item.status == status)


def _reader_identity_inventory(reader: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        re.findall(
            r"^#### (REC-\d{3}) — (.+)$",
            reader,
            flags=re.MULTILINE,
        )
    )


def _repository_evidence_targets(section: str) -> tuple[str, ...]:
    evidence_lines = re.findall(
        r"^- Repository evidence: (.*)$",
        section,
        flags=re.MULTILINE,
    )
    targets = tuple(
        target
        for evidence_line in evidence_lines
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", evidence_line)
    )
    assert (
        len(evidence_lines) == 1
    ), f"expected exactly one repository evidence line; found {len(evidence_lines)}"
    assert targets, "repository evidence line must contain at least one link"
    assert len(targets) == len(set(targets)), "repository evidence links must be unique"
    return targets
