"""Mechanical consistency checks for the V5 recommendation reader."""

from __future__ import annotations

import re
from os.path import relpath
from pathlib import Path

from evaluation.recommendations import load_recommendation_registry

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
