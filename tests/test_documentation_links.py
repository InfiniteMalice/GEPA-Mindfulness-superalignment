"""Resolve repository-local links in the V5 documentation surface."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from evaluation.recommendations import (
    load_recommendation_registry,
    load_research_reference_registry,
)

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "17_CASE_FRAMEWORK.md",
    ROOT / "docs" / "ALIGNMENT_EVAL_BATTERY.md",
    ROOT / "docs" / "structured_logging.md",
    ROOT / "docs" / "thought_alignment.md",
    ROOT / "docs" / "epistemic_process_rewards.md",
    ROOT / "docs" / "controlled_evolution.md",
    ROOT / "docs" / "VERIFICATION_AND_RUNTIME_AUTHORITY.md",
    ROOT / "docs" / "FOUNDATIONAL_REPRESENTATION_ARCHITECTURE.md",
    ROOT / "docs" / "experimental_v5_overlays.md",
    ROOT / "docs" / "adr" / "0001-17-case-v5-unified-architecture.md",
    ROOT / "docs" / "recommendations" / "UNIFIED_RECOMMENDATIONS.md",
    ROOT / "docs" / "recommendations" / "RESEARCH_TRACEABILITY.md",
    ROOT / "gepa_mindfulness" / "core" / "README.md",
    ROOT / "gepa_mindfulness" / "factuality_observability" / "README.md",
    ROOT / "gepa_mindfulness" / "schema_v3" / "README.md",
    ROOT / "modules" / "semantic_intent_robustness" / "README.md",
    ROOT / "modules" / "objective_validator_robustness" / "README.md",
)

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_EXPLICIT_ID = re.compile(r"<a\s+id=[\"']([^\"']+)[\"']\s*></a>", re.IGNORECASE)
_REC_ID = re.compile(r"\bREC-\d{3}\b")
_REF_ID = re.compile(r"\bREF-[A-Z0-9-]+\b")


def _markdown_link_targets(markdown: str) -> tuple[str, ...]:
    visible_lines: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if fence is None and marker in {"```", "~~~"}:
            fence = marker
            continue
        if fence == marker:
            fence = None
            continue
        if fence is None:
            visible_lines.append(re.sub(r"`[^`\n]*`", "", line))
    return tuple(_MARKDOWN_LINK.findall("\n".join(visible_lines)))


def test_v5_documentation_relative_links_and_fragments_resolve() -> None:
    failures: list[str] = []

    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        for raw_target in _markdown_link_targets(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target:
                failures.append(f"{document.relative_to(ROOT)} -> empty link target")
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not any(character in target for character in ("/", "\\", ".", "#")):
                continue

            path_text, separator, fragment = target.partition("#")
            linked_path = document if not path_text else document.parent / unquote(path_text)
            linked_path = linked_path.resolve()
            if not linked_path.exists():
                failures.append(f"{document.relative_to(ROOT)} -> missing {target}")
                continue
            if separator and linked_path.is_file() and linked_path.suffix.lower() == ".md":
                anchors = _github_anchors(linked_path.read_text(encoding="utf-8"))
                if unquote(fragment).lower() not in anchors:
                    failures.append(f"{document.relative_to(ROOT)} -> missing fragment {target}")

    assert not failures, "\n".join(failures)


def test_markdown_link_targets_ignore_code_and_retain_empty_targets() -> None:
    markdown = """
[real](docs/README.md)
`[inline](missing-inline.md)`
```markdown
[fenced](missing-fenced.md)
```
[empty]( )
"""

    assert _markdown_link_targets(markdown) == ("docs/README.md", " ")


def test_v5_documentation_traceability_ids_resolve_to_registries() -> None:
    recommendation_ids = {
        recommendation.recommendation_id for recommendation in load_recommendation_registry()
    }
    reference_ids = {reference.reference_id for reference in load_research_reference_registry()}
    found_recommendations: set[str] = set()
    found_references: set[str] = set()

    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        found_recommendations.update(_REC_ID.findall(text))
        found_references.update(_REF_ID.findall(text))

    assert found_recommendations <= recommendation_ids
    assert found_references <= reference_ids
    assert recommendation_ids <= found_recommendations
    assert reference_ids <= found_references


def _github_anchors(document: str) -> set[str]:
    anchors = {match.lower() for match in _EXPLICIT_ID.findall(document)}
    occurrences: dict[str, int] = {}
    for heading in _HEADING.findall(document):
        plain = re.sub(r"<[^>]+>", "", heading)
        plain = re.sub(r"[`*_~]", "", plain).lower().strip()
        base = re.sub(r"[^\w\- ]", "", plain, flags=re.UNICODE).replace(" ", "-")
        suffix = occurrences.get(base, 0)
        occurrences[base] = suffix + 1
        anchors.add(base if suffix == 0 else f"{base}-{suffix}")
    return anchors
