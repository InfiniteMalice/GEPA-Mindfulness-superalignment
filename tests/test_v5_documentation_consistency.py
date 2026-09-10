"""Keep the documented V5 canonical-case table aligned with its manifest."""

from pathlib import Path

from evaluation.cases import load_case_manifest

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "17_CASE_FRAMEWORK.md"
START_MARKER = "<!-- canonical-cases:start -->"
END_MARKER = "<!-- canonical-cases:end -->"


def test_documented_canonical_cases_match_manifest() -> None:
    """The marked documentation table is the manifest's complete identity view."""

    document = DOC_PATH.read_text(encoding="utf-8")
    marked_table = _canonical_cases_table(document)

    assert _parse_case_rows(marked_table) == [
        (str(case.id), case.key, case.title) for case in load_case_manifest().cases
    ]


def _canonical_cases_table(document: str) -> str:
    """Return the uniquely marked canonical-case table from the document."""

    assert document.count(START_MARKER) == 1
    assert document.count(END_MARKER) == 1
    start = document.index(START_MARKER) + len(START_MARKER)
    end = document.index(END_MARKER)
    assert start < end
    return document[start:end]


def _parse_case_rows(table: str) -> list[tuple[str, str, str]]:
    """Parse ID, machine key, and human title cells from a Markdown table."""

    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table.splitlines()
        if line.strip().startswith("|")
    ]
    assert rows[0] == ["ID", "Machine key", "Human title"]
    assert all(set(cell) == {"-"} for cell in rows[1])
    assert all(len(row) == 3 for row in rows[2:])
    return [tuple(row) for row in rows[2:]]
