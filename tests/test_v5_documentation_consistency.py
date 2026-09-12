"""Keep documented V5 identities aligned with their canonical registries."""

import re
from pathlib import Path

from evaluation.cases import load_case_manifest, load_stripe_registry

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "17_CASE_FRAMEWORK.md"
README_PATH = ROOT / "README.md"
FACTUALITY_PATH = ROOT / "gepa_mindfulness" / "factuality_observability" / "README.md"
START_MARKER = "<!-- canonical-cases:start -->"
END_MARKER = "<!-- canonical-cases:end -->"
STRIPE_START_MARKER = "<!-- canonical-stripes:start -->"
STRIPE_END_MARKER = "<!-- canonical-stripes:end -->"
README_TOP_LEVEL_HEADINGS = (
    "Project Description and Navigation",
    "Project Status and Maturity",
    "Quick Start",
    "Core Alignment Architecture",
    "17-Case Framework V5",
    "Reward and Epistemic Process",
    "Safety and Robustness Modules",
    "Verification, Evidence, and Provenance",
    "Deception and Interpretability",
    "Training and Runtime",
    "Evaluation",
    "Datasets and Synthetic Data",
    "Repository Layout",
    "Research Basis and Traceability",
    "Limitations and Research Maturity",
    "Contributing and License",
)


def test_documented_canonical_cases_match_manifest() -> None:
    """The marked documentation table is the manifest's complete identity view."""

    document = DOC_PATH.read_text(encoding="utf-8")
    marked_table = _canonical_cases_table(document)

    assert _parse_case_rows(marked_table) == [
        (str(case.id), case.key, case.title) for case in load_case_manifest().cases
    ]


def test_readme_names_the_canonical_framework_and_reproduces_all_case_identities() -> None:
    manifest = load_case_manifest()
    readme = README_PATH.read_text(encoding="utf-8")

    assert manifest.framework_name in readme
    assert f"framework version `{manifest.framework_version}`" in readme
    assert "exactly **17 canonical cases**" in readme
    assert _parse_case_rows(_marked_section(readme, START_MARKER, END_MARKER)) == [
        (str(case.id), case.key, case.title) for case in manifest.cases
    ]


def test_readme_uses_the_approved_conceptual_order_and_reward_boundaries() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert tuple(re.findall(r"^## (.+)$", readme, flags=re.MULTILINE)) == README_TOP_LEVEL_HEADINGS
    assert "`CASE × STRIPE × REPEAT`" in readme
    assert "`thought_align` is a diagnostic compatibility field" in readme
    assert "Deception signals, unverified trace content, circuit features" in readme
    assert "PyTorch CUDA and distributed | Implemented; hardware unqualified" in readme
    assert "Pure Mojo learner | Unsupported" in readme


def test_framework_document_lists_exact_canonical_stripe_ids() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")
    marked = _marked_section(document, STRIPE_START_MARKER, STRIPE_END_MARKER)
    documented_ids = tuple(
        match.group(1) for line in marked.splitlines() if (match := re.match(r"^- `([^`]+)`", line))
    )

    assert documented_ids == tuple(stripe.id for stripe in load_stripe_registry().stripes)


def test_factuality_document_freezes_case_14_through_17_meanings() -> None:
    manifest = load_case_manifest()
    factuality = FACTUALITY_PATH.read_text(encoding="utf-8")
    documented_rows = _parse_case_rows(_canonical_cases_table(factuality))

    assert documented_rows == [
        (str(case.id), case.key, case.title) for case in manifest.cases if case.id >= 14
    ]


def _canonical_cases_table(document: str) -> str:
    """Return the uniquely marked canonical-case table from the document."""

    return _marked_section(document, START_MARKER, END_MARKER)


def _marked_section(document: str, start_marker: str, end_marker: str) -> str:
    """Return one uniquely marked documentation section."""

    assert document.count(start_marker) == 1
    assert document.count(end_marker) == 1
    start = document.index(start_marker) + len(start_marker)
    end = document.index(end_marker)
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
    return [(row[0], row[1], row[2]) for row in rows[2:]]
