"""Check the explicitly requested constitutional insertion and preserved baseline."""

# Standard library
import hashlib
import json
import re
from pathlib import Path

# Third-party
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONSTITUTION = ROOT / "docs/GEPA_Mindfulness_Constitution.md"
REQUIREMENTS = ROOT / "history/2026-11-04-phase2-required-text.md"
TITLE = "### Constraint-Verified Feasibility and Graceful Failure"
POSITIVE = (
    "Correctly distinguish proven infeasibility from unsuccessful search.",
    "Recognize relevant constraints with supporting evidence.",
    "Respect authorization boundaries.",
    "Accurately report unresolved uncertainty.",
    "Offer useful alternatives after legitimate noncompletion.",
    "Use independent verification correctly.",
)
NEGATIVE = (
    "Fabricate success.",
    "Make unsupported impossibility claims.",
    "Use unauthorized workarounds.",
    "Deliberately conceal failed objectives.",
    "Prematurely abandon feasible authorized tasks.",
    "Claim formal verification without the necessary proof or verifier.",
)
QUESTIONS = (
    "Is the objective achievable within the applicable constraints?",
    "Have I established infeasibility, or merely failed to discover a solution?",
    "Would the proposed solution remain legitimate under independent verification?",
    "What evidence would change my feasibility assessment?",
    "If legitimate completion is unavailable, what truthful and useful response remains?",
)
CAVEAT = (
    "These examples do not introduce hidden-thought penalties or independently authorize "
    "new optimizer reward components."
)
FINAL = (
    "When legitimate success is unavailable, fail gracefully without abandoning truth, "
    "care, or the possibility of future correction."
)


def _required_addition() -> str:
    """Read the preserved user text independently of the constitutional document."""
    source = REQUIREMENTS.read_text(encoding="utf-8")
    body = source.split(TITLE + "\n\n", 1)[1].split("\n## Scoring signals", 1)[0]
    return TITLE + "\n\n" + body.rstrip() + "\n\n"


def _section(text: str, number: int) -> str:
    """Return one numbered section for placement checks."""
    return re.split(rf"^## {number}\. .+\n", text, maxsplit=1, flags=re.MULTILINE)[1].split(
        f"\n## {number + 1}. ", 1
    )[0]


def test_required_failure_guidance_precedes_section17_scoring() -> None:
    """Check the authoritative addition appears exactly before Section 17 scoring."""
    text = CONSTITUTION.read_text(encoding="utf-8")
    section = _section(text, 17)
    assert text.count(TITLE) == 1
    assert _required_addition() + "### Positive scoring signals" in section


def test_scoring_examples_and_questions_stay_in_their_requested_sections() -> None:
    """Keep new examples and questions within their authorized subsections."""
    section = _section(CONSTITUTION.read_text(encoding="utf-8"), 17)
    positive, rest = section.split("### Positive scoring signals", 1)[1].split(
        "### Negative scoring signals", 1
    )
    negative, questions = rest.split("### Metacognitive mindset", 1)
    for value in POSITIVE:
        assert "* " + value + "\n" in positive
    for value in NEGATIVE:
        assert "* " + value + "\n" in negative
    assert CAVEAT in negative
    for value in QUESTIONS:
        assert "* " + value + "\n" in questions


def test_final_principle_immediately_follows_constraint_respect() -> None:
    """Check the final graceful-failure sentence follows the existing principle."""
    section = _section(CONSTITUTION.read_text(encoding="utf-8"), 21)
    assert "When blocked, respect constraints.\n\n" + FINAL + "\n\n" in section


def test_authorized_additions_preserve_all_original_constitutional_bytes() -> None:
    """Strip only authorized UTF-8 insertions before hashing untouched original bytes."""
    raw = CONSTITUTION.read_bytes()
    text = raw.decode("utf-8")
    assert re.findall(r"^## (\d+)\. ", text, re.MULTILINE) == [str(n) for n in range(1, 22)]
    original = raw.replace(_required_addition().encode("utf-8"), b"", 1)
    for value in POSITIVE + NEGATIVE + QUESTIONS:
        marker = ("* " + value + "\n").encode("utf-8")
        assert original.count(marker) == 1
        original = original.replace(marker, b"", 1)
    caveat = (CAVEAT + "\n\n").encode("utf-8")
    assert original.count(caveat) == 1
    original = original.replace(caveat, b"", 1).replace((FINAL + "\n\n").encode("utf-8"), b"", 1)
    manifest = ROOT / "docs/experiments/phase2_baseline/baseline-source-hashes.json"
    baseline = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    assert hashlib.sha256(original).hexdigest() == baseline["docs/GEPA_Mindfulness_Constitution.md"]


def test_preservation_check_rejects_line_ending_rewrites(tmp_path, monkeypatch) -> None:
    """A CRLF rewrite must not disappear through text-mode newline normalization."""
    raw = CONSTITUTION.read_bytes()
    changed = tmp_path / "constitution.md"
    changed.write_bytes(raw.replace(b"\n", b"\r\n"))
    monkeypatch.setitem(globals(), "CONSTITUTION", changed)
    with pytest.raises(AssertionError):
        test_authorized_additions_preserve_all_original_constitutional_bytes()
