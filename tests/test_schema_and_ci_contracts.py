"""Compatibility contracts for schema and CI coverage."""

# Standard library
import subprocess
import sys
from pathlib import Path

# Local
from evaluation.cases import load_case_manifest, load_stripe_registry
from gepa_mindfulness.core.clarifying_abstention import (
    FRAMEWORK_CASE_IDS,
    ORIGINAL_CASE_IDS,
)


def test_existing_17_case_schema_unchanged() -> None:
    manifest = load_case_manifest()

    assert manifest.framework_name == "GEPA Mindfulness 17-Case Framework V5"
    assert manifest.framework_version == "17case-v5"
    assert manifest.canonical_case_count == 17
    assert FRAMEWORK_CASE_IDS == tuple(case.id for case in manifest.cases)
    assert ORIGINAL_CASE_IDS == tuple(
        case.id for case in manifest.cases if "v3" in case.compatibility["legacy_versions"]
    )
    assert tuple((case.id, case.key, case.title) for case in manifest.cases[13:]) == (
        (
            14,
            "correct_high_stakes_clarifying_abstention",
            "Correct high-stakes clarifying abstention",
        ),
        (15, "over_eager_ambiguous_compliance", "Over-eager ambiguous/high-stakes compliance"),
        (
            16,
            "unnecessary_clarification_on_low_stakes_ambiguity",
            "Unnecessary low-stakes clarification",
        ),
        (
            17,
            "clarification_loop_or_failure_to_resume",
            "Clarification loop, repeated unnecessary questioning, or failure to resume after "
            "sufficient clarification",
        ),
    )


def test_v5_robustness_stripe_ids_are_frozen() -> None:
    assert tuple(stripe.id for stripe in load_stripe_registry().stripes) == (
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


def test_rg_tracer_derived_case_view_matches_canonical_manifest() -> None:
    """The dependency-free tracer view must match the authored manifest exactly."""

    from rg_tracer.schema_v3._case_manifest_v5 import CASE_IDENTITIES, FRAMEWORK_VERSION

    manifest = load_case_manifest()

    assert FRAMEWORK_VERSION == manifest.framework_version
    assert CASE_IDENTITIES == tuple((case.id, case.key) for case in manifest.cases)


def test_rg_tracer_schema_v3_imports_with_only_its_source_tree(tmp_path: Path) -> None:
    """The standalone tracer package must not import the repository registry or PyYAML."""

    root = Path(__file__).resolve().parents[1]
    tracer_src = root / "reasoning-generalization-tracer" / "src"
    probe = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(tracer_src)!r})",
            "sys.path[:] = [sys.path[0]] + [",
            "    entry for entry in sys.path[1:]",
            "    if 'site-packages' not in entry",
            "    and '__editable__' not in entry",
            "    and entry != sys.prefix",
            "]",
            "from rg_tracer.schema_v3 import (",
            "    CANONICAL_CASE_NAMES,",
            "    CASE_NAMES,",
            "    classify_case_v3,",
            ")",
            "assert tuple(CANONICAL_CASE_NAMES) == tuple(range(1, 18))",
            "assert tuple(CASE_NAMES) == tuple(range(0, 18))",
            "assert CASE_NAMES[0] == 'fallback_or_internal_error'",
            "assert 'evaluation' not in sys.modules",
            "assert 'yaml' not in sys.modules",
            "result = classify_case_v3(",
            "    output_text='fallback',",
            "    expected_answer=None,",
            "    is_idk=False,",
            "    confidence=None,",
            "    thought_aligned=True,",
            ")",
            "assert result.case_id == 0",
            "assert result.base_case_name == 'fallback_or_internal_error'",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
