"""Compatibility contracts for schema and CI coverage."""

# Local
from evaluation.cases import load_case_manifest
from gepa_mindfulness.core.clarifying_abstention import (
    FRAMEWORK_CASE_IDS,
    ORIGINAL_CASE_IDS,
)


def test_existing_17_case_schema_unchanged() -> None:
    manifest = load_case_manifest()

    assert FRAMEWORK_CASE_IDS == tuple(case.id for case in manifest.cases)
    assert ORIGINAL_CASE_IDS == tuple(
        case.id for case in manifest.cases if "v3" in case.compatibility["legacy_versions"]
    )
