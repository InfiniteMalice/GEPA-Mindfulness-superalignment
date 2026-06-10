"""Compatibility contracts for schema and CI coverage."""

# Local
from gepa_mindfulness.core.clarifying_abstention import FRAMEWORK_CASE_IDS


def test_existing_17_case_schema_unchanged() -> None:
    assert FRAMEWORK_CASE_IDS == tuple(range(1, 18))
