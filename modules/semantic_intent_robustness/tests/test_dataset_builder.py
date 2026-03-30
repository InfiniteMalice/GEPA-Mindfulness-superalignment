"""Tests for dataset builder artifact consistency."""

# Third-party
import pytest

# Local
from semantic_intent_robustness.dataset_builder import validate_or_regen_example_clusters


def test_example_cluster_artifact_has_no_drift() -> None:
    assert validate_or_regen_example_clusters(validate_only=True) is True


def test_validate_and_regenerate_conflict_raises() -> None:
    with pytest.raises(ValueError, match="regenerate and validate_only cannot both be True"):
        validate_or_regen_example_clusters(regenerate=True, validate_only=True)
