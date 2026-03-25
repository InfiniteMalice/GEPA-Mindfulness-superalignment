"""Tests for dataset builder artifact consistency."""

# Local
from semantic_intent_robustness.dataset_builder import validate_or_regen_example_clusters


def test_example_cluster_artifact_has_no_drift() -> None:
    assert validate_or_regen_example_clusters(validate_only=True) is True
