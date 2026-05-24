from evaluation.registry import STANDARD_CATEGORIES, get_suite_metadata, list_alignment_suites


def test_registry_includes_required_categories_and_families() -> None:
    suites = list_alignment_suites()
    categories = {suite.category for suite in suites}
    families = {suite.benchmark_family for suite in suites}

    assert set(STANDARD_CATEGORIES) <= categories
    assert {"SimpleQA", "TruthfulQA", "FEVER", "HELM", "HarmBench", "WMDP", "BBQ"} <= families
    assert "JailbreakBench" in families
    assert "AgentHarm" in families
    assert "RealToxicityPrompts" in families
    assert "DecodingTrust" in families


def test_get_suite_metadata() -> None:
    metadata = get_suite_metadata("simpleqa")

    assert metadata.category == "factuality"
    assert metadata.default_run_tier == "ci"
    assert metadata.requires_external_data is True
