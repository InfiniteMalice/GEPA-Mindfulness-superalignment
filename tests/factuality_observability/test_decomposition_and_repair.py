from gepa_mindfulness.factuality_observability.decomposition import decompose_verify_and_repair


def test_decomposition_detects_unsupported_fact_and_repairs() -> None:
    answer = "Paris is in France and Paris has 4 moons."
    evidence_lookup = {"Paris is in France": ["atlas"]}

    result = decompose_verify_and_repair(answer, evidence_lookup)

    assert result.atomic_fact_list == ["Paris is in France", "Paris has 4 moons."]
    assert result.unsupported_fact_indices == [1]
    assert "Paris is in France" in result.repaired_answer
    assert "Paris has 4 moons" not in result.repaired_answer
