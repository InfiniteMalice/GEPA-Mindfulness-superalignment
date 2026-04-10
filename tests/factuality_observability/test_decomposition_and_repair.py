from gepa_mindfulness.factuality_observability.decomposition import decompose_verify_and_repair


def test_decomposition_detects_unsupported_fact_and_repairs() -> None:
    answer = "Paris is in France and Paris has 4 moons."
    evidence_lookup = {"Paris is in France": ["atlas"]}

    result = decompose_verify_and_repair(answer, evidence_lookup)

    assert result.atomic_fact_list
    assert result.unsupported_fact_indices
    assert "Paris is in France" in result.repaired_answer
