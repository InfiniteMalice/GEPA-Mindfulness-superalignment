from gepa_mindfulness.factuality_observability.consistency import (
    QueryRecord,
    compute_related_query_consistency,
)
from gepa_mindfulness.factuality_observability.schemas import (
    GuessingAbstentionDiagnostics,
    HallucinationPrimaryType,
)


def test_taxonomy_enum_is_stable() -> None:
    assert HallucinationPrimaryType.CITATION.value == "citation_hallucination"


def test_guessing_abstention_diagnostics_fields() -> None:
    diag = GuessingAbstentionDiagnostics(
        model_guessed_when_uncertain=True,
        binary_eval_pressure=0.9,
        would_binary_eval_reward_guess=True,
    )
    assert diag.model_guessed_when_uncertain
    assert diag.binary_eval_pressure > 0.5


def test_related_query_consistency_analysis() -> None:
    consistency = compute_related_query_consistency(
        query_family_id="fam1",
        paraphrase_group_id="para1",
        records=[
            QueryRecord(query_id="q1", answer="Paris is in France", abstained=False),
            QueryRecord(query_id="q2", answer="Paris is in France", abstained=False),
        ],
    )
    assert consistency.cross_prompt_answer_consistency_score == 1.0
