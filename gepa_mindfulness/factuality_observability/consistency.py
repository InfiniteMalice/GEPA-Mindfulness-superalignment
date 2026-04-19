"""Related-query consistency diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import RelatedQueryConsistency


@dataclass(slots=True)
class QueryRecord:
    """Minimal record for consistency comparisons."""

    query_id: str
    answer: str
    abstained: bool


def _pairwise_similarity(left: str, right: str) -> float:
    left_tokens = set(left.lower().split())
    right_tokens = set(right.lower().split())
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def compute_related_query_consistency(
    query_family_id: str,
    paraphrase_group_id: str,
    records: list[QueryRecord],
) -> RelatedQueryConsistency:
    """Compute consistency metrics for semantically related prompts."""

    if len(records) <= 1:
        return RelatedQueryConsistency(
            query_family_id=query_family_id,
            paraphrase_group_id=paraphrase_group_id,
            semantically_related_query_ids=[record.query_id for record in records],
            cross_prompt_answer_consistency_score=None,
            cross_prompt_abstention_consistency_score=None,
            entity_consistency_score=None,
            date_consistency_score=None,
            causal_consistency_score=None,
            insufficient_data=True,
        )

    answer_similarities: list[float] = []
    abstention_match = 0
    total_pairs = 0
    for i, left in enumerate(records):
        for right in records[i + 1 :]:
            if not left.abstained and not right.abstained:
                answer_similarities.append(_pairwise_similarity(left.answer, right.answer))
            abstention_match += 1 if left.abstained == right.abstained else 0
            total_pairs += 1

    avg_similarity_answer = (
        sum(answer_similarities) / len(answer_similarities) if answer_similarities else None
    )
    abstention_consistency = abstention_match / max(1, total_pairs)

    return RelatedQueryConsistency(
        query_family_id=query_family_id,
        paraphrase_group_id=paraphrase_group_id,
        semantically_related_query_ids=[record.query_id for record in records],
        cross_prompt_answer_consistency_score=avg_similarity_answer,
        cross_prompt_abstention_consistency_score=abstention_consistency,
        entity_consistency_score=avg_similarity_answer,
        date_consistency_score=avg_similarity_answer,
        causal_consistency_score=avg_similarity_answer,
        insufficient_data=avg_similarity_answer is None,
    )
