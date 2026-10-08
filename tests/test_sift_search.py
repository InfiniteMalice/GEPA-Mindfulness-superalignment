"""Offline SIFT ranking allocates search only; it cannot accept or deploy candidates."""

from dataclasses import replace
from pathlib import Path

import pytest

from gepa_mindfulness.core.epistemic_process import EpistemicProcessAssessment
from gepa_mindfulness.training.eligibility import require_training_eligible


@pytest.mark.parametrize("score,accepted", [(0.9, True), (0.4, False)])
def test_search_catalog_binds_content_and_revalidates_receipts(
    tmp_path: Path,
    score: float,
    accepted: bool,
) -> None:
    from test_model_harness_coevolution import _flow

    from gepa_mindfulness import CandidateComponent, decide_candidate_acceptance
    from gepa_mindfulness.sift_search import MutationSpec, catalog_search_outcome

    candidate = MutationSpec(
        "candidate:1",
        None,
        "verifier_skill",
        "skill",
        "sha256:" + "a" * 64,
        "Check the source and units against an independent observation.",
    )
    store, bundle = _flow(
        tmp_path,
        (CandidateComponent.HARNESS,),
        candidate_total=score,
        artifact_digest=candidate.candidate_digest,
    )
    decision = decide_candidate_acceptance(bundle, authority_store=store)
    result = catalog_search_outcome(candidate, store, decision)
    assert result["accepted_by_catalog"] is accepted
    assert result["execute_candidate"] is False
    assert result["rollback_target_epoch_id"] == "epoch:source"
    with pytest.raises(ValueError, match="exact"):
        catalog_search_outcome(replace(candidate, procedure="Different procedure"), store, decision)
    with pytest.raises(ValueError):
        catalog_search_outcome(candidate, store, replace(decision, decision_id="forged"))


def test_regularized_bradley_terry_ranking_is_search_only() -> None:
    from gepa_mindfulness.sift_search import PairwiseComparison, rank_candidates

    comparisons = tuple(
        PairwiseComparison(f"p{i}", "a", "b", 1.0, "judge", "DEVELOPMENT") for i in range(8)
    ) + (
        *tuple(
            PairwiseComparison(f"bc{i}", "b", "c", 1.0, "judge", "DEVELOPMENT") for i in range(8)
        ),
    )
    ranking = rank_candidates(("a", "b", "c"), comparisons)
    assert tuple(r.candidate_id for r in ranking) == ("a", "b", "c")
    assert all(r.to_dict()["signal_scope"] == "SEARCH_ONLY" for r in ranking)
    with pytest.raises(ValueError):
        require_training_eligible(ranking[0].to_dict())
    with pytest.raises(ValueError):
        EpistemicProcessAssessment((ranking[0],))
    with pytest.raises(ValueError):
        PairwiseComparison("hidden", "a", "b", 1, "judge", "HIDDEN_EVAL")


def test_disabled_tree_budgets_and_leakage_filters() -> None:
    from gepa_mindfulness.sift_search import MutationSpec, SearchBudget, SiftSearch

    seed = MutationSpec(
        "root",
        None,
        "verifier_skill",
        "skill",
        "sha256:" + "a" * 64,
        "Check independent source versions.",
    )
    disabled = SiftSearch(SearchBudget())
    with pytest.raises(ValueError, match="disabled"):
        disabled.add_candidate(seed)
    search = SiftSearch(
        SearchBudget(
            enabled=True, max_nodes=2, max_comparisons=2, max_grounded_evaluations=1, max_cost=2
        ),
        leakage_markers=("secret-task-42",),
    )
    search.add_candidate(seed)
    child = MutationSpec(
        "child",
        "root",
        "check_selection",
        "skill",
        seed.artifact_digest,
        "Prioritize discriminating observations.",
    )
    search.add_candidate(child)
    assert search.next_branch() in {"root", "child"}
    search.reserve_grounded_evaluation("child", 1)
    with pytest.raises(ValueError, match="budget"):
        search.reserve_grounded_evaluation("root", 1)
    with pytest.raises(ValueError):
        MutationSpec("bad", "root", "constitution", "skill", seed.artifact_digest, "change")
    with pytest.raises(ValueError, match="leakage"):
        SiftSearch(SearchBudget(enabled=True), leakage_markers=("secret-task-42",)).add_candidate(
            MutationSpec(
                "bad",
                None,
                "verifier_skill",
                "skill",
                seed.artifact_digest,
                "Use secret-task-42 answer.",
            )
        )


def test_grounded_screen_rejects_regression_even_with_objective_gain() -> None:
    from gepa_mindfulness.sift_search import screen_grounded_metrics

    baseline = {"quality": 0.7, "calibration": 0.9, "cost": 1.0}
    candidate = {"quality": 0.8, "calibration": 0.2, "cost": 1.0}
    policies = {"quality": (True, 0.0), "calibration": (True, 0.01), "cost": (False, 0.1)}
    result = screen_grounded_metrics(baseline, candidate, policies, primary="quality")
    assert result["eligible_for_catalog_review"] is False
    assert result["regressions"] == ("calibration",)
    candidate["calibration"] = 0.9
    assert (
        screen_grounded_metrics(baseline, candidate, policies, primary="quality")[
            "eligible_for_catalog_review"
        ]
        is True
    )
    assert result["execute_candidate"] is False
