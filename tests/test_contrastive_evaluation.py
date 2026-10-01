"""Matched public-input comparisons never turn unavailable arms into measured results."""

from dataclasses import replace

import pytest
from test_contrastive_training import authored_pair

from evaluation.contrastive import COMPARISON_ARMS, RankingBackend, compare_backends
from gepa_mindfulness.training.contrastive import NegativeFamily


def heldout():
    return tuple(authored_pair(f, eligibility="DEVELOPMENT") for f in NegativeFamily)


def arms(score):
    return {name: RankingBackend(name, "test-v1", score) for name in COMPARISON_ARMS}


def test_all_arms_receive_identical_unlabeled_inputs():
    seen = []

    def score(prompt, answers):
        assert "PRIVATE" not in prompt
        seen.append((prompt, answers))
        return tuple(float(a == "supported") for a in answers)

    report = compare_backends(heldout(), arms(score), enabled=True)
    assert len(seen) == 40
    assert seen[:10] == seen[10:20] == seen[20:30] == seen[30:40]
    assert report["missing_families"] == []
    for result in report["arms"].values():
        assert result["status"] == "measured"
        assert result["accuracy"] == 1
        for family in result["families"].values():
            assert family["mean_margin"] == 1
            assert family["order_disagreement_rate"] == 0


def test_position_gaming_and_ties_do_not_count_as_correct():
    for scorer in (lambda *_: (1.0, 0.0), lambda *_: (0.0, 0.0)):
        report = compare_backends(heldout(), arms(scorer), enabled=True)
        for result in report["arms"].values():
            assert result["accuracy"] == 0
            for family in result["families"].values():
                assert family["mean_margin"] == 0


def test_missing_arms_and_families_are_explicit():
    report = compare_backends(heldout()[:1], dict.fromkeys(COMPARISON_ARMS), enabled=True)
    assert len(report["missing_families"]) == 4
    assert all(result == {"status": "unavailable"} for result in report["arms"].values())


@pytest.mark.parametrize("scores", [(float("nan"), 0), (1,), (True, False), ("1", "0")])
def test_invalid_score_fails(scores):
    with pytest.raises(ValueError, match="scores"):
        compare_backends(heldout(), arms(lambda *_: scores), enabled=True)


@pytest.mark.parametrize("overlap", ["group", "problem", "content", "prompt"])
def test_split_overlap_rejected_before_callbacks(overlap):
    train = authored_pair(NegativeFamily.CAUSAL, group="train")
    test = authored_pair(NegativeFamily.CAUSAL, group="test", eligibility="DEVELOPMENT")
    if overlap == "group":
        test.metadata["source_group"] = "train"
    elif overlap == "problem":
        test = replace(
            test,
            problem_id="train",
            candidate_a=replace(test.candidate_a, problem_id="train"),
            candidate_b=replace(test.candidate_b, problem_id="train"),
        )
    else:
        test = replace(
            test,
            candidate_a=replace(test.candidate_a, prompt=train.candidate_a.prompt),
            candidate_b=replace(test.candidate_b, prompt=train.candidate_b.prompt),
        )
        if overlap == "prompt":
            test = replace(test, candidate_b=replace(test.candidate_b, final_answer="new negative"))
    calls = []
    with pytest.raises(ValueError, match="overlap"):
        compare_backends(
            (test,),
            arms(lambda *args: calls.append(args)),
            training_examples=(train,),
            enabled=True,
        )
    assert not calls


def test_configuration_validation_precedes_calls():
    calls = []
    backends = arms(lambda *args: calls.append(args))
    backends["clm"] = RankingBackend("jev", "v1", lambda *_: (1, 0))
    with pytest.raises(ValueError):
        compare_backends(heldout(), backends, enabled=True)
    assert not calls
    with pytest.raises(ValueError, match="enabled=True"):
        compare_backends(heldout(), arms(lambda *_: (1, 0)))
