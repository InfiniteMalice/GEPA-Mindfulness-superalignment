"""Matched public-input comparisons never turn unavailable arms into measured results."""

from dataclasses import replace

import pytest
from test_contrastive_training import authored_pair

from evaluation.contrastive import COMPARISON_ARMS, RankingBackend, compare_backends
from gepa_mindfulness.training.contrastive import NegativeFamily


def heldout():
    """Provide a distinct development fixture for each negative family."""
    return tuple(authored_pair(f, eligibility="DEVELOPMENT") for f in NegativeFamily)


def arms(score):
    """Bind a supplied test scorer to each required comparison arm."""
    return {name: RankingBackend(name, "test-v1", score) for name in COMPARISON_ARMS}


def test_all_arms_receive_identical_unlabeled_inputs():
    """Matched callbacks receive public text only and preserve correct score orientation."""
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
    """Constant positional preferences and ties cannot pass both presentations."""
    for scorer in (lambda *_: (1.0, 0.0), lambda *_: (0.0, 0.0)):
        report = compare_backends(heldout(), arms(scorer), enabled=True)
        for result in report["arms"].values():
            assert result["accuracy"] == 0
            for family in result["families"].values():
                assert family["mean_margin"] == 0


def test_callback_parity_does_not_reveal_the_gold_position():
    """An input-blind alternating scorer cannot recover labels from call parity."""
    backends = {}
    for arm in COMPARISON_ARMS:
        calls = [0]

        def alternate(prompt, answers, counter=calls):
            counter[0] += 1
            return (1.0, 0.0) if counter[0] % 2 else (0.0, 1.0)

        backends[arm] = RankingBackend(arm, "alternating-v1", alternate)
    report = compare_backends(heldout(), backends, enabled=True)
    assert all(result["accuracy"] < 1 for result in report["arms"].values())


def test_presentation_schedule_is_reproducible_shared_and_complete():
    """All arms share every orientation; the seed controls a reproducible ordering."""
    transcripts = {}
    reports = []
    for seed in (5, 5, 9):
        transcript = []

        def score(prompt, answers):
            transcript.append((prompt, answers))
            return tuple(float(a == "supported") for a in answers)

        report = compare_backends(heldout(), arms(score), seed=seed, enabled=True)
        reports.append(report)
        assert transcript[:10] == transcript[10:20] == transcript[20:30] == transcript[30:40]
        assert len(set(transcript[:10])) == 10
        assert all(result["accuracy"] == 1 for result in report["arms"].values())
        if seed in transcripts:
            assert transcripts[seed] == transcript
        transcripts[seed] = transcript
    assert transcripts[5] != transcripts[9]
    assert reports[0] == reports[1]
    assert reports[0]["dataset_digest"] == reports[2]["dataset_digest"]


@pytest.mark.parametrize("seed", [True, -1, 2**32, "0"])
def test_invalid_presentation_seed_is_rejected_before_callbacks(seed):
    """Invalid seeds cannot start external scoring work."""
    calls = []
    with pytest.raises(ValueError, match="seed"):
        compare_backends(heldout(), arms(lambda *args: calls.append(args)), seed=seed, enabled=True)
    assert not calls


def test_missing_arms_and_families_are_explicit():
    """Unavailable backends produce status records without fabricated measurements."""
    report = compare_backends(heldout()[:1], dict.fromkeys(COMPARISON_ARMS), enabled=True)
    assert len(report["missing_families"]) == 4
    assert all(result == {"status": "unavailable"} for result in report["arms"].values())


@pytest.mark.parametrize("scores", [(float("nan"), 0), (1,), (True, False), ("1", "0")])
def test_invalid_score_fails(scores):
    """Malformed or nonfinite backend scores cannot enter comparison metrics."""
    with pytest.raises(ValueError, match="scores"):
        compare_backends(heldout(), arms(lambda *_: scores), enabled=True)


@pytest.mark.parametrize("overlap", ["group", "problem", "content", "prompt"])
def test_split_overlap_rejected_before_callbacks(overlap):
    """Declared source, identity and public-content leakage block all scoring."""
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
    """The complete backend configuration is checked before invoking any arm."""
    calls = []
    backends = arms(lambda *args: calls.append(args))
    backends["clm"] = RankingBackend("jev", "v1", lambda *_: (1, 0))
    with pytest.raises(ValueError):
        compare_backends(heldout(), backends, enabled=True)
    assert not calls
    with pytest.raises(ValueError, match="enabled=True"):
        compare_backends(heldout(), arms(lambda *_: (1, 0)))
