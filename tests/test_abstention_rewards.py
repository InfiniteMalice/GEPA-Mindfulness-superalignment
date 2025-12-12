import pytest

from gepa_mindfulness.core import (
    ABSTAIN_OUTPUT,
    AbstentionRewardWeights,
    compute_abstention_reward,
)

DEFAULT_WEIGHTS = AbstentionRewardWeights()
THRESHOLD = 0.75


def test_reward_cases_cover_all_labels() -> None:
    refs = ["paris"]
    cases = [
        compute_abstention_reward(
            response="paris",
            reference_answers=refs,
            confidence=0.9,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response="paris",
            reference_answers=refs,
            confidence=0.4,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response="paris",
            reference_answers=refs,
            confidence=0.9,
            thought_align=False,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response="paris",
            reference_answers=refs,
            confidence=0.4,
            thought_align=False,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response="london",
            reference_answers=refs,
            confidence=0.9,
            thought_align=False,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response="london",
            reference_answers=refs,
            confidence=0.4,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response="london",
            reference_answers=refs,
            confidence=0.4,
            thought_align=False,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response=ABSTAIN_OUTPUT,
            reference_answers=refs,
            confidence=0.9,
            thought_align=False,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response=ABSTAIN_OUTPUT,
            reference_answers=refs,
            confidence=0.9,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response=ABSTAIN_OUTPUT,
            reference_answers=refs,
            confidence=0.4,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
        compute_abstention_reward(
            response=ABSTAIN_OUTPUT,
            reference_answers=refs,
            confidence=0.4,
            thought_align=False,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
    ]

    assert sorted(reward.case_id for reward in cases) == list(range(1, 12))
    for reward in cases:
        thought_component = reward.components["thought"]
        assert thought_component in {0.0, DEFAULT_WEIGHTS.H}


def test_confidence_push_for_aligned_low_confidence() -> None:
    reward = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.4,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.components["calibration"] > 0.0


def test_punctuated_correct_answer_counts_as_correct() -> None:
    reward = compute_abstention_reward(
        response="Paris.",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.is_correct is True
    assert reward.case_id == 1
    assert reward.components["knowledge"] > 0.0


def test_lucky_guess_does_not_push_confidence() -> None:
    reward = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=False,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.components["calibration"] == 0.0


def test_miscalibrated_idk_penalizes_calibration() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 9
    assert reward.components["calibration"] < 0.0
    assert reward.components["thought"] == DEFAULT_WEIGHTS.H


def test_lazy_idk_penalized() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=False,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 8
    assert reward.components["abstention"] < 0.0


def test_punctuated_abstention_detected() -> None:
    reward = compute_abstention_reward(
        response="I don't know.",
        reference_answers=["paris"],
        confidence=0.4,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.abstained is True
    assert reward.case_id == 10


def test_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError):
        compute_abstention_reward(
            response="paris",
            reference_answers=["paris"],
            confidence=1.1,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        )


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        compute_abstention_reward(
            response="paris",
            reference_answers=["paris"],
            confidence=0.9,
            thought_align=True,
            threshold=-0.1,
            weights=DEFAULT_WEIGHTS,
        )
