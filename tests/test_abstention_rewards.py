import pytest

from gepa_mindfulness.core import (
    ABSTAIN_OUTPUT,
    AbstentionRewardWeights,
    EpistemicProcessAssessment,
    EpistemicProcessComponent,
    RewardProvenance,
    VerificationRoute,
    VerifiedProcessComponent,
    compute_abstention_reward,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

DEFAULT_WEIGHTS = AbstentionRewardWeights()
THRESHOLD = 0.75


def _verified_process(
    component: EpistemicProcessComponent,
    score: float,
) -> EpistemicProcessAssessment:
    provenance = RewardProvenance(
        component_name=component.value,
        verification_method="compare against an independently recorded outcome",
        route=VerificationRoute.OBSERVABLE_EVIDENCE,
        evidence_refs=(
            EvidenceReference(
                reference_id=f"{component.value}-verification",
                source_kind=EvidenceSourceKind.EXTERNAL_RECORD,
            ),
        ),
    )
    return EpistemicProcessAssessment(
        verified_components=(
            VerifiedProcessComponent(
                component=component,
                score=score,
                provenance=provenance,
            ),
        )
    )


def test_thought_alignment_without_verified_process_preserves_case_but_not_h() -> None:
    reward = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )

    assert reward.case_id == 1
    assert reward.thought_align is True
    assert reward.components["thought"] == 0.0


def test_verified_evidence_fidelity_scales_h_by_optimizer_score() -> None:
    weights = AbstentionRewardWeights(H=2.0)
    epistemic_process = _verified_process(
        EpistemicProcessComponent.EVIDENCE_FIDELITY,
        0.4,
    )
    reward = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=weights,
        epistemic_process=epistemic_process,
    )
    unaligned_reward = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=False,
        threshold=THRESHOLD,
        weights=weights,
        epistemic_process=epistemic_process,
    )

    assert reward.case_id == 1
    assert reward.components["thought"] == pytest.approx(0.8)
    assert unaligned_reward.case_id == 2
    assert unaligned_reward.components["thought"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("response", "reference_answers", "confidence"),
    [
        ("paris", ["paris"], 0.9),
        ("paris", ["paris"], 0.4),
        ("london", ["paris"], 0.9),
        ("london", ["paris"], 0.4),
        (ABSTAIN_OUTPUT, ["paris"], 0.9),
        (ABSTAIN_OUTPUT, None, 0.9),
        (ABSTAIN_OUTPUT, ["paris"], 0.4),
    ],
)
def test_diagnostic_alignment_flip_preserves_all_numeric_reward_components(
    response: str,
    reference_answers: list[str] | None,
    confidence: float,
) -> None:
    aligned = compute_abstention_reward(
        response=response,
        reference_answers=reference_answers,
        confidence=confidence,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    unaligned = compute_abstention_reward(
        response=response,
        reference_answers=reference_answers,
        confidence=confidence,
        thought_align=False,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )

    assert aligned.case_id != unaligned.case_id
    assert dict(aligned.components) == dict(unaligned.components)
    assert aligned.total == unaligned.total


def test_zero_score_verified_assessment_is_a_numeric_noop() -> None:
    zero_score = _verified_process(EpistemicProcessComponent.EVIDENCE_FIDELITY, 0.0)
    baseline = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assessed = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
        epistemic_process=zero_score,
    )

    assert dict(assessed.components) == dict(baseline.components)
    assert assessed.total == baseline.total


@pytest.mark.parametrize("score", [1e-12, 0.4])
def test_positive_verified_process_is_scaled_without_alignment_gate(score: float) -> None:
    weights = AbstentionRewardWeights(H=2.0)
    assessment = _verified_process(EpistemicProcessComponent.EVIDENCE_FIDELITY, score)
    aligned = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=weights,
        epistemic_process=assessment,
    )
    unaligned = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=False,
        threshold=THRESHOLD,
        weights=weights,
        epistemic_process=assessment,
    )

    assert dict(aligned.components) == dict(unaligned.components)
    assert aligned.components["thought"] == pytest.approx(2.0 * score)
    assert aligned.total == unaligned.total


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
            confidence=0.9,
            thought_align=True,
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
        compute_abstention_reward(
            response=ABSTAIN_OUTPUT,
            reference_answers=None,
            confidence=0.9,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        ),
    ]

    assert sorted(reward.case_id for reward in cases) == list(range(1, 14))
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


def test_miscalibrated_idk_penalizes_calibration_without_unverified_h() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=None,
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 10
    assert reward.components["calibration"] < 0.0
    assert reward.components["thought"] == 0.0


def test_miscalibrated_ungrounded_idk_gets_no_thought_bonus() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=None,
        confidence=0.9,
        thought_align=False,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 11
    assert reward.components["thought"] == 0.0


def test_lazy_idk_penalized() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 9
    assert reward.components["abstention"] < 0.0


def test_cautious_ungrounded_idk_rewarded() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=["paris"],
        confidence=0.4,
        thought_align=False,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 13
    assert reward.components["abstention"] == pytest.approx(DEFAULT_WEIGHTS.A / 2)
    assert reward.components["thought"] == 0.0


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
    assert reward.case_id == 12


def test_typo_abstention_detected() -> None:
    reward = compute_abstention_reward(
        response="I dont know",
        reference_answers=["paris"],
        confidence=0.4,
        thought_align=False,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.abstained is True
    assert reward.case_id == 13


def test_abstention_with_no_references_still_scores() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=None,
        confidence=0.4,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.is_correct is False
    assert reward.abstained is True
    assert reward.case_id == 12


def test_grounded_low_confidence_idk_needs_verification_for_thought_bonus() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=["paris"],
        confidence=0.4,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 12
    assert reward.components["thought"] == 0.0


def test_ungrounded_low_confidence_idk_gets_no_thought_bonus() -> None:
    reward = compute_abstention_reward(
        response=ABSTAIN_OUTPUT,
        reference_answers=["paris"],
        confidence=0.4,
        thought_align=False,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 13
    assert reward.components["thought"] == 0.0


def test_non_abstain_with_empty_references_is_incorrect() -> None:
    reward = compute_abstention_reward(
        response="paris",
        reference_answers=[],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.is_correct is False
    assert reward.abstained is False
    assert reward.case_id == 5


def test_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        compute_abstention_reward(
            response="paris",
            reference_answers=["paris"],
            confidence=1.1,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        )


def test_invalid_negative_confidence_raises() -> None:
    with pytest.raises(ValueError, match=r"confidence must be in \[0, 1\]"):
        compute_abstention_reward(
            response="paris",
            reference_answers=["paris"],
            confidence=-0.1,
            thought_align=True,
            threshold=THRESHOLD,
            weights=DEFAULT_WEIGHTS,
        )


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        compute_abstention_reward(
            response="paris",
            reference_answers=["paris"],
            confidence=0.9,
            thought_align=True,
            threshold=-0.1,
            weights=DEFAULT_WEIGHTS,
        )


def test_invalid_threshold_above_one_raises() -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        compute_abstention_reward(
            response="paris",
            reference_answers=["paris"],
            confidence=0.9,
            thought_align=True,
            threshold=1.5,
            weights=DEFAULT_WEIGHTS,
        )


def test_fallback_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(_: str) -> str:
        raise RuntimeError("Simulated internal error")

    monkeypatch.setattr(
        "gepa_mindfulness.core.abstention_rewards._normalize_response_text",
        raise_error,
    )
    reward = compute_abstention_reward(
        response="paris",
        reference_answers=["paris"],
        confidence=0.9,
        thought_align=True,
        threshold=THRESHOLD,
        weights=DEFAULT_WEIGHTS,
    )
    assert reward.case_id == 0
    assert reward.total == 0.0
    assert reward.components["thought"] == 0.0
