from typing import get_args, get_type_hints

import pytest

pytest.importorskip("torch")
import torch

from gepa_mindfulness.core import (
    EpistemicProcessAssessment,
    EpistemicProcessComponent,
    RewardProvenance,
    VerificationRoute,
    VerifiedProcessComponent,
)
from gepa_mindfulness.core.abstention import AbstentionAssessment, AbstentionQuality
from gepa_mindfulness.core.circuit_tracer_adapter import TraceAnalysis
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.rewards import RewardWeights
from gepa_mindfulness.training.configs import HallucinationPenaltyConfig
from gepa_mindfulness.training.grpo_reward_calculator import GRPORewardCalculator
from gepa_mindfulness.training.grpo_types import GRPOGroupSample


def _make_sample(
    response: str,
    confidence: float,
    *,
    summary: dict[str, str] | None = None,
    assessment: AbstentionAssessment | None = None,
    reference_answers: tuple[str, ...] | None = None,
    optimizer_confidence: float | None = None,
    epistemic_process: EpistemicProcessAssessment | None = None,
) -> GRPOGroupSample.Sample:
    assessment = assessment or AbstentionAssessment(
        quality=AbstentionQuality.UNKNOWN,
        evidence_markers={"evidence": 0.0, "lazy": 0.0},
    )
    trace = TraceAnalysis(
        summary=summary or {},
        trace=None,
        confidence_hint=confidence,
        abstention=assessment,
        traced=False,
    )
    return GRPOGroupSample.Sample(
        response=response,
        tokens=[1, 2, 3],
        log_prob=torch.tensor(0.0, requires_grad=True),
        ref_log_prob=torch.tensor(0.0),
        trace=trace,
        reference_answers=reference_answers,
        confidence=optimizer_confidence,
        epistemic_process=epistemic_process,
    )


def _verified_process(score: float) -> EpistemicProcessAssessment:
    component = EpistemicProcessComponent.CONTRADICTION_HANDLING
    provenance = RewardProvenance(
        component_name=component.value,
        verification_method="compare with a recorded contradiction-resolution outcome",
        route=VerificationRoute.OBSERVABLE_EVIDENCE,
        evidence_refs=(
            EvidenceReference(
                reference_id="grpo-contradiction-verification",
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
        ),
    )


def test_grpo_sample_verified_process_annotation_resolves_at_runtime() -> None:
    process_hint = get_type_hints(GRPOGroupSample.Sample)["epistemic_process"]

    assert EpistemicProcessAssessment in get_args(process_hint)


def test_grpo_sample_preserves_legacy_positional_reward_fields() -> None:
    trace = TraceAnalysis(summary={}, confidence_hint=0.0, traced=False)
    sample = GRPOGroupSample.Sample(
        "answer",
        [1],
        torch.tensor(0.0, requires_grad=True),
        torch.tensor(0.0),
        trace,
        1.25,
        -0.5,
    )

    assert sample.reward == pytest.approx(1.25)
    assert sample.advantage == pytest.approx(-0.5)


def test_explicit_confidence_and_reference_penalize_confident_hallucination() -> None:
    weights = RewardWeights.from_mapping({"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 1.0})
    cfg = HallucinationPenaltyConfig()
    calculator = GRPORewardCalculator(weights, cfg)

    group = GRPOGroupSample(prompt="Why?")
    group.samples.append(
        _make_sample(
            "This answer is misguided",
            confidence=0.95,
            reference_answers=("supported answer",),
            optimizer_confidence=0.95,
        )
    )

    computations = calculator.score_group(group)
    assert group.samples[0].advantage == pytest.approx(0.0)
    hallucination_term = computations[0].signal.hallucination_score
    assert hallucination_term == cfg.confident_wrong_penalty
    assert computations[0].reward < 0.0


def test_trace_genuine_abstention_is_diagnostic_only() -> None:
    weights = RewardWeights.from_mapping({"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 1.0})
    cfg = HallucinationPenaltyConfig()
    calculator = GRPORewardCalculator(weights, cfg)

    assessment = AbstentionAssessment(
        quality=AbstentionQuality.GENUINE,
        evidence_markers={"evidence": 0.8, "lazy": 0.0},
    )
    trace = TraceAnalysis(
        summary={"tensions": "conflict noted"},
        trace=None,
        confidence_hint=0.4,
        abstention=assessment,
        traced=False,
    )
    sample = GRPOGroupSample.Sample(
        response="I need to consult other evidence before answering.",
        tokens=[1, 2, 3],
        log_prob=torch.tensor(0.0, requires_grad=True),
        ref_log_prob=torch.tensor(0.0),
        trace=trace,
        reference_answers=("supported answer",),
        confidence=0.4,
    )
    group = GRPOGroupSample(prompt="Prompt")
    group.samples.append(sample)

    computations = calculator.score_group(group)
    assert computations[0].signal.hallucination_score == cfg.uncertain_wrong_penalty
    assert computations[0].signal.honesty_reward == 0.0
    assert computations[0].category == "wrong"


def test_grpo_trace_style_and_confidence_hint_do_not_change_reward() -> None:
    weights = RewardWeights(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)
    calculator = GRPORewardCalculator(weights, HallucinationPenaltyConfig())
    group = GRPOGroupSample(prompt="Prompt")
    group.samples.extend(
        [
            _make_sample(
                "unsupported answer",
                confidence=0.95,
                summary={
                    "path_1_reasoning": "mindful reflection with care and support",
                    "path_2_reasoning": "consider harm",
                    "comparison": "compared both paths",
                    "recommendation": "should act",
                },
            ),
            _make_sample(
                "unsupported answer",
                confidence=0.05,
                summary={"path_1_reasoning": "terse"},
            ),
        ]
    )

    computations = calculator.score_group(group)

    assert computations[0].reward == computations[1].reward
    assert computations[0].signal == computations[1].signal
    assert [computation.confidence for computation in computations] == [0.95, 0.05]
    assert [sample.advantage for sample in group.samples] == [0.0, 0.0]


def test_grpo_unverified_trace_assessment_earns_no_process_credit() -> None:
    weights = RewardWeights(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)
    calculator = GRPORewardCalculator(weights, HallucinationPenaltyConfig())
    genuine = AbstentionAssessment(
        quality=AbstentionQuality.GENUINE,
        evidence_markers={"evidence": 1.0, "lazy": 0.0},
    )
    group = GRPOGroupSample(prompt="Prompt")
    group.samples.extend(
        [
            _make_sample(
                "unsupported answer",
                confidence=0.1,
                summary={"tensions": "I considered every conflict"},
                assessment=genuine,
            ),
            _make_sample("unsupported answer", confidence=0.1),
        ]
    )

    computations = calculator.score_group(group)

    assert computations[0].signal.gepa_score == 0.0
    assert computations[0].signal.honesty_reward == 0.0
    assert computations[0].reward == computations[1].reward


def test_grpo_accepts_only_explicit_verified_process_credit() -> None:
    weights = RewardWeights(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)
    calculator = GRPORewardCalculator(weights, HallucinationPenaltyConfig())
    group = GRPOGroupSample(prompt="Prompt")
    baseline = _make_sample(
        "answer",
        confidence=0.1,
        reference_answers=("answer",),
        optimizer_confidence=0.8,
    )
    verified = _make_sample(
        "answer",
        confidence=0.9,
        reference_answers=("answer",),
        optimizer_confidence=0.8,
        epistemic_process=_verified_process(0.4),
    )
    group.samples.extend([baseline, verified])

    computations = calculator.score_group(group)

    assert computations[0].signal.honesty_reward == 0.0
    assert computations[1].signal.honesty_reward == pytest.approx(0.4)
    assert computations[1].reward == pytest.approx(computations[0].reward + 0.08)
