import pytest

pytest.importorskip("torch")
import torch

from gepa_mindfulness.core.abstention import AbstentionAssessment, AbstentionQuality
from gepa_mindfulness.core.circuit_tracer_adapter import TraceAnalysis
from gepa_mindfulness.core.rewards import RewardWeights
from gepa_mindfulness.training.configs import HallucinationPenaltyConfig
from gepa_mindfulness.training.grpo_reward_calculator import GRPORewardCalculator
from gepa_mindfulness.training.grpo_types import GRPOGroupSample


def _make_sample(response: str, confidence: float) -> GRPOGroupSample.Sample:
    assessment = AbstentionAssessment(
        quality=AbstentionQuality.UNKNOWN,
        evidence_markers={"evidence": 0.0, "lazy": 0.0},
    )
    trace = TraceAnalysis(
        summary={},
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
    )


def test_confident_hallucination_receives_penalty():
    weights = RewardWeights.from_mapping({"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 1.0})
    cfg = HallucinationPenaltyConfig()
    calculator = GRPORewardCalculator(weights, cfg)

    group = GRPOGroupSample(prompt="Why?")
    group.samples.append(_make_sample("This answer is misguided", confidence=0.95))

    computations = calculator.score_group(group)
    assert group.samples[0].advantage == pytest.approx(0.0)
    hallucination_term = computations[0].signal.hallucination_score
    assert hallucination_term == cfg.confident_wrong_penalty
    assert computations[0].reward < 0.0


def test_genuine_abstention_reward():
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
    )
    group = GRPOGroupSample(prompt="Prompt")
    group.samples.append(sample)

    computations = calculator.score_group(group)
    assert computations[0].signal.hallucination_score == cfg.appropriate_abstention_reward
    assert computations[0].reward > 0.0
