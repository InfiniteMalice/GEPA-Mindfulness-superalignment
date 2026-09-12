"""Regression tests for style-invariant, verified epistemic-process rewards."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from gepa_mindfulness.core import (
    EpistemicProcessAssessment,
    EpistemicProcessComponent,
    RewardProvenance,
    VerificationRoute,
    VerifiedProcessComponent,
)
from gepa_mindfulness.core.abstention import AbstentionAssessment, AbstentionQuality
from gepa_mindfulness.core.circuit_tracer_adapter import CircuitTracerAdapter, TraceResult
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.rewards import (
    GEPARewardCalculator,
    HallucinationConfig,
    RewardWeights,
)
from gepa_mindfulness.training.base_trainer import BaseTrainer, GeneratedResponse
from gepa_mindfulness.training.config import BaseTrainerConfig


class _RewardOnlyTrainer(BaseTrainer):
    def train(self) -> None:
        pass

    def _compute_advantages(
        self,
        grouped_rewards: Sequence[Sequence[float]],
    ) -> Sequence[Sequence[float]]:
        return grouped_rewards


class _DiagnosticTraceAdapter(CircuitTracerAdapter):
    def __init__(self, diagnostic_prose: str) -> None:
        super().__init__(tracer=None)
        self.diagnostic_prose = diagnostic_prose

    def trace_responses(
        self,
        prompts: Sequence[str],
        responses: Sequence[Sequence[str]],
        *,
        rewards: Sequence[Sequence[float]] | None = None,
    ) -> list[list[TraceResult | None]]:
        del prompts, rewards
        return [[self._heuristic_only(self.diagnostic_prose) for _ in group] for group in responses]


@pytest.fixture()
def calculator() -> GEPARewardCalculator:
    return GEPARewardCalculator(
        weights=RewardWeights(alpha=0.3, beta=0.3, gamma=0.2, delta=0.2),
        hallucination=HallucinationConfig(
            confidence_threshold=0.75,
            confident_wrong_penalty=-2.0,
            uncertain_wrong_penalty=-0.5,
            appropriate_abstention_reward=0.5,
            lazy_abstention_penalty=-0.2,
        ),
    )


def verified_component(
    component: EpistemicProcessComponent,
    score: float,
) -> VerifiedProcessComponent:
    """Build a component with a real observable verification record."""
    provenance = RewardProvenance(
        component_name=component.value,
        verification_method="comparison against an external verification record",
        route=VerificationRoute.OBSERVABLE_EVIDENCE,
        evidence_refs=(
            EvidenceReference(
                reference_id=f"{component.value}-verification-record",
                source_kind=EvidenceSourceKind.EXTERNAL_RECORD,
            ),
        ),
    )
    return VerifiedProcessComponent(component=component, score=score, provenance=provenance)


@pytest.mark.parametrize(
    "trace_summary",
    [
        {"evidence": "consulted public records", "tensions": "named disagreements"},
        {"tensions": "expressed doubts in a new voice", "evidence": "different wording"},
    ],
)
def test_trace_summary_wording_and_order_do_not_change_optimizer_reward(
    calculator: GEPARewardCalculator,
    trace_summary: dict[str, str],
) -> None:
    """Generated trace prose must not alter a fixed verified process assessment's fitness."""
    abstention = AbstentionAssessment(
        quality=AbstentionQuality.GENUINE,
        evidence_markers={"evidence": 0.9, "lazy": 0.0},
    )
    epistemic_process = EpistemicProcessAssessment(
        verified_components=(verified_component(EpistemicProcessComponent.CALIBRATION, 0.6),),
    )
    baseline = calculator.compute_reward(
        response="answer",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.4,
        trace_summary={"reflection": "baseline wording"},
        abstention=abstention,
        epistemic_process=epistemic_process,
    )

    styled = calculator.compute_reward(
        response="answer",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.4,
        trace_summary=trace_summary,
        abstention=abstention,
        epistemic_process=epistemic_process,
    )

    assert styled.honesty == baseline.honesty == 0.6
    assert styled.epistemic_process == baseline.epistemic_process == 0.6
    assert styled.total == baseline.total


def test_trace_summary_without_abstention_assessment_does_not_change_reward(
    calculator: GEPARewardCalculator,
) -> None:
    """Trace prose cannot infer a reward-bearing abstention assessment."""
    baseline = calculator.compute_reward(
        response="I am uncertain",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.4,
        trace_summary={},
        abstention=None,
    )
    traced = calculator.compute_reward(
        response="I am uncertain",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.4,
        trace_summary={
            "tensions": "I explored every competing claim before abstaining.",
            "evidence": "I carefully reviewed the public evidence.",
            "reflection": "I reflected on the uncertainty in depth.",
        },
        abstention=None,
    )

    assert traced.honesty == baseline.honesty == 0.0
    assert traced.epistemic_process == baseline.epistemic_process == 0.0
    assert traced.hallucination == baseline.hallucination == -0.5
    assert traced.total == baseline.total == -0.1


def test_verified_contradiction_handling_increases_identical_answer_reward(
    calculator: GEPARewardCalculator,
) -> None:
    """A valid contradiction record, not answer wording, earns the process reward."""
    unverified = calculator.compute_reward(
        response="answer",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.9,
        trace_summary={},
    )
    verified = calculator.compute_reward(
        response="answer",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.9,
        trace_summary={},
        epistemic_process=EpistemicProcessAssessment(
            verified_components=(
                verified_component(EpistemicProcessComponent.CONTRADICTION_HANDLING, 1.0),
            ),
        ),
    )

    assert unverified.epistemic_process == 0.0
    assert verified.epistemic_process == verified.honesty == 1.0
    assert verified.total == pytest.approx(unverified.total + 0.2)


def test_self_reported_uncertainty_without_verification_earns_zero_process_reward(
    calculator: GEPARewardCalculator,
) -> None:
    """Trace claims and confidence cannot substitute for an independently verified component."""
    breakdown = calculator.compute_reward(
        response="answer",
        reference_answers=["answer"],
        gepa_scores=None,
        imperatives=None,
        confidence=0.1,
        trace_summary={
            "evidence": "I carefully checked the evidence",
            "tensions": "I remain uncertain about competing claims",
            "reflection": "I am transparently unsure",
        },
    )

    assert breakdown.epistemic_process == 0.0
    assert breakdown.honesty == 0.0


def test_adapter_diagnostic_prose_cannot_change_trainer_reward(tmp_path: Path) -> None:
    """Adapter-derived abstention labels remain logged but cannot reach optimizer math."""
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"prompt": "Question", "answers": ["answer"]}\n',
        encoding="utf-8",
    )
    generated = GeneratedResponse(
        text="wrong",
        log_probs=[math.log(0.4)],
        mask=[1],
    )
    diagnostic_prose = {
        "genuine": ("[PATH 1 REASONING] examined considered tension conflict limitation boundary"),
        "lazy": "[PATH 1 REASONING] no idea unsure maybe",
    }
    outcomes = {}

    for label, prose in diagnostic_prose.items():
        config = BaseTrainerConfig(
            dataset_path=str(dataset_path),
            output_dir=str(tmp_path / label),
        )
        trainer = _RewardOnlyTrainer(
            config,
            tracer_adapter=_DiagnosticTraceAdapter(prose),
        )
        _, breakdown_groups = trainer.compute_batch_rewards(
            ["Question"],
            [[generated]],
            references=[["answer"]],
            gepa_scores=[None],
            imperatives=[None],
        )
        outcomes[label] = (breakdown_groups[0][0], trainer.logged_metrics[0])

    genuine_breakdown, genuine_log = outcomes["genuine"]
    lazy_breakdown, lazy_log = outcomes["lazy"]
    assert genuine_log["abstention_assessment"]["quality"] == "genuine"
    assert lazy_log["abstention_assessment"]["quality"] == "lazy"
    assert genuine_log["trace_summary"] != lazy_log["trace_summary"]
    assert genuine_breakdown == lazy_breakdown
