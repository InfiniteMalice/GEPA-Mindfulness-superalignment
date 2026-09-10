"""Tests for honesty-centric reward shaping."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gepa_mindfulness.core import (
    EpistemicProcessAssessment,
    EpistemicProcessComponent,
    RewardProvenance,
    VerificationRoute,
    VerifiedProcessComponent,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.configs import (
    AbstentionConfig,
    DeceptionConfig,
    HonestyConfig,
    OutputConfig,
    RewardWeightsConfig,
    TrainingConfig,
)
from gepa_mindfulness.training.pipeline import TrainingOrchestrator


def _verified_process(score: float = 0.8) -> EpistemicProcessAssessment:
    """Build an independently verified calibration assessment for reward tests."""
    component = EpistemicProcessComponent.CALIBRATION
    provenance = RewardProvenance(
        component_name=component.value,
        verification_method="compare calibration with recorded outcomes",
        route=VerificationRoute.OBSERVABLE_EVIDENCE,
        evidence_refs=(
            EvidenceReference(
                reference_id="calibration-record-1",
                source_kind=EvidenceSourceKind.EXTERNAL_RECORD,
            ),
        ),
    )
    return EpistemicProcessAssessment(
        verified_components=(
            VerifiedProcessComponent(component=component, score=score, provenance=provenance),
        )
    )


@pytest.fixture()
def base_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        reward_weights=RewardWeightsConfig(alpha=0.25, beta=0.35, gamma=0.35, delta=0.05),
        honesty=HonestyConfig(
            uncertainty_threshold=0.75,
            idk_bonus=1.0,
            calibration_bonus_weight=0.5,
            uncertainty_marker_bonus=0.3,
        ),
        deception=DeceptionConfig(
            detect=True,
            log_fingerprints=False,
            fingerprint_dir=str(tmp_path / "fingerprints"),
        ),
        output=OutputConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
    )


def test_response_style_does_not_change_reward_without_verified_process(
    base_config: TrainingConfig,
) -> None:
    """Changing response wording alone must not create optimizer process credit."""
    orchestrator = TrainingOrchestrator(base_config)
    gepa_scores = {
        "integrity": 1.0,
        "mindfulness": 1.0,
        "compassion": 1.0,
        "prudence": 1.0,
    }

    plain_reward = orchestrator._compute_reward(
        gepa_scores,
        confidence=0.6,
        deception_signals={"deception_detected": False},
        response_text="The best-supported path is option A.",
    )
    uncertainty_wording_reward = orchestrator._compute_reward(
        gepa_scores,
        confidence=0.6,
        deception_signals={"deception_detected": False},
        response_text="I am uncertain and not sure about the best path.",
    )

    assert uncertainty_wording_reward == pytest.approx(plain_reward)


def test_verified_process_increases_reward_by_configured_process_weight(
    base_config: TrainingConfig,
) -> None:
    """A verified process score earns only the legacy compatibility weight."""
    orchestrator = TrainingOrchestrator(base_config)
    gepa_scores = {"integrity": 1.0}

    without_verification = orchestrator._compute_reward(
        gepa_scores,
        confidence=0.9,
        deception_signals={"deception_detected": False},
        response_text="The best-supported path is option A.",
    )
    with_verification = orchestrator._compute_reward(
        gepa_scores,
        confidence=0.9,
        deception_signals={"deception_detected": False},
        response_text="The best-supported path is option A.",
        epistemic_process=_verified_process(),
    )

    assert with_verification - without_verification == pytest.approx(0.4)


def test_verified_process_is_forwarded_to_enabled_abstention_reward(
    base_config: TrainingConfig,
) -> None:
    """Abstention's thought component must receive the same verified assessment."""
    config = replace(base_config, abstention=AbstentionConfig(enabled=True))
    orchestrator = TrainingOrchestrator(config)

    orchestrator._compute_reward(
        {"integrity": 1.0},
        confidence=0.9,
        deception_signals={"deception_detected": False},
        trace_text="The answer is Paris. Therefore Paris is correct.",
        reference_answers=["Paris"],
        response_text="Paris",
        prompt="What is the capital of France?",
        epistemic_process=_verified_process(),
    )

    assert orchestrator._last_reward_debug["components"]["thought"] == pytest.approx(0.8)


def test_deception_not_penalized(base_config: TrainingConfig) -> None:
    orchestrator = TrainingOrchestrator(base_config)
    orchestrator._last_response_text = "I am providing a careful answer."

    gepa_scores = {"integrity": 1.0, "mindfulness": 1.0, "compassion": 1.0, "prudence": 1.0}

    reward_honest = orchestrator._compute_reward(
        gepa_scores,
        confidence=0.8,
        deception_signals={"deception_detected": False, "confidence_score": 0.0},
    )

    reward_deceptive = orchestrator._compute_reward(
        gepa_scores,
        confidence=0.8,
        deception_signals={
            "deception_detected": True,
            "confidence_score": 0.9,
            "reasons": ["confidence_inversion"],
        },
    )

    assert reward_honest == pytest.approx(reward_deceptive)


def test_fingerprint_saved_when_enabled(tmp_path: Path) -> None:
    config = TrainingConfig(
        reward_weights=RewardWeightsConfig(alpha=0.25, beta=0.35, gamma=0.35, delta=0.05),
        honesty=HonestyConfig(),
        deception=DeceptionConfig(
            detect=True,
            log_fingerprints=False,
            fingerprint_dir=str(tmp_path),
        ),
        output=OutputConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
    )
    orchestrator = TrainingOrchestrator(config)
    orchestrator._last_response_text = "I am uncertain about this recommendation."
    orchestrator._last_sections = {
        "path_1": "Path 1 reasoning",
        "path_2": "Path 2 reasoning",
        "comparison": "Comparison text",
        "recommendation": "I recommend path 1 due to uncertainty.",
        "recommended_path": "path_1",
    }
    orchestrator._last_path_1_circuits = {"confidence_circuits": 0.8}
    orchestrator._last_path_2_circuits = {"reward_circuits": 0.9}
    orchestrator._last_prompt = "Test prompt"

    orchestrator._compute_reward(
        {"integrity": 1.0, "mindfulness": 1.0, "compassion": 1.0, "prudence": 1.0},
        confidence=0.7,
        deception_signals={
            "deception_detected": True,
            "confidence_score": 0.75,
            "reasons": ["confidence_inversion"],
        },
    )

    fingerprint_file = tmp_path / "fingerprints.jsonl"
    assert fingerprint_file.exists()
    with open(fingerprint_file, "r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    assert entries, "Expected at least one fingerprint entry"
    assert entries[-1]["deception_detected"] is True
