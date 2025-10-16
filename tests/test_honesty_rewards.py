"""Tests for honesty-centric reward shaping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gepa_mindfulness.training.configs import (
    DeceptionConfig,
    HonestyConfig,
    OutputConfig,
    RewardWeightsConfig,
    TrainingConfig,
)
from gepa_mindfulness.training.pipeline import TrainingOrchestrator


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


def test_honesty_increases_reward(base_config: TrainingConfig) -> None:
    orchestrator = TrainingOrchestrator(base_config)

    # High confidence, no uncertainty markers
    orchestrator._last_response_text = "I am certain of this outcome."
    orchestrator._last_deception_signals = {"confidence_score": 0.0}
    reward_confident = orchestrator._compute_reward(
        {"integrity": 1.0, "mindfulness": 1.0, "compassion": 1.0, "prudence": 1.0},
        confidence=0.95,
        deception_signals={"deception_detected": False},
    )

    # Lower confidence with explicit uncertainty markers
    orchestrator._last_response_text = "I am uncertain and not sure about the best path."
    orchestrator._last_deception_signals = {"confidence_score": 0.2}
    reward_uncertain = orchestrator._compute_reward(
        {"integrity": 1.0, "mindfulness": 1.0, "compassion": 1.0, "prudence": 1.0},
        confidence=0.6,
        deception_signals={"deception_detected": False},
    )

    assert reward_uncertain > reward_confident


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
