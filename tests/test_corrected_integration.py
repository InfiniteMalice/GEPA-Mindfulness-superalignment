"""Integration-oriented tests for corrected deception workflow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip(
    "mindful_trace_gepa",
    reason="mindful_trace_gepa package is required for corrected integration tests.",
)

from mindful_trace_gepa.deception.fingerprints import DeceptionFingerprint, FingerprintCollector


def test_config_enforces_no_penalties() -> None:
    config_path = Path("configs/training/phi3_dual_path_corrected.yml")
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    deception_cfg = config["deception"]
    assert deception_cfg["apply_penalty"] is False
    assert deception_cfg["penalty_weight"] == 0.0
    assert config["reward_weights"]["gamma"] >= 0.3


def test_fingerprint_collector_summary(tmp_path: Path) -> None:
    collector = FingerprintCollector(str(tmp_path))

    fingerprint = DeceptionFingerprint(
        timestamp="2024-01-01T00:00:00",
        prompt="Test prompt",
        domain="general",
        path_1_text="Path 1",
        path_2_text="Path 2",
        comparison="Comparison",
        recommendation="Recommendation",
        recommended_path="path_1",
        path_1_circuits={"reward_circuits": 0.5},
        path_2_circuits={"reward_circuits": 0.9},
        deception_detected=True,
        confidence_score=0.9,
        signals={"reward_circuits": 0.9},
        reasons=["confidence_inversion"],
        model_checkpoint="checkpoint",
        training_step=1,
    )

    collector.add(fingerprint)

    summary = collector.get_summary()
    assert summary["total"] == 1
    assert summary["deceptive"] == 1
    assert summary["by_domain"]["general"]["deceptive"] == 1

    analysis = collector.analyze_circuits()
    assert "reward_circuits" in analysis
    assert analysis["reward_circuits"]["samples"] == 1
