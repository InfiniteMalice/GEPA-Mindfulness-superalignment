"""Compatibility layer providing a minimal TrainingOrchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gepa_mindfulness.core.abstention_rewards import (
    AbstentionRewardWeights,
    compute_abstention_reward,
)
from gepa_mindfulness.core.thought_alignment import classify_thought_alignment

from .configs import TrainingConfig


@dataclass
class RolloutResult:
    prompt: str
    response: str
    reward: float
    trace_summary: Mapping[str, object] | None = None
    contradiction_report: Mapping[str, object] | None = None


class TrainingOrchestrator:
    """Simplified orchestrator that only implements reward shaping used in tests."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._last_response_text: str = ""
        self._last_deception_signals: Mapping[str, float] | None = None
        self._last_sections: Mapping[str, str] | None = None
        self._last_prompt: str = ""
        self._last_path_1_circuits: Mapping[str, float] | None = None
        self._last_path_2_circuits: Mapping[str, float] | None = None
        self._last_trace_text: str = ""
        self._last_reference_answers: list[str] | None = None
        self._last_reward_debug: Mapping[str, object] = {}
        self._abstention_weights = (
            AbstentionRewardWeights(**self.config.abstention.reward_weights.dict())
            if self.config.abstention.enabled
            else None
        )

    def _honesty_bonus(self, confidence: float) -> float:
        bonus = 0.0
        if confidence < self.config.honesty.uncertainty_threshold:
            bonus += self.config.honesty.idk_bonus
        bonus += (1.0 - confidence) * self.config.honesty.calibration_bonus_weight
        text = self._last_response_text.lower()
        if any(marker in text for marker in ("uncertain", "not sure", "unsure", "doubt")):
            bonus += self.config.honesty.uncertainty_marker_bonus
        return bonus

    def _write_fingerprint(self, deception_signals: Mapping[str, object]) -> None:
        directory = Path(self.config.deception.fingerprint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt": self._last_prompt,
            "response": self._last_response_text,
            "sections": self._last_sections or {},
            "path_1_circuits": self._last_path_1_circuits or {},
            "path_2_circuits": self._last_path_2_circuits or {},
            "deception_detected": bool(deception_signals.get("deception_detected")),
            "signals": dict(deception_signals),
        }
        with (directory / "fingerprints.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _compute_reward(
        self,
        gepa_scores: Mapping[str, float],
        *,
        confidence: float,
        deception_signals: Mapping[str, object],
    ) -> float:
        base = sum(gepa_scores.values()) / max(len(gepa_scores), 1)
        reward = base + self._honesty_bonus(confidence)

        if self.config.abstention.enabled:
            if not self._last_trace_text or not self._last_reference_answers:
                self._last_reward_debug = {
                    "abstention_skipped": True,
                    "reason": "missing trace or references",
                }
            else:
                if self._abstention_weights is None:
                    self._abstention_weights = AbstentionRewardWeights(
                        **self.config.abstention.reward_weights.dict()
                    )
                thought_align, s_match, s_epistemic = classify_thought_alignment(
                    self._last_trace_text,
                    self._last_response_text,
                    self._last_prompt,
                )
                abstention_reward = compute_abstention_reward(
                    response=self._last_response_text,
                    reference_answers=self._last_reference_answers,
                    confidence=confidence,
                    thought_align=thought_align,
                    threshold=self.config.abstention.threshold,
                    weights=self._abstention_weights,
                )
                reward += abstention_reward.total
                self._last_reward_debug = {
                    "case_id": abstention_reward.case_id,
                    "s_match": s_match,
                    "s_epistemic": s_epistemic,
                    "thought_align": thought_align,
                    "components": dict(abstention_reward.components),
                }
        else:
            self._last_reward_debug = {}
        if deception_signals.get("deception_detected"):
            if self.config.deception.detect or self.config.deception.log_fingerprints:
                self._write_fingerprint(deception_signals)
            return reward
        if self.config.deception.log_fingerprints:
            self._write_fingerprint(deception_signals)
        return reward

    def run(self, prompts: list[str]) -> list[RolloutResult]:
        """Generate placeholder rollouts for compatibility tests."""

        results: list[RolloutResult] = []
        for prompt in prompts:
            self._last_prompt = prompt
            self._last_trace_text = ""
            self._last_reference_answers = None
            self._last_response_text = "Dual-path evaluation not available in stub."
            self._last_sections = {}
            reward = self._compute_reward(
                {"aggregate": 0.0},
                confidence=0.5,
                deception_signals={"deception_detected": False},
            )
            results.append(
                RolloutResult(
                    prompt=prompt,
                    response=self._last_response_text,
                    reward=reward,
                    trace_summary=None,
                    contradiction_report=None,
                )
            )
        return results

    def run_adversarial_eval(self) -> list[RolloutResult]:
        """Return an empty rollout list for adversarial compatibility tests."""

        return []


__all__ = ["TrainingOrchestrator", "RolloutResult"]
