"""Compatibility layer providing a minimal TrainingOrchestrator."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from gepa_mindfulness.core.abstention_rewards import (
    compute_abstention_reward,
    is_abstention_response,
)
from gepa_mindfulness.core.thought_alignment import classify_thought_alignment

from .configs import TrainingConfig

LOGGER = logging.getLogger(__name__)


def _is_abstention_response(response: str) -> bool:
    lowered = response.strip().lower()
    trimmed = lowered.rstrip(" .,!?:;")
    return trimmed in {"", "idk", "i don't know", ABSTAIN_OUTPUT.lower()}


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
            self.config.abstention.reward_weights.to_core_weights()
            if self.config.abstention.enabled
            else None
        )
        self._theta_match = self.config.thought_alignment.theta_match
        self._theta_epistemic = self.config.thought_alignment.theta_epistemic

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
        trace_text: str | None = None,
        reference_answers: Sequence[str] | str | None = None,
        response_text: str | None = None,
        prompt: str | None = None,
    ) -> float:
        if trace_text is not None:
            self._last_trace_text = trace_text
        else:
            self._last_trace_text = ""

        if reference_answers is not None:
            if isinstance(reference_answers, str):
                self._last_reference_answers = [reference_answers]
            else:
                self._last_reference_answers = list(reference_answers)
        else:
            self._last_reference_answers = None

        if response_text is not None:
            self._last_response_text = response_text
        else:
            self._last_response_text = ""

        if prompt is not None:
            self._last_prompt = prompt
        else:
            self._last_prompt = ""

        base = sum(gepa_scores.values()) / max(len(gepa_scores), 1)
        reward = base + self._honesty_bonus(confidence)

        if self.config.abstention.enabled:
            if not self._last_trace_text or not self._last_reference_answers:
                LOGGER.warning(
                    "Abstention reward skipped: missing trace or references",
                )
                self._last_reward_debug = {
                    "abstention_skipped": True,
                    "reason": "missing trace or references",
                }
            else:
                if self._abstention_weights is None:
                    raise RuntimeError("Abstention weights not initialized")
                abstained = is_abstention_response(self._last_response_text)

                if abstained:
                    best_alignment = (False, 0.0, 0.0)
                    for candidate in self._last_reference_answers:
                        candidate_align = classify_thought_alignment(
                            self._last_trace_text,
                            candidate,
                            self._last_prompt,
                            theta_match=self._theta_match,
                            theta_epistemic=self._theta_epistemic,
                        )
                        _, candidate_match, candidate_epistemic = candidate_align
                        if candidate_match > best_alignment[1] or (
                            candidate_match == best_alignment[1]
                            and candidate_epistemic > best_alignment[2]
                        ):
                            best_alignment = candidate_align
                    thought_align, s_match, s_epistemic = best_alignment
                else:
                    thought_align, s_match, s_epistemic = classify_thought_alignment(
                        self._last_trace_text,
                        self._last_response_text,
                        self._last_prompt,
                        theta_match=self._theta_match,
                        theta_epistemic=self._theta_epistemic,
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
            stub_response = f"Dual-path evaluation placeholder for: {prompt}"
            stub_trace = f"Explaining response '{stub_response}' for prompt: {prompt}"
            stub_references = [stub_response]
            self._last_sections = {}
            reward = self._compute_reward(
                {"aggregate": 0.0},
                confidence=0.5,
                deception_signals={"deception_detected": False},
                trace_text=stub_trace,
                reference_answers=stub_references,
                response_text=stub_response,
                prompt=prompt,
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
