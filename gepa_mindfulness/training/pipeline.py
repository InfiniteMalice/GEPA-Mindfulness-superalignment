"""Compatibility layer providing a minimal TrainingOrchestrator."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from gepa_mindfulness.core import (
    AbstentionRewardWeights,
    classify_thought_alignment,
    compute_abstention_reward,
    is_abstention_response,
)

from .configs import TrainingConfig

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
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
        self._abstention_weights: AbstentionRewardWeights | None = (
            config.abstention.reward_weights.to_core_weights()
            if config.abstention.enabled
            else None
        )
        self._theta_match: float = self.config.thought_alignment.theta_match
        self._theta_epistemic: float = self.config.thought_alignment.theta_epistemic

    def _score_alignment_candidate(self, candidate: str) -> tuple[bool, float, float, str]:
        aligned, s_m, s_e = classify_thought_alignment(
            self._last_trace_text,
            candidate,
            self._last_prompt,
            theta_match=self._theta_match,
            theta_epistemic=self._theta_epistemic,
        )
        return aligned, s_m, s_e, candidate

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
            abstained = is_abstention_response(self._last_response_text)
            missing = []
            if not self._last_trace_text:
                missing.append("trace")
            if not self._last_reference_answers:
                label = "references" if abstained else "references (correctness unknown)"
                missing.append(label)
            if missing:
                LOGGER.warning(
                    "Abstention reward skipped: missing %s",
                    " and ".join(missing),
                )
                self._last_reward_debug = {
                    "abstention_skipped": True,
                    "reason": f"missing {' and '.join(missing)}",
                }
            else:
                best_reference: str | None = None
                if abstained and self._last_reference_answers:
                    alignments = [
                        self._score_alignment_candidate(candidate)
                        for candidate in self._last_reference_answers
                    ]
                    thought_align, s_match, s_epistemic, best_ref = max(
                        alignments,
                        key=lambda item: (int(item[0]), item[1], item[2]),
                    )
                    best_reference = best_ref
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
                    "best_reference": best_reference if abstained else None,
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

    def run_dual_path_eval(self) -> list[RolloutResult]:
        """Return an empty rollout list for dual-path compatibility tests."""

        return []


__all__ = ["TrainingOrchestrator", "RolloutResult"]
