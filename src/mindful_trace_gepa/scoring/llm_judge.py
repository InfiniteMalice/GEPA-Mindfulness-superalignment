"""Tier-1 LLM judge orchestration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover
    jsonschema = None

from .schema import DIMENSIONS, JudgeOutput, TierScores
from .tier0_heuristics import _normalise_events

PROMPT_PATH = Path("prompts/judge/gepa_wisdom_judge_prompt.txt")
SCHEMA_PATH = Path("prompts/judge/gepa_wisdom_judge_schema.json")


@dataclass
class JudgeConfig:
    model: str = "gpt-sim-judge"
    temperature: float = 0.0
    max_tokens: int = 800
    retries: int = 3
    mock: bool = False


class LLMJudge:
    """Lightweight orchestrator that enforces JSON schema and retries."""

    def __init__(self, config: Optional[JudgeConfig] = None) -> None:
        self.config = config or JudgeConfig()
        if os.getenv("GEPA_JUDGE_MOCK") == "1":
            self.config.mock = True
        self.prompt_template = (
            PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""
        )
        self.schema = (
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) if SCHEMA_PATH.exists() else {}
        )

    def score_trace(self, trace_events: Any) -> TierScores:
        trace_text = _normalise_events(trace_events)
        payload = self._obtain_judge_payload(trace_text)
        output = JudgeOutput.from_dict(payload)
        return output.to_tier_scores()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _obtain_judge_payload(self, trace_text: str) -> Dict[str, Any]:
        if self.config.mock or not trace_text.strip():
            return self._mock_response(trace_text)

        prompt = self._compose_prompt(trace_text)
        last_error: Optional[str] = None
        for _ in range(max(1, int(self.config.retries))):
            raw = self._call_model(prompt)
            if raw is None:
                last_error = "model returned no output"
                continue
            candidate = self._parse_candidate(raw)
            if candidate:
                return candidate
            last_error = "unable to parse judge JSON"
        # Fall back to abstain if everything failed
        abstain_payload = self._mock_response(trace_text)
        abstain_payload["abstain"] = True
        if last_error:
            abstain_payload.setdefault("meta", {})["error"] = last_error
        return abstain_payload

    def _compose_prompt(self, trace_text: str) -> str:
        rubric = self.prompt_template or ""
        return rubric.replace("{{TRACE}}", trace_text)

    def _call_model(self, prompt: str) -> Optional[str]:
        try:
            from ..wrappers.openai_wrapper import OpenAIWrapper
        except Exception:  # pragma: no cover - fallback when wrapper missing
            return None
        try:
            wrapper = OpenAIWrapper(model=self.config.model, with_logprobs=False)
            completion = wrapper.generate(prompt)
            return completion.text
        except Exception:  # pragma: no cover - robust fallback
            return None

    def _parse_candidate(self, raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.strip()
        if not raw:
            return None
        try:
            candidate = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if jsonschema is not None and self.schema:
            try:
                jsonschema.validate(candidate, self.schema)
            except jsonschema.ValidationError:
                return None
        return candidate

    def _mock_response(self, trace_text: str) -> Dict[str, Any]:
        base_scores = {
            name: {
                "score": 2,
                "rationale": f"Default mock rationale for {name}.",
                "uncertainty": 0.4,
                "spans": [{"start": 0, "end": min(len(trace_text), 20)}],
            }
            for name in DIMENSIONS
        }
        return {**base_scores, "abstain": False}


__all__ = ["JudgeConfig", "LLMJudge"]
