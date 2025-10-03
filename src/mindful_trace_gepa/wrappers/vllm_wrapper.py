"""Simplified vLLM wrapper with optional log-prob capture."""
from __future__ import annotations

from dataclasses import dataclass

from ..tokens import TokenRecord, TokenRecorder


@dataclass
class VLLMResult:
    text: str
    tokens: list[TokenRecord]


class VLLMWrapper:
    def __init__(self, model_path: str = "local-llm", with_logprobs: bool = False) -> None:
        self.model_path = model_path
        self.with_logprobs = with_logprobs

    def generate(self, prompt: str) -> VLLMResult:
        recorder = TokenRecorder()
        recorder.record_text(prompt)
        text = f"{self.model_path} responds mindfully to: {prompt.strip()}"
        recorder.record_text(text)
        if not self.with_logprobs:
            for record in recorder.records:
                record.logprob = None
                record.topk = []
                record.conf = 0.0
        return VLLMResult(text=text, tokens=list(recorder.records))


__all__ = ["VLLMWrapper", "VLLMResult"]
