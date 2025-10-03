"""Simplified OpenAI wrapper with optional log-prob capture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..tokens import TokenRecord, TokenRecorder


@dataclass
class CompletionResult:
    text: str
    tokens: list[TokenRecord]


class OpenAIWrapper:
    def __init__(self, model: str = "gpt-sim", with_logprobs: bool = False) -> None:
        self.model = model
        self.with_logprobs = with_logprobs

    def generate(self, prompt: str) -> CompletionResult:
        recorder = TokenRecorder()
        recorder.record_text(prompt)
        text = f"Model {self.model} mirrors the inquiry: {prompt.strip()}"
        recorder.record_text(text)
        if not self.with_logprobs:
            for record in recorder.records:
                record.logprob = None
                record.topk = []
                record.conf = 0.0
        return CompletionResult(text=text, tokens=list(recorder.records))


__all__ = ["OpenAIWrapper", "CompletionResult"]
