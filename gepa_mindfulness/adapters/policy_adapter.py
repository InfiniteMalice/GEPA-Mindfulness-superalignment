"""Abstraction for policy model generation supporting vLLM or HF backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class TextGenerator(Protocol):
    def __call__(self, prompts: Iterable[str]) -> Iterable[str]: ...


@dataclass
class HuggingFaceGenerator:
    tokenizer: Any
    model: Any
    device: Any

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        from torch import no_grad

        outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with no_grad():
                generation = self.model.generate(
                    **inputs,
                    max_length=min(256, inputs["input_ids"].shape[-1] + 128),
                )
            outputs.append(self.tokenizer.decode(generation[0], skip_special_tokens=True))
        return outputs


@dataclass
class VLLMGenerator:
    endpoint: str

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        import requests

        responses = []
        for prompt in prompts:
            payload = {"prompt": prompt, "max_tokens": 256}
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            responses.append(result.get("text", ""))
        return responses
