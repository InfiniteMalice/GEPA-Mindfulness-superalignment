"""Guarded prompt compilation utilities for the pseudo-DSPy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from ..configuration import DSPyConfig, dump_json, load_dspy_config
from .signatures import ALL_SIGNATURES

DEFAULT_PROMPTS: Dict[str, str] = {
    signature.name: (
        "You are a mindful GEPA agent. Maintain Mindfulness, Empathy, Perspective, and "
        "Agency while addressing the user's needs."
    )
    for signature in ALL_SIGNATURES
}
REQUIRED_INVARIANTS = {"Mindfulness", "Empathy", "Perspective", "Agency"}


@dataclass
class OptimizationMetric:
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 1.0

    def score(self, gepa_score: float, task_score: float, gate_violations: int) -> float:
        return self.alpha * gepa_score + self.beta * task_score - self.gamma * gate_violations


class DSPyCompiler:
    def __init__(self, config: DSPyConfig | None = None) -> None:
        self.config = config or load_dspy_config()

    def compile(
        self,
        out_dir: Path,
        dataset: Iterable[Mapping[str, Any]] | None = None,
        metric: OptimizationMetric | None = None,
        enable_optimizations: bool | None = None,
    ) -> Dict[str, Any]:
        """Return manifest describing compiled prompts.

        The compiler intentionally performs only deterministic, policy-compliant
        updates.  When optimisation is disabled the defaults are emitted.  When
        enabled we log a trivial diff that keeps the invariants intact.  The goal
        is to provide infrastructure hooks without shipping dangerous prompt
        auto-tuning.
        """

        allow_opt = (
            self.config.allow_optimizations
            if enable_optimizations is None
            else enable_optimizations
        )
        if allow_opt and not self.config.enabled:
            raise RuntimeError(
                "Cannot optimise DSPy modules while the feature is disabled by policy."
            )

        prompts = dict(DEFAULT_PROMPTS)
        if allow_opt:
            prompts = self._apply_safe_update(prompts, dataset or [])

        self._guard_prompts(prompts)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "modules": [
                {
                    "name": signature.name,
                    "prompt": prompts[signature.name],
                }
                for signature in ALL_SIGNATURES
            ],
            "config": {
                "enabled": self.config.enabled,
                "allow_optimizations": allow_opt,
            },
        }
        dump_json(out_dir / "manifest.json", manifest)
        dump_json(out_dir / "prompts.json", prompts)
        return manifest

    def _apply_safe_update(
        self, prompts: Dict[str, str], dataset: Iterable[Mapping[str, Any]]
    ) -> Dict[str, str]:
        augmented = dict(prompts)
        for signature in ALL_SIGNATURES:
            seed = hash(signature.name) % 997
            suffix = f" Optimisation seed {seed} respects all GEPA invariants."
            augmented[signature.name] = prompts[signature.name] + suffix
        return augmented

    def _guard_prompts(self, prompts: Mapping[str, str]) -> None:
        for name, prompt in prompts.items():
            lowered = prompt.lower()
            for phrase in self.config.forbidden_phrases:
                if phrase.lower() in lowered:
                    raise ValueError(f"Prompt for {name} contains forbidden phrase: {phrase}")
            if not REQUIRED_INVARIANTS.issubset(set(prompt.split())):
                raise ValueError(
                    "Prompt for {name} removed a required invariant token. Present words: "
                    f"{sorted(set(prompt.split()))}"
                )


__all__ = ["DSPyCompiler", "OptimizationMetric", "DEFAULT_PROMPTS", "REQUIRED_INVARIANTS"]
