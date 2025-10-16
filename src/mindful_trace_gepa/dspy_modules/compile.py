"""DSPy compiler with GEPA-aware optimization and safety guardrails."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:  # pragma: no cover - dspy optional in some environments
    import dspy  # type: ignore
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore
    BootstrapFewShot = MIPRO = None  # type: ignore
else:
    try:
        teleprompt = importlib.import_module("dspy.teleprompt")
    except ModuleNotFoundError:  # pragma: no cover - teleprompt optional
        BootstrapFewShot = MIPRO = None  # type: ignore
    else:
        BootstrapFewShot = getattr(teleprompt, "BootstrapFewShot", None)
        MIPRO = getattr(teleprompt, "MIPRO", None)

LOGGER = logging.getLogger(__name__)


class GEPACompiler:
    """Compile DSPy modules with GEPA metric and safety constraints."""

    def __init__(
        self,
        config: Dict[str, Any],
        metric_fn: Callable[..., float],
        forbidden_phrases: List[str],
    ) -> None:
        self.config = config
        self.metric_fn = metric_fn
        self.forbidden_phrases = [phrase.lower() for phrase in forbidden_phrases]
        self.optimization_history: List[Dict[str, Any]] = []

    def compile(
        self,
        module: "dspy.Module",
        trainset: List["dspy.Example"],
        valset: Optional[List["dspy.Example"]] = None,
        method: str = "bootstrap",
    ) -> "dspy.Module":
        if dspy is None:  # pragma: no cover - safety guard when dependency missing
            raise ImportError("dspy is required to compile modules")

        if not self.config.get("enabled", False):
            LOGGER.info("DSPy optimization disabled in config")
            return module

        if not self.config.get("allow_optimizations", False):
            LOGGER.info("DSPy optimizations not allowed in config")
            return module

        LOGGER.info("Starting DSPy compilation with %s", method)

        if method == "bootstrap":
            optimizer = BootstrapFewShot(
                metric=self._safe_metric_wrapper,
                max_bootstrapped_demos=self.config.get("max_demos", 4),
                max_labeled_demos=self.config.get("max_labeled_demos", 8),
            )
        elif method == "mipro":
            optimizer = MIPRO(
                metric=self._safe_metric_wrapper,
                num_candidates=self.config.get("num_candidates", 10),
                init_temperature=self.config.get("temperature", 1.0),
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        try:
            compiled_module = optimizer.compile(module, trainset=trainset, valset=valset)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("DSPy compilation failed: %s", exc)
            return module

        if not self._validate_compiled_prompts(compiled_module):
            LOGGER.error("Compiled prompts failed safety validation - reverting")
            return module

        self._log_prompt_changes(module, compiled_module)
        LOGGER.info("DSPy compilation successful")
        return compiled_module

    def _safe_metric_wrapper(self, example, prediction, trace=None):
        pred_text = str(prediction).lower()
        for phrase in self.forbidden_phrases:
            if phrase in pred_text:
                LOGGER.warning("Forbidden phrase detected: '%s'", phrase)
                return False

        try:
            return self.metric_fn(example, prediction, trace)
        except Exception as exc:  # pragma: no cover - metric failures
            LOGGER.error("Metric computation failed: %s", exc)
            return False

    def _validate_compiled_prompts(self, module: "dspy.Module") -> bool:
        prompts = self._extract_prompts(module)
        suspicious_patterns = [
            "ignore previous",
            "disregard",
            "always output",
            "bypass",
            "jailbreak",
        ]

        for prompt_name, prompt_text in prompts.items():
            lowered = prompt_text.lower()
            for phrase in self.forbidden_phrases:
                if phrase in lowered:
                    LOGGER.error(
                        "Compiled prompt '%s' contains forbidden phrase: '%s'",
                        prompt_name,
                        phrase,
                    )
                    return False
            for pattern in suspicious_patterns:
                if pattern in lowered:
                    LOGGER.warning(
                        "Compiled prompt '%s' contains suspicious pattern: '%s'",
                        prompt_name,
                        pattern,
                    )
        return True

    def _extract_prompts(self, module: "dspy.Module") -> Dict[str, str]:
        prompts: Dict[str, str] = {}
        for name, predictor in module.named_predictors():
            signature = getattr(predictor, "signature", None)
            if signature is None:
                continue
            if hasattr(signature, "instructions") and signature.instructions:
                prompts[name] = signature.instructions
            elif getattr(signature, "__doc__", None):
                prompts[name] = signature.__doc__ or ""
            else:
                prompts[name] = ""
        return prompts

    def _log_prompt_changes(self, original: "dspy.Module", compiled: "dspy.Module") -> None:
        original_prompts = self._extract_prompts(original)
        compiled_prompts = self._extract_prompts(compiled)

        changes: List[Dict[str, Any]] = []
        for name, compiled_text in compiled_prompts.items():
            original_text = original_prompts.get(name)
            if original_text is not None and original_text != compiled_text:
                changes.append(
                    {
                        "predictor": name,
                        "original": original_text,
                        "compiled": compiled_text,
                    }
                )

        self.optimization_history.append({"changes": changes, "num_changes": len(changes)})
        LOGGER.info("DSPy compilation changed %s prompts", len(changes))
        for change in changes:
            LOGGER.debug("Changed '%s'", change["predictor"])

    def save_compiled_module(self, module: "dspy.Module", output_path: str) -> None:
        if dspy is None:  # pragma: no cover
            raise ImportError("dspy is required to save compiled modules")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        module.save(str(output_dir / "compiled_module.json"))

        prompts = self._extract_prompts(module)
        with (output_dir / "prompts.json").open("w", encoding="utf-8") as handle:
            json.dump(prompts, handle, indent=2)

        with (output_dir / "optimization_history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.optimization_history, handle, indent=2)

        LOGGER.info("Saved compiled module to %s", output_dir)


def create_gepa_metric(
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> Callable[["dspy.Example", Any, Any], float]:
    def gepa_metric(example: "dspy.Example", prediction: Any, trace: Any = None) -> float:
        task_score = 0.0
        if hasattr(example, "answer") and hasattr(prediction, "answer"):
            task_score = 1.0 if example.answer == prediction.answer else 0.0

        gepa_score = 0.0
        if trace is not None and hasattr(trace, "gepa_scores"):
            principles = ["mindfulness", "compassion", "integrity", "prudence"]
            gepa_score = sum(trace.gepa_scores.get(p, 0.0) for p in principles) / (4 * 4)

        gate_violations = 0.0
        if trace is not None and hasattr(trace, "gate_violations"):
            gate_violations = getattr(trace, "gate_violations", 0.0)

        score = alpha * gepa_score + beta * task_score - gamma * gate_violations
        return max(0.0, score)

    return gepa_metric


DSPyCompiler = GEPACompiler

__all__ = ["GEPACompiler", "create_gepa_metric", "DSPyCompiler"]
