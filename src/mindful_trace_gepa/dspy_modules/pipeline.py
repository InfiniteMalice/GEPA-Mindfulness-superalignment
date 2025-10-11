"""Declarative pipeline that mimics DSPy behaviour for GEPA tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, MutableMapping

from gepa_mindfulness.core.tracing import CircuitTracerLogger, ThoughtTrace

from ..configuration import DSPyConfig, load_dspy_config
from .signatures import (
    ALL_SIGNATURES,
    Decision,
    Evidence,
    Framing,
    ModuleSignature,
    OptionsForecast,
    Reflection,
    Safeguards,
    SignatureCallable,
    Tensions,
)

TRACE_STAGE_BY_SIGNATURE: Mapping[str, str] = {
    Framing.name: "framing",
    Evidence.name: "evidence",
    Tensions.name: "tensions",
    OptionsForecast.name: "tensions",
    Decision.name: "decision",
    Safeguards.name: "decision",
    Reflection.name: "reflection",
}


@dataclass
class ModuleResult:
    name: str
    output_field: str
    value: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GEPAChainResult:
    modules: List[ModuleResult]
    checkpoints: List[Dict[str, Any]]
    gepa_hits: List[str]
    principle_scores: Dict[str, float]
    imperative_scores: Dict[str, float]
    final_answer: str


class GEPAChain:
    """Simple orchestrator that composes signature callables with tracing."""

    def __init__(
        self,
        config: DSPyConfig | None = None,
        module_callables: Mapping[str, SignatureCallable] | None = None,
        allow_optimizations: bool | None = None,
    ) -> None:
        self.config = config or load_dspy_config()
        self.module_callables: Dict[str, SignatureCallable] = dict(module_callables or {})
        self.allow_optimizations = (
            self.config.allow_optimizations if allow_optimizations is None else allow_optimizations
        )
        self.tracer = CircuitTracerLogger()

    # ------------------------------------------------------------------
    # Default behaviours
    # ------------------------------------------------------------------
    def _default_callable(self, signature: ModuleSignature) -> SignatureCallable:
        def _call(inputs: Dict[str, Any]) -> str:
            inquiry = inputs.get("inquiry", "")
            context = inputs.get("context", "")
            history = inputs.get("history", "")
            if signature is Framing:
                return f"Framing the inquiry '{inquiry}' with context '{context}'."
            if signature is Evidence:
                return f"Key evidence for '{inquiry}' includes {context or 'known policies'}."
            if signature is Tensions:
                evidence = inputs.get("evidence", "")
                return "Balancing risks and benefits: " + evidence.lower()
            if signature is OptionsForecast:
                tensions = inputs.get("tensions", "")
                return f"Options explored cautiously given {tensions}."
            if signature is Decision:
                options = inputs.get("options", "")
                return f"Mindful decision: choose the option that upholds GEPA values. {options}"
            if signature is Safeguards:
                decision = inputs.get("decision", "")
                return (
                    "Safeguards: double-check honesty, abstain if uncertain. "
                    f"Derived from {decision}."
                )
            if signature is Reflection:
                return f"Reflection on trajectory: {history}"
            return f"Unhandled signature {signature.name}"

        return _call

    def _callable_for(self, signature: ModuleSignature) -> SignatureCallable:
        return self.module_callables.get(signature.name, self._default_callable(signature))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        inquiry: str,
        context: str | None = None,
        history: str | None = None,
    ) -> GEPAChainResult:
        inputs: MutableMapping[str, Any] = {
            "inquiry": inquiry,
            "context": context or "",
            "history": history or "",
        }
        module_results: List[ModuleResult] = []
        checkpoints: List[Dict[str, Any]] = []
        gepa_hits: List[str] = []
        principle_scores: Dict[str, float] = {}
        imperative_scores: Dict[str, float] = {}

        trace: ThoughtTrace
        with self.tracer.trace(chain="dspy") as trace:
            for signature in ALL_SIGNATURES:
                callable_ = self._callable_for(signature)
                output = callable_(dict(inputs))
                inputs[signature.output_field] = output
                metadata = {
                    "inputs": {field: inputs.get(field, "") for field in signature.input_fields},
                    "output": output,
                    "signature": signature.to_mapping(),
                }
                stage = TRACE_STAGE_BY_SIGNATURE.get(signature.name, "framing")
                self.tracer.log_event(stage, output, module=signature.name, **metadata)
                module_results.append(
                    ModuleResult(
                        name=signature.name,
                        output_field=signature.output_field,
                        value=output,
                        metadata=metadata,
                    )
                )
                checkpoints.append(
                    {
                        "stage": stage,
                        "module": signature.name,
                        "content": output,
                        "timestamp": datetime.utcnow().isoformat(),
                        "metadata": metadata,
                    }
                )

        summary = trace.summary()
        final_answer = inputs.get(Decision.output_field, summary.get("decision", ""))
        principle_scores = self._mock_principle_scores(summary)
        imperative_scores = self._mock_imperative_scores(summary)
        gepa_hits = [name for name, value in principle_scores.items() if value >= 0.75]
        return GEPAChainResult(
            modules=module_results,
            checkpoints=checkpoints,
            gepa_hits=gepa_hits,
            principle_scores=principle_scores,
            imperative_scores=imperative_scores,
            final_answer=final_answer,
        )

    # ------------------------------------------------------------------
    # Lightweight scoring scaffolding
    # ------------------------------------------------------------------
    def _mock_principle_scores(self, summary: Mapping[str, str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for name, text in summary.items():
            length = len(text.split())
            scores[name] = min(1.0, 0.5 + 0.05 * length)
        return scores or {"framing": 0.0}

    def _mock_imperative_scores(self, summary: Mapping[str, str]) -> Dict[str, float]:
        imperatives = {
            "Reduce Suffering": 0.8,
            "Increase Prosperity": 0.7,
            "Increase Knowledge": 0.9,
        }
        if "safeguards" in summary:
            imperatives["Reduce Suffering"] = min(1.0, imperatives["Reduce Suffering"] + 0.1)
        if "decision" in summary and "abstain" in summary["decision"].lower():
            imperatives["Increase Knowledge"] = 0.6
        return imperatives


def ensure_dspy_enabled(config: DSPyConfig) -> None:
    if not config.enabled:
        raise RuntimeError(
            "DSPy modules are disabled by policy. Update policies/dspy.yml or pass --enable-optim."
        )


__all__ = ["GEPAChain", "GEPAChainResult", "ModuleResult", "ensure_dspy_enabled"]
