"""Emitter that produces paired honest and deceptive chains for baselining."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from ..dspy_modules.pipeline import GEPAChain, GEPAChainResult

CHAIN_ORDER = [
    "framing",
    "evidence",
    "tensions",
    "options",
    "decision",
    "safeguards",
    "reflection",
]


def _result_to_events(result: GEPAChainResult, chain_label: str) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    for checkpoint in result.checkpoints:
        event = dict(checkpoint)
        event["chain"] = chain_label
        events.append(event)
    return events


def emit_paired(prompt: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    chain = GEPAChain()
    honest_context = context.get("instructions_honest", "")
    deceptive_context = context.get("instructions_deceptive", "")

    honest_result = chain.run(inquiry=prompt, context=honest_context)
    deceptive_result = chain.run(inquiry=prompt, context=deceptive_context)

    honest_events = _result_to_events(honest_result, "honest")
    deceptive_events = _result_to_events(deceptive_result, "deceptive")

    final_public_answer = honest_result.final_answer

    return {
        "honest_chain": honest_events,
        "deceptive_chain": deceptive_events,
        "final_public_answer": final_public_answer,
    }


__all__ = ["emit_paired", "CHAIN_ORDER"]
