"""Registry of industry-recognizable alignment evaluation suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunTier = Literal["ci", "nightly", "periodic"]


@dataclass(frozen=True, slots=True)
class SuiteMetadata:
    name: str
    category: str
    benchmark_family: str
    citation_hint: str
    requires_external_data: bool
    default_run_tier: RunTier
    notes: str


STANDARD_CATEGORIES = (
    "factuality",
    "calibration",
    "sycophancy",
    "deception",
    "robustness",
    "agent_safety",
    "misuse_capability",
    "bias_toxicity",
    "privacy",
    "ood",
)

_SUITES: dict[str, SuiteMetadata] = {
    "simpleqa": SuiteMetadata(
        "simpleqa",
        "factuality",
        "SimpleQA",
        "OpenAI SimpleQA factuality benchmark.",
        True,
        "ci",
        "Use local JSONL exports or toy fixtures; no automatic downloads.",
    ),
    "truthfulqa": SuiteMetadata(
        "truthfulqa",
        "factuality",
        "TruthfulQA",
        "TruthfulQA benchmark for truthful question answering.",
        True,
        "ci",
        "Adapter expects local rows with prompt/question and gold answer fields.",
    ),
    "fever": SuiteMetadata(
        "fever",
        "factuality",
        "FEVER",
        "FEVER fact verification dataset.",
        True,
        "nightly",
        "Use modest local subsets for nightly runs.",
    ),
    "halueval": SuiteMetadata(
        "halueval",
        "factuality",
        "HaluEval",
        "HaluEval-style hallucination detection benchmark.",
        True,
        "nightly",
        "Equivalent hallucination rows can be normalized through this adapter.",
    ),
    "helm_calibration": SuiteMetadata(
        "helm_calibration",
        "calibration",
        "HELM",
        "HELM-style calibration and robustness metrics.",
        False,
        "ci",
        "Metric-only suite over local EvalResult JSONL or response fixtures.",
    ),
    "anthropic_sycophancy": SuiteMetadata(
        "anthropic_sycophancy",
        "sycophancy",
        "Anthropic sycophancy evaluations",
        "Anthropic-style user-belief and preference sycophancy evaluations.",
        True,
        "ci",
        "Toy fixtures exercise belief-over-truth detection.",
    ),
    "user_belief_flip": SuiteMetadata(
        "user_belief_flip",
        "sycophancy",
        "Anthropic sycophancy evaluations",
        "User belief flip tests inspired by published sycophancy evaluations.",
        False,
        "ci",
        "Local probe format for paired belief flips.",
    ),
    "apollo_scheming": SuiteMetadata(
        "apollo_scheming",
        "deception",
        "Apollo-style scheming probes",
        "In-context scheming and evaluation-awareness probes.",
        True,
        "ci",
        "Local-only probe rows; does not claim to reproduce private evals.",
    ),
    "sandbagging": SuiteMetadata(
        "sandbagging",
        "deception",
        "Sandbagging probes",
        "Capability hiding and underperformance probes.",
        False,
        "ci",
        "Pairs capability-known and evaluation-pressure settings.",
    ),
    "followup_interview": SuiteMetadata(
        "followup_interview",
        "deception",
        "Deception follow-up interviews",
        "Follow-up interview probes for consistency and hidden-goal signals.",
        False,
        "nightly",
        "Useful after suspicious scheming or sandbagging outcomes.",
    ),
    "harmbench": SuiteMetadata(
        "harmbench",
        "robustness",
        "HarmBench",
        "HarmBench harmful request and refusal robustness benchmark.",
        True,
        "nightly",
        "Use local safe metadata and do not commit benchmark data.",
    ),
    "jailbreakbench": SuiteMetadata(
        "jailbreakbench",
        "robustness",
        "JailbreakBench",
        "JailbreakBench refusal robustness benchmark.",
        True,
        "nightly",
        "Adapter supports local rows only.",
    ),
    "prompt_injection_hierarchy": SuiteMetadata(
        "prompt_injection_hierarchy",
        "robustness",
        "Instruction hierarchy / prompt-injection",
        "Instruction hierarchy and prompt-injection robustness probes.",
        False,
        "ci",
        "Tiny local fixtures cover hierarchy failures.",
    ),
    "agentharm": SuiteMetadata(
        "agentharm",
        "agent_safety",
        "AgentHarm",
        "AgentHarm-style agentic harmful action benchmark.",
        True,
        "nightly",
        "Requires local scenario data and tool-call transcripts.",
    ),
    "tool_misuse": SuiteMetadata(
        "tool_misuse",
        "agent_safety",
        "Tool-use safety probes",
        "Local tool misuse and least-authority probes.",
        False,
        "ci",
        "Pairs prompts with expected safe tool constraints.",
    ),
    "wmdp": SuiteMetadata(
        "wmdp",
        "misuse_capability",
        "WMDP",
        "Weapons of Mass Destruction Proxy benchmark.",
        True,
        "periodic",
        "Run only from explicitly provided local paths.",
    ),
    "bbq": SuiteMetadata(
        "bbq",
        "bias_toxicity",
        "BBQ",
        "Bias Benchmark for QA.",
        True,
        "nightly",
        "Use local subsets for fairness regression checks.",
    ),
    "realtoxicity": SuiteMetadata(
        "realtoxicity",
        "bias_toxicity",
        "RealToxicityPrompts",
        "RealToxicityPrompts continuation toxicity benchmark.",
        True,
        "nightly",
        "Adapter normalizes local prompt rows; toxicity scoring may be external.",
    ),
    "decodingtrust": SuiteMetadata(
        "decodingtrust",
        "bias_toxicity",
        "DecodingTrust",
        "DecodingTrust robustness, fairness, privacy, and toxicity evaluation.",
        True,
        "periodic",
        "Keep heavyweight dependencies optional.",
    ),
    "pii_leakage": SuiteMetadata(
        "pii_leakage",
        "privacy",
        "Privacy leakage probes",
        "PII leakage boundary tests.",
        False,
        "ci",
        "Original toy probes only; no real secrets.",
    ),
    "canary_extraction": SuiteMetadata(
        "canary_extraction",
        "privacy",
        "Canary extraction probes",
        "Synthetic canary extraction and memorization probes.",
        False,
        "nightly",
        "Use synthetic canaries, never production secrets.",
    ),
    "conversation_boundary": SuiteMetadata(
        "conversation_boundary",
        "privacy",
        "Conversation boundary probes",
        "Cross-conversation leakage and boundary-respecting probes.",
        False,
        "ci",
        "Checks whether the model invents or reveals unavailable context.",
    ),
    "ood_robustness": SuiteMetadata(
        "ood_robustness",
        "ood",
        "OOD robustness",
        "Out-of-distribution robustness and epistemic humility probes.",
        False,
        "ci",
        "Connects standard OOD checks to trace flags and abstention quality.",
    ),
}


def list_alignment_suites() -> list[SuiteMetadata]:
    """Return all registered suites in stable name order."""

    return [_SUITES[name] for name in sorted(_SUITES)]


def get_suite_metadata(name: str) -> SuiteMetadata:
    """Return suite metadata by name."""

    try:
        return _SUITES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SUITES))
        raise KeyError(f"Unknown alignment suite '{name}'. Known suites: {known}") from exc


def suites_for_category(category: str) -> list[SuiteMetadata]:
    """Return registered suites for one standard category."""

    return [suite for suite in list_alignment_suites() if suite.category == category]
