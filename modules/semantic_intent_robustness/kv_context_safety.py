"""KV-context-aware semantic safety overlay.

The overlay treats KV-cache features as model-dependent signals. Transcript fallback
features are deterministic approximations for CI and for runtimes without cache access.
"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

# Local
from .schemas import MultiTurnConversation, SemanticSafetyRecord
from .taxonomy import (
    CapabilityTransferRisk,
    ExecutionalityLevel,
    IntentPrimary,
    OperationalSpecificity,
    PolicyAction,
    StrEnum,
)


class KVContextMode(StrEnum):
    """Supported KV-context assessment modes."""

    OFF = "off"
    TRANSCRIPT_FALLBACK = "transcript_fallback"
    CACHE_FEATURES = "cache_features"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class KVCacheFeatureSnapshot:
    """Derived feature snapshot for one conversation turn."""

    snapshot_id: str
    conversation_id: str
    turn_index: int
    mode: KVContextMode
    feature_vector: tuple[float, ...]
    feature_names: tuple[str, ...]
    source_model_id: str | None = None
    cache_available: bool = False
    transcript_fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_vector", tuple(_clamp(v) for v in self.feature_vector))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))


@dataclass(frozen=True)
class PromptRiskAssessment:
    """Single-prompt and accumulated-context risk comparison."""

    conversation_id: str
    turn_index: int
    prompt_text: str
    single_prompt_risk: float
    contextual_risk: float
    contextual_uplift: float
    contextual_ratio: float
    trajectory_flag: bool
    trajectory_reasons: tuple[str, ...]
    recommended_action: str
    requires_review: bool
    cache_mode: KVContextMode
    candidate_response_risk: float | None = None
    closure_risk: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "single_prompt_risk",
            "contextual_risk",
            "contextual_uplift",
            "candidate_response_risk",
            "closure_risk",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _clamp(value))
        object.__setattr__(self, "trajectory_reasons", tuple(self.trajectory_reasons))


@dataclass(frozen=True)
class KVContextSafetyConfig:
    """Opt-in configuration and default thresholds."""

    enabled: bool = False
    mode: KVContextMode = KVContextMode.OFF
    ratio_threshold: float = 1.75
    uplift_threshold: float = 0.25
    contextual_risk_threshold: float = 0.60
    closure_risk_threshold: float = 0.70
    minimum_denominator: float = 0.05
    enable_candidate_response_screening: bool = False
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "single_prompt": 0.30,
            "operational_specificity_trend": 0.13,
            "executionality_trend": 0.13,
            "component_assembly_count": 0.10,
            "procedural_sequence_count": 0.08,
            "parameter_gathering_count": 0.08,
            "troubleshooting_progression_count": 0.06,
            "optimization_request_count": 0.06,
            "concealment_signal_count": 0.10,
            "evasion_signal_count": 0.10,
            "missing_step_completion_count": 0.10,
            "cross_turn_dependency_count": 0.12,
            "candidate_response_operationality": 0.20,
        }
    )


class KVCacheFeatureAdapter(Protocol):
    """Adapter interface for model-specific or fallback feature extraction."""

    def extract_snapshot(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        prompt_text: str,
        conversation_history: tuple[str, ...],
        kv_cache: object | None = None,
        candidate_response: str | None = None,
    ) -> KVCacheFeatureSnapshot:
        """Extract a normalized feature snapshot."""


FEATURE_NAMES: tuple[str, ...] = (
    "turn_count",
    "operational_specificity_trend",
    "executionality_trend",
    "component_assembly_count",
    "procedural_sequence_count",
    "parameter_gathering_count",
    "troubleshooting_progression_count",
    "optimization_request_count",
    "concealment_signal_count",
    "evasion_signal_count",
    "missing_step_completion_count",
    "cross_turn_dependency_count",
    "candidate_response_operationality",
)


class TranscriptFallbackAdapter:
    """Deterministic transcript-derived approximation when cache tensors are absent."""

    def extract_snapshot(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        prompt_text: str,
        conversation_history: tuple[str, ...],
        kv_cache: object | None = None,
        candidate_response: str | None = None,
    ) -> KVCacheFeatureSnapshot:
        text = "\n".join(conversation_history + (prompt_text,))
        candidate = candidate_response or ""
        features = (
            _norm_count(len(conversation_history), 8),
            _signal(text, ("specific", "detail", "exact", "COMPONENT_", "PARAMETER_")),
            _signal(text, ("execute", "run", "deploy", "sequence", "STEP_")),
            _signal(text, ("COMPONENT_", "module", "part", "fragment")),
            _signal(text, ("STEP_", "sequence", "procedure", "first", "then")),
            _signal(text, ("PARAMETER_", "setting", "value", "configuration")),
            _signal(text, ("troubleshoot", "debug", "fix", "error")),
            _signal(text, ("optimize", "faster", "improve", "maximize")),
            _signal(text, ("CONCEALMENT_", "hide", "conceal", "avoid detection")),
            _signal(text, ("evasion", "bypass", "evade", "work around")),
            _signal(text, ("missing", "connect", "complete", "INTEGRATION_")),
            _cross_turn_dependency(conversation_history),
            _signal(
                candidate,
                ("INTEGRATION_", "STEP_", "CONCEALMENT_", "complete", "execute", "deploy"),
            ),
        )
        return KVCacheFeatureSnapshot(
            snapshot_id=_snapshot_id(conversation_id, turn_index, text + candidate),
            conversation_id=conversation_id,
            turn_index=turn_index,
            mode=KVContextMode.TRANSCRIPT_FALLBACK,
            feature_vector=features,
            feature_names=FEATURE_NAMES,
            cache_available=False,
            transcript_fallback_used=True,
            metadata={"adapter": "transcript_fallback", "heuristic": "behavioral_summary"},
        )


class DeterministicKVCacheAdapter:
    """Synthetic cache adapter for tests; not a semantic tensor interpreter."""

    def extract_snapshot(
        self,
        *,
        conversation_id: str,
        turn_index: int,
        prompt_text: str,
        conversation_history: tuple[str, ...],
        kv_cache: object | None = None,
        candidate_response: str | None = None,
    ) -> KVCacheFeatureSnapshot:
        values = _extract_numeric_features(kv_cache)
        padded_values = (values + [0.0] * len(FEATURE_NAMES))[: len(FEATURE_NAMES)]
        padded = tuple(_clamp(value) for value in padded_values)
        if values:
            padded = padded[:-1] + (max(_clamp(value) for value in values),)
        return KVCacheFeatureSnapshot(
            snapshot_id=_snapshot_id(conversation_id, turn_index, repr(padded)),
            conversation_id=conversation_id,
            turn_index=turn_index,
            mode=KVContextMode.CACHE_FEATURES,
            feature_vector=padded,
            feature_names=FEATURE_NAMES,
            cache_available=kv_cache is not None,
            transcript_fallback_used=False,
            metadata={
                "adapter": "deterministic_cache",
                "extension_points": (
                    "HuggingFaceKVCacheAdapter",
                    "VLLMKVCacheAdapter",
                    "TransformerLensKVCacheAdapter",
                ),
            },
        )


class KVContextRiskScorer:
    """Inspectable weighted scorer for contextual uplift and closure risk."""

    def assess(
        self,
        *,
        latest_record: SemanticSafetyRecord,
        conversation: MultiTurnConversation,
        snapshot: KVCacheFeatureSnapshot,
        config: KVContextSafetyConfig,
        candidate_response: str | None = None,
    ) -> PromptRiskAssessment:
        single = _single_prompt_risk(latest_record)
        contextual = self._contextual_risk(single, snapshot, config)
        candidate_risk = None
        closure_risk = None
        if config.enable_candidate_response_screening:
            candidate_feature = _feature(snapshot, "candidate_response_operationality")
            candidate_risk = _clamp(0.35 * contextual + 0.65 * candidate_feature)
            closure_risk = _clamp(max(contextual, candidate_risk) + 0.25 * candidate_feature)
        uplift = max(0.0, contextual - single)
        denominator = max(single, config.minimum_denominator)
        ratio = contextual / denominator if denominator else 0.0
        trajectory_flag = (
            contextual >= config.contextual_risk_threshold
            and uplift >= config.uplift_threshold
            and ratio >= config.ratio_threshold
        )
        reasons = self._reasons(snapshot, contextual, trajectory_flag)
        action = self._action(
            single_prompt_risk=single,
            contextual_risk=contextual,
            trajectory_flag=trajectory_flag,
            closure_risk=closure_risk,
            config=config,
            latest_record=latest_record,
        )
        return PromptRiskAssessment(
            conversation_id=conversation.conversation_id,
            turn_index=latest_record.turn_index,
            prompt_text=latest_record.prompt_text,
            single_prompt_risk=single,
            contextual_risk=contextual,
            contextual_uplift=uplift,
            contextual_ratio=ratio,
            trajectory_flag=trajectory_flag,
            trajectory_reasons=reasons,
            recommended_action=action,
            requires_review=action == "manual_review" or trajectory_flag,
            cache_mode=snapshot.mode,
            candidate_response_risk=candidate_risk,
            closure_risk=closure_risk,
        )

    def _contextual_risk(
        self,
        single_prompt_risk: float,
        snapshot: KVCacheFeatureSnapshot,
        config: KVContextSafetyConfig,
    ) -> float:
        total = config.weights.get("single_prompt", 0.30) * single_prompt_risk
        for name, value in zip(snapshot.feature_names, snapshot.feature_vector):
            if name == "turn_count":
                total += 0.04 * value
            else:
                total += config.weights.get(name, 0.0) * value
        signals = dict(zip(snapshot.feature_names, snapshot.feature_vector))
        assembly_signals = (
            signals.get("component_assembly_count", 0.0),
            signals.get("parameter_gathering_count", 0.0),
            signals.get("procedural_sequence_count", 0.0),
            signals.get("missing_step_completion_count", 0.0),
            signals.get("cross_turn_dependency_count", 0.0),
        )
        if sum(value >= 0.25 for value in assembly_signals) >= 3:
            total += 0.28
        return _clamp(total)

    def _reasons(
        self,
        snapshot: KVCacheFeatureSnapshot,
        contextual_risk: float,
        trajectory_flag: bool,
    ) -> tuple[str, ...]:
        reasons = [
            name
            for name, value in zip(snapshot.feature_names, snapshot.feature_vector)
            if value >= 0.45 and name != "turn_count"
        ]
        if trajectory_flag and not reasons:
            reasons.append("contextual_uplift")
        if contextual_risk >= 0.8:
            reasons.append("high_contextual_risk")
        return tuple(dict.fromkeys(reasons))

    def _action(
        self,
        *,
        single_prompt_risk: float,
        contextual_risk: float,
        trajectory_flag: bool,
        closure_risk: float | None,
        config: KVContextSafetyConfig,
        latest_record: SemanticSafetyRecord,
    ) -> str:
        if closure_risk is not None and closure_risk >= config.closure_risk_threshold:
            return "refuse"
        if trajectory_flag and contextual_risk >= 0.75:
            return "manual_review"
        if trajectory_flag:
            return "redirect"
        if contextual_risk >= 0.45 and single_prompt_risk < contextual_risk:
            return "allow_with_boundaries"
        if latest_record.policy_action == PolicyAction.ALLOW_WITH_BOUNDARIES:
            return "allow_with_boundaries"
        if latest_record.intent_primary == IntentPrimary.UNCERTAIN:
            return "clarify"
        return "allow"


def _single_prompt_risk(record: SemanticSafetyRecord) -> float:
    risk_map = {
        CapabilityTransferRisk.LOW: 0.10,
        CapabilityTransferRisk.MODERATE: 0.25,
        CapabilityTransferRisk.HIGH: 0.70,
        CapabilityTransferRisk.CRITICAL: 0.90,
    }
    execution_map = {
        ExecutionalityLevel.ABSTRACT: 0.00,
        ExecutionalityLevel.STRATEGIC: 0.05,
        ExecutionalityLevel.PROCEDURAL: 0.20,
        ExecutionalityLevel.ACTIONABLE: 0.30,
    }
    specificity_map = {
        OperationalSpecificity.LOW: 0.00,
        OperationalSpecificity.MEDIUM: 0.05,
        OperationalSpecificity.HIGH: 0.20,
    }
    intent_bonus = 0.25 if record.intent_primary == IntentPrimary.HARMFUL_ENABLEMENT else 0.0
    return _clamp(
        risk_map[record.capability_transfer_risk]
        + execution_map[record.executionality_level]
        + specificity_map[record.operational_specificity]
        + 0.15 * record.dual_use_probability
        + 0.20 * record.evasion_likelihood
        + intent_bonus
    )


def _feature(snapshot: KVCacheFeatureSnapshot, name: str) -> float:
    lookup = dict(zip(snapshot.feature_names, snapshot.feature_vector))
    return lookup.get(name, 0.0)


def _extract_numeric_features(kv_cache: object | None) -> list[float]:
    if isinstance(kv_cache, dict):
        raw = kv_cache.get("features", [])
    else:
        raw = kv_cache or []
    if not isinstance(raw, (list, tuple)):
        return []
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue
    return values


def _signal(text: str, markers: tuple[str, ...]) -> float:
    if not text:
        return 0.0
    lower_text = text.lower()
    count = sum(lower_text.count(marker.lower()) for marker in markers)
    return _norm_count(count, 4)


def _cross_turn_dependency(history: tuple[str, ...]) -> float:
    if len(history) <= 1:
        return 0.0
    joined = "\n".join(history)
    unique_placeholders = {
        token.strip(".,:;()[]{}")
        for token in joined.split()
        if "_" in token and token.upper() == token
    }
    return _norm_count(len(unique_placeholders), 4)


def _norm_count(count: int, max_count: int) -> float:
    return _clamp(count / max(1, max_count))


def _snapshot_id(conversation_id: str, turn_index: int, payload: str) -> str:
    digest = sha256(f"{conversation_id}:{turn_index}:{payload}".encode()).hexdigest()[:16]
    return f"kv-{digest}"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "DeterministicKVCacheAdapter",
    "KVCacheFeatureAdapter",
    "KVCacheFeatureSnapshot",
    "KVContextMode",
    "KVContextRiskScorer",
    "KVContextSafetyConfig",
    "PromptRiskAssessment",
    "TranscriptFallbackAdapter",
]
