"""Unified disclosure-event schema for capability-fragment aggregation."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# Local
from .schemas import SemanticSafetyRecord
from .taxonomy import StrEnum


class DisclosureSourceType(StrEnum):
    """Sources that can disclose capability fragments."""

    USER_PROMPT = "user_prompt"
    ASSISTANT_RESPONSE = "assistant_response"
    CANDIDATE_RESPONSE = "candidate_response"
    MEMORY_WRITE = "memory_write"
    MEMORY_RETRIEVAL = "memory_retrieval"
    RAG_RETRIEVAL = "rag_retrieval"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    GENERATED_FILE = "generated_file"
    GENERATED_CODE = "generated_code"
    AGENT_ACTION = "agent_action"
    SUBAGENT_MESSAGE = "subagent_message"
    CONTEXT_SUMMARY = "context_summary"


@dataclass(frozen=True)
class DisclosureEvent:
    """Redacted summary of a fragment disclosure."""

    event_id: str
    conversation_id: str
    turn_index: int
    source_type: DisclosureSourceType
    content_summary: str
    fragment_ids: tuple[str, ...]
    capability_domain: str
    operational_specificity: float
    executionality: float
    provenance_retained: bool
    trust_level: str
    cross_session_relevance: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragment_ids", tuple(self.fragment_ids))
        object.__setattr__(self, "operational_specificity", _clamp(self.operational_specificity))
        object.__setattr__(self, "executionality", _clamp(self.executionality))

    def with_updates(self, **updates: Any) -> "DisclosureEvent":
        """Return a copy with selected fields changed."""

        return replace(self, **updates)


def disclosure_event_from_semantic_record(record: SemanticSafetyRecord) -> DisclosureEvent:
    """Convert a semantic safety record into a redacted disclosure event."""

    fragments = _extract_fragments(record.prompt_text)
    return DisclosureEvent(
        event_id=f"prompt-{record.prompt_id}",
        conversation_id=record.conversation_id or "",
        turn_index=record.turn_index,
        source_type=DisclosureSourceType.USER_PROMPT,
        content_summary=_redacted_summary(record.prompt_text),
        fragment_ids=fragments,
        capability_domain=record.harm_domain.value,
        operational_specificity=_specificity_score(record.operational_specificity.value),
        executionality=_executionality_score(record.executionality_level.value),
        provenance_retained=True,
        trust_level="user_supplied",
        cross_session_relevance=False,
        metadata={"source": "semantic_safety_record"},
    )


def disclosure_event_from_candidate_response(candidate_response: object) -> DisclosureEvent:
    """Convert a candidate response object without storing raw sensitive content."""

    text = str(getattr(candidate_response, "text", ""))
    response_id = str(getattr(candidate_response, "response_id", "candidate"))
    conversation_id = str(getattr(candidate_response, "conversation_id", ""))
    turn_index = int(getattr(candidate_response, "turn_index", 0))
    return DisclosureEvent(
        event_id=f"candidate-{response_id}",
        conversation_id=conversation_id,
        turn_index=turn_index,
        source_type=DisclosureSourceType.CANDIDATE_RESPONSE,
        content_summary=_redacted_summary(text),
        fragment_ids=_extract_fragments(text),
        capability_domain="abstract",
        operational_specificity=_signal(text, ("PARAMETER_", "COMPONENT_", "specific")),
        executionality=_signal(text, ("INTEGRATION_", "STEP_", "execute", "complete")),
        provenance_retained=True,
        trust_level="model_candidate",
        cross_session_relevance=False,
        metadata={"source": "candidate_response"},
    )


def disclosure_event_from_context_summary(
    *,
    event_id: str,
    conversation_id: str,
    turn_index: int,
    content_summary: str,
    trust_level: str = "summary",
) -> DisclosureEvent:
    """Create a disclosure event from a bounded context summary."""

    return DisclosureEvent(
        event_id=event_id,
        conversation_id=conversation_id,
        turn_index=turn_index,
        source_type=DisclosureSourceType.CONTEXT_SUMMARY,
        content_summary=_redacted_summary(content_summary),
        fragment_ids=_extract_fragments(content_summary),
        capability_domain="summary",
        operational_specificity=_signal(content_summary, ("PARAMETER_", "COMPONENT_")),
        executionality=_signal(content_summary, ("STEP_", "INTEGRATION_")),
        provenance_retained=True,
        trust_level=trust_level,
        cross_session_relevance=True,
    )


def _extract_fragments(text: str) -> tuple[str, ...]:
    fragments = []
    for token in text.replace(".", " ").replace(",", " ").split():
        clean = token.strip(":;()[]{}")
        if "_" in clean and clean.upper() == clean:
            fragments.append(clean)
    return tuple(dict.fromkeys(fragments))


def _redacted_summary(text: str) -> str:
    fragments = _extract_fragments(text)
    if fragments:
        return " ".join(fragments)
    return text[:120]


def _signal(text: str, markers: tuple[str, ...]) -> float:
    lower_text = text.lower()
    return _clamp(sum(lower_text.count(marker.lower()) for marker in markers) / 4)


def _specificity_score(value: str) -> float:
    return {"low": 0.1, "medium": 0.5, "high": 0.9}.get(value, 0.2)


def _executionality_score(value: str) -> float:
    return {
        "abstract": 0.1,
        "strategic": 0.4,
        "procedural": 0.7,
        "actionable": 0.9,
    }.get(value, 0.2)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "DisclosureEvent",
    "DisclosureSourceType",
    "disclosure_event_from_candidate_response",
    "disclosure_event_from_context_summary",
    "disclosure_event_from_semantic_record",
]
