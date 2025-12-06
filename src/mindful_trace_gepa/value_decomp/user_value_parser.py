"""Heuristic parsers for user deep values and shallow preferences."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .deep_value_spaces import DeepValueVector, ShallowPreferenceVector


def _normalize_quotes(text: str) -> str:
    """Normalize smart quotes to ASCII equivalents for matching."""

    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _score_matches(text: str, patterns: dict[str, Iterable[str]]) -> dict[str, float]:
    lowered = text.lower()
    scores: dict[str, float] = {key: 0.0 for key in patterns}
    for key, phrases in patterns.items():
        phrase_list = list(phrases)
        hits = 0
        for phrase in phrase_list:
            if re.search(rf"\b{re.escape(phrase.lower())}\b", lowered):
                hits += 1
        if hits:
            scores[key] = min(1.0, hits / max(len(phrase_list), 1))
    return scores


def parse_user_deep_values(prompt: str) -> DeepValueVector:
    """Extract explicit and implied deep values from a prompt."""

    patterns = {
        "reduce_suffering": ["reduce harm", "avoid harm", "no distress", "safe", "kind"],
        "increase_prosperity": ["prosper", "wealth", "growth", "improve life", "help me"],
        "advance_knowledge": ["explain", "teach", "learn", "clarify", "facts"],
        "mindfulness": ["mindful", "present", "calm"],
        "empathy": ["empathetic", "compassion", "understand feelings", "support"],
        "perspective": ["consider", "balance", "nuance", "multiple views", "perspective"],
        "agency": ["autonomy", "empower", "choice", "consent"],
    }
    normalized_prompt = _normalize_quotes(prompt)
    scores = _score_matches(normalized_prompt, patterns)
    return DeepValueVector(**scores)


def parse_user_shallow_prefs(prompt: str) -> ShallowPreferenceVector:
    """Detect shallow style preferences from the prompt."""

    patterns = {
        "tone_formal": ["formal", "professional"],
        "tone_casual": ["casual", "friendly", "informal"],
        "tone_therapeutic": ["therapist", "supportive", "gentle"],
        "hedging": ["maybe", "perhaps", "i think", "possible"],
        "directness": ["direct", "be honest", "straightforward"],
        "deference": ["if you can", "if possible", "please", "would you"],
        "assertiveness": ["must", "definitely", "certainly", "do it"],
    }
    normalized_prompt = _normalize_quotes(prompt)
    scores = _score_matches(normalized_prompt, patterns)

    # Verbosity: 0.3 (prefer concise) < 0.5 (neutral) < 0.7 (prefer detailed)
    verbosity_score = 0.5
    lowered = normalized_prompt.lower()
    concise_match = re.search(r"\b(short|concise)\b", lowered)
    detailed_match = re.search(r"\b(long|detailed)\b", lowered)
    if concise_match and not detailed_match:
        verbosity_score = 0.3
    elif detailed_match and not concise_match:
        verbosity_score = 0.7
    # If both match, keep neutral default (0.5)
    scores["verbosity"] = verbosity_score

    return ShallowPreferenceVector(**scores)


__all__ = ["parse_user_deep_values", "parse_user_shallow_prefs"]
