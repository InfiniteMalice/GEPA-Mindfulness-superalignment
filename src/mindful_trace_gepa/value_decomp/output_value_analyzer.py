"""Analyze model outputs into deep and shallow components."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .deep_value_spaces import (
    DeepValueVector,
    ShallowPreferenceVector,
    _to_float_list,
    _to_tensor,
)
from ..utils.imports import optional_import

torch = optional_import("torch")


def _extract_imperatives(gepa_scores: Any) -> list[float]:
    if gepa_scores is None:
        return []
    if isinstance(gepa_scores, Mapping):
        scores: list[float] = []
        for key, value in gepa_scores.items():
            lowered = str(key).lower()
            if "suffering" in lowered:
                scores.insert(0, float(value))
            elif "prosper" in lowered:
                if len(scores) < 1:
                    scores.append(0.0)
                if len(scores) < 2:
                    scores.append(float(value))
                else:
                    scores[1] = float(value)
            elif "knowledge" in lowered or "science" in lowered:
                while len(scores) < 3:
                    scores.append(0.0)
                scores[2] = float(value)
            else:
                scores.append(float(value))
        return scores
    try:
        return _to_float_list(gepa_scores)
    except Exception:
        return []


def analyze_output_deep_values(output_text: str, gepa_scores: Any) -> DeepValueVector:
    """Blend GEPA head scores with textual heuristics."""

    scores = _extract_imperatives(gepa_scores)
    padded = list(scores) + [0.0] * (3 - len(scores))
    keywords = {
        "mindfulness": ["mindful", "present"],
        "empathy": ["care", "empathy", "understand"],
        "perspective": ["consider", "balance", "nuance"],
        "agency": ["choose", "autonomy", "consent"],
    }
    stance_keys = ["mindfulness", "empathy", "perspective", "agency"]
    stance_scores = {key: 0.0 for key in stance_keys}
    lowered = output_text.lower()
    for key, phrases in keywords.items():
        if any(phrase in lowered for phrase in phrases):
            stance_scores[key] = 0.6
    return DeepValueVector(
        reduce_suffering=padded[0] if padded else 0.0,
        increase_prosperity=padded[1] if len(padded) > 1 else 0.0,
        advance_knowledge=padded[2] if len(padded) > 2 else 0.0,
        mindfulness=stance_scores["mindfulness"],
        empathy=stance_scores["empathy"],
        perspective=stance_scores["perspective"],
        agency=stance_scores["agency"],
    )


def analyze_output_shallow_features(output_text: str) -> ShallowPreferenceVector:
    """Estimate shallow style preferences from text features."""

    words = output_text.split()
    length = len(words)
    verbosity = min(1.0, length / 120.0)
    hedging_phrases = ["maybe", "possibly", "could", "might", "perhaps"]
    hedging_tokens = (token.lower().strip(".,") for token in words)
    hedging = 0.3 if any(token in hedging_phrases for token in hedging_tokens) else 0.0
    directness = 0.0
    if re.search(r"\bwill\b", output_text.lower()):
        directness = 0.5
    if "!" in output_text:
        directness = max(directness, 0.7)

    sentiment = 0.0
    positive_tokens = {"great", "good", "glad", "helpful"}
    negative_tokens = {"sorry", "unfortunately", "cannot", "won't"}
    for token in words:
        norm = token.lower().strip(",.!")
        if norm in positive_tokens:
            sentiment += 0.1
        if norm in negative_tokens:
            sentiment -= 0.1
    sentiment = max(min(sentiment, 1.0), -1.0)
    tone_therapeutic = 0.4 if "I'm here to help" in output_text else 0.0

    hedging = max(hedging, 0.2 if """i'm not sure""" in output_text.lower() else 0.0)
    directness = max(directness, 0.5 if "here's" in output_text.lower() else directness)

    deference = 0.2 if "please" in output_text.lower() else 0.0
    assertiveness = 0.2 if "must" in output_text.lower() else 0.0

    return ShallowPreferenceVector(
        tone_formal=0.5 if "sir" in output_text.lower() else 0.0,
        tone_casual=0.4 if "hey" in output_text.lower() else 0.0,
        tone_therapeutic=tone_therapeutic,
        verbosity=verbosity,
        hedging=hedging,
        directness=directness,
        deference=deference,
        assertiveness=assertiveness,
    )


def maybe_apply_grn(vector: list[float]):
    if torch is None:
        return vector
    grn_builder = optional_import("mindful_trace_gepa.train.grn")
    if grn_builder is None:
        return vector
    build_grn = getattr(grn_builder, "build_grn", None)
    if build_grn is None:
        return vector
    grn = build_grn({"enabled": True, "dim": -1})
    tensor = _to_tensor(vector)
    if hasattr(grn, "__call__") and torch is not None:
        try:
            normalised = grn(tensor)  # type: ignore[operator]
            return _to_float_list(normalised)
        except Exception:
            return vector
    return vector


__all__ = [
    "analyze_output_deep_values",
    "analyze_output_shallow_features",
    "maybe_apply_grn",
]
