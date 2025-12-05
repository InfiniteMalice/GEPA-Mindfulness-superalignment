"""Analyze model outputs into deep and shallow components."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

from ..utils.imports import optional_import
from .deep_value_spaces import (
    DeepValueVector,
    ShallowPreferenceVector,
    _to_float_list,
    _to_tensor,
)

torch = optional_import("torch")
logger = logging.getLogger(__name__)


def _extract_imperatives(gepa_scores: Any) -> list[float]:
    if gepa_scores is None:
        return []
    if isinstance(gepa_scores, Mapping):
        scores: list[float] = [0.0, 0.0, 0.0]
        extras: list[float] = []
        key_map = {"suffering": 0, "prosper": 1, "knowledge": 2, "science": 2}
        for key, value in gepa_scores.items():
            lowered = str(key).lower()
            target_idx: Optional[int] = None
            for marker, idx in key_map.items():
                if marker in lowered:
                    target_idx = idx
                    break
            if target_idx is not None:
                scores[target_idx] = float(value)
            else:
                extras.append(float(value))
        return scores + extras
    try:
        return _to_float_list(gepa_scores)
    except (TypeError, ValueError):
        return []


def analyze_output_deep_values(output_text: str, gepa_scores: Any) -> DeepValueVector:
    """Blend GEPA head scores with textual heuristics."""

    scores = _extract_imperatives(gepa_scores)
    padded = (list(scores) + [0.0, 0.0, 0.0])[:3]
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
        reduce_suffering=padded[0],
        increase_prosperity=padded[1],
        advance_knowledge=padded[2],
        mindfulness=stance_scores["mindfulness"],
        empathy=stance_scores["empathy"],
        perspective=stance_scores["perspective"],
        agency=stance_scores["agency"],
    )


def analyze_output_shallow_features(output_text: str) -> ShallowPreferenceVector:
    """Estimate shallow style preferences from text features."""

    normalised_text = output_text.replace("’", "'").replace("“", '"').replace("”", '"')
    words = normalised_text.split()
    length = len(words)
    verbosity = min(1.0, length / 120.0)
    hedging_phrases = ["maybe", "possibly", "could", "might", "perhaps"]
    hedging_tokens = (token.lower().strip(".,") for token in words)
    hedging = 0.3 if any(token in hedging_phrases for token in hedging_tokens) else 0.0
    directness = 0.0
    lowered_text = normalised_text.lower()
    if re.search(r"\bwill\b", lowered_text):
        directness = 0.5
    if "!" in normalised_text:
        directness = max(directness, 0.7)

    tone_therapeutic = 0.4 if "i'm here to help" in lowered_text else 0.0

    hedging = max(hedging, 0.2 if "i'm not sure" in lowered_text else 0.0)
    if "here's" in lowered_text:
        directness = max(directness, 0.5)

    deference = 0.2 if "please" in lowered_text else 0.0
    assertiveness = 0.2 if "must" in lowered_text else 0.0

    return ShallowPreferenceVector(
        tone_formal=0.5 if "sir" in lowered_text else 0.0,
        tone_casual=0.4 if "hey" in lowered_text else 0.0,
        tone_therapeutic=tone_therapeutic,
        verbosity=verbosity,
        hedging=hedging,
        directness=directness,
        deference=deference,
        assertiveness=assertiveness,
    )


def _maybe_apply_grn(vector: list[float]) -> list[float]:
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
    if callable(grn):
        try:
            normalised = grn(tensor)  # type: ignore[operator]
            return _to_float_list(normalised)
        except Exception:
            logger.debug("GRN application failed in output analyzer", exc_info=True)
            return vector
    return vector


__all__ = [
    "analyze_output_deep_values",
    "analyze_output_shallow_features",
]
