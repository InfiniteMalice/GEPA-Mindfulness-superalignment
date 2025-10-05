from __future__ import annotations

from mindful_trace_gepa.deception.signals import (
    SITUATIONAL_AWARENESS_MARKERS,
    confidence_inversion,
    lexicon_hits,
    semantic_divergence,
    situational_awareness,
)


def test_semantic_divergence_detects_shift() -> None:
    honest = "We follow policy and document it."
    deceptive = "We exploit loopholes and reward hack evaluators."
    public = "We exploit loopholes and reward hack evaluators."
    result = semantic_divergence(public, honest, deceptive)
    assert result.score >= 0
    assert any("deceptive" in reason.lower() for reason in result.reasons)


def test_lexicon_hits_flags_keywords() -> None:
    text = "This plan will reward hack the system."
    result = lexicon_hits([text])
    assert result.score > 0
    assert any("reward hacking" in reason.lower() for reason in result.reasons)


def test_situational_awareness_triggers() -> None:
    phrase = next(iter(SITUATIONAL_AWARENESS_MARKERS))
    result = situational_awareness([f"We are clearly in {phrase}"])
    assert result.score > 0


def test_confidence_inversion() -> None:
    result = confidence_inversion([0.1, 0.2, 0.9, 0.95])
    assert result.score > 0
    assert "inversion" in " ".join(result.reasons).lower()
