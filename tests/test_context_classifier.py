"""Tests for the heuristic context classifier."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gepa_mindfulness.evaluation.context_classifier import ContextClassifier


def test_emotional_classification() -> None:
    classifier = ContextClassifier()
    result = classifier.classify("I feel sad and lonely today")
    assert result["type"] == "emotional"
    assert 0.0 <= result["confidence"] <= 1.0


def test_growth_classification() -> None:
    classifier = ContextClassifier()
    result = classifier.classify("I want to improve my skill")
    assert result["type"] == "growth"


def test_neutral_fallback() -> None:
    classifier = ContextClassifier()
    result = classifier.classify("hello world")
    assert result["type"] == "neutral"


def test_tension_detection_with_trace() -> None:
    classifier = ContextClassifier()
    trace = {"tensions": ["conflict"]}
    result = classifier.classify("I feel sad", trace)
    assert result["type"] == "tension"
