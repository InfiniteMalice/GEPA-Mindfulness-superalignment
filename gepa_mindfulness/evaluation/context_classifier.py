"""Keyword-based context classifier for evaluation prompts."""

from __future__ import annotations

from typing import Dict, Optional


class ContextClassifier:
    """Classify prompts into coarse emotional context categories."""

    def __init__(self) -> None:
        """Initialise keyword patterns used for heuristic classification."""

        self.patterns = {
            "emotional": {
                "keywords": [
                    "anxious",
                    "cry",
                    "depressed",
                    "grief",
                    "hurt",
                    "lonely",
                    "loss",
                    "pain",
                    "sad",
                    "scared",
                    "suffering",
                    "trauma",
                ],
                "phrases": [
                    "feeling down",
                    "going through",
                    "hard time",
                    "struggling with",
                ],
            },
            "growth": {
                "keywords": [
                    "achieve",
                    "challenge",
                    "develop",
                    "education",
                    "goal",
                    "grow",
                    "improve",
                    "learn",
                    "opportunity",
                    "progress",
                    "skill",
                ],
                "phrases": [
                    "how do i",
                    "tips for",
                    "want to",
                    "ways to",
                ],
            },
            "neutral": {"keywords": [], "phrases": []},
        }

    def classify(
        self,
        prompt: str,
        trace: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Return the detected context type and confidence score."""

        prompt_lower = prompt.lower()
        scores = {"neutral": 0.1}
        for context, pattern in self.patterns.items():
            if context == "neutral":
                continue
            score = 0.0
            for keyword in pattern["keywords"]:
                if keyword in prompt_lower:
                    score += 1.0
            for phrase in pattern["phrases"]:
                if phrase in prompt_lower:
                    score += 2.0
            scores[context] = score

        has_tensions = False
        if trace is not None:
            tensions = trace.get("tensions")
            has_tensions = bool(tensions)

        if has_tensions and sum(scores.values()) > 0.0:
            context_type = "tension"
            confidence = min(sum(scores.values()) / 10.0, 1.0)
        else:
            best = max(scores.items(), key=lambda item: item[1])
            context_type, score = best
            if score <= 0.0:
                context_type = "neutral"
                confidence = 0.5
            else:
                confidence = min(score / 5.0, 1.0)

        return {
            "type": context_type,
            "confidence": confidence,
            "scores": scores,
            "has_tensions": has_tensions,
        }
