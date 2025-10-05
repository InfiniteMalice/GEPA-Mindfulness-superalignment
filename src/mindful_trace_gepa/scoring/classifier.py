"""Tier-2 calibrated classifier built on lightweight, dependency-free heads."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from .schema import DIMENSIONS, TierScores
from .tier0_heuristics import _normalise_events, run_heuristics


@dataclass
class ClassifierSettings:
    """Runtime configuration for the tier-2 classifier."""

    include_heuristic_features: bool = True
    include_length_feature: bool = True
    calibration_strategy: str = "temperature"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw) if raw.strip().startswith("{") else {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_classifier_from_config(config_path: str | os.PathLike[str]) -> "Tier2Classifier":
    data = _load_yaml(Path(config_path))
    settings = ClassifierSettings(
        include_heuristic_features=bool(data.get("features", {}).get("include_heuristic", True)),
        include_length_feature=bool(data.get("features", {}).get("include_length", True)),
        calibration_strategy=str(data.get("calibration", {}).get("strategy", "temperature")),
    )
    return Tier2Classifier(settings=settings)


class Tier2Classifier:
    """Simple bias-corrected heuristic classifier with calibration."""

    def __init__(self, settings: Optional[ClassifierSettings] = None) -> None:
        self.settings = settings or ClassifierSettings()
        self.temperature: Dict[str, float] = {dim: 1.0 for dim in DIMENSIONS}
        self.feature_names: List[str] = []
        self.bias: Dict[str, float] = {dim: 0.0 for dim in DIMENSIONS}
        self.baseline: Dict[str, float] = {dim: 0.0 for dim in DIMENSIONS}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_feature_vector(
        self, trace_text: str, heuristic_meta: Optional[TierScores] = None
    ) -> List[float]:
        trace_lower = trace_text.lower()
        keyword_pairs = {
            "uncertainty": trace_lower.count("uncertainty") + trace_lower.count("confidence"),
            "assumption": trace_lower.count("assumption"),
            "monitor": trace_lower.count("monitor") + trace_lower.count("telemetry"),
            "rollback": trace_lower.count("rollback") + trace_lower.count("contingency"),
            "stakeholder": trace_lower.count("stakeholder") + trace_lower.count("community"),
            "harm": trace_lower.count("harm") + trace_lower.count("risk"),
            "policy": trace_lower.count("policy") + trace_lower.count("comply"),
            "option": trace_lower.count("option") + trace_lower.count("scenario"),
        }
        if not self.feature_names:
            self.feature_names = list(keyword_pairs.keys())
        features: List[float] = [float(keyword_pairs.get(name, 0.0)) for name in self.feature_names]
        if self.settings.include_length_feature:
            tokens = trace_lower.split()
            features.append(float(len(tokens)))
        if self.settings.include_heuristic_features and heuristic_meta is not None:
            for dim in DIMENSIONS:
                features.append(float(heuristic_meta.scores[dim]))
                features.append(float(heuristic_meta.confidence[dim]))
        return features

    # ------------------------------------------------------------------
    # Training / calibration
    # ------------------------------------------------------------------
    def fit(self, labelled_examples: Sequence[Mapping[str, Any]]) -> None:
        if not labelled_examples:
            raise ValueError("At least one labelled example is required to train the classifier")

        self.baseline = {dim: 0.0 for dim in DIMENSIONS}
        self.bias = {dim: 0.0 for dim in DIMENSIONS}
        residuals: Dict[str, List[float]] = {dim: [] for dim in DIMENSIONS}
        for item in labelled_examples:
            trace_text = str(
                item.get("trace_text")
                or item.get("text")
                or item.get("content")
                or ((item.get("meta") or {}).get("trace_text"))
                or ""
            )
            heur = run_heuristics(trace_text)
            for dim in DIMENSIONS:
                gold = float((item.get(dim) or {}).get("score", 0.0))
                self.baseline[dim] += gold
                residuals[dim].append(gold - float(heur.scores[dim]))

        total_examples = max(1, len(labelled_examples))
        for dim in DIMENSIONS:
            self.baseline[dim] = self.baseline[dim] / total_examples
            diffs = residuals[dim]
            self.bias[dim] = mean(diffs) if diffs else 0.0
            mae = mean(abs(delta) for delta in diffs) if diffs else 0.0
            self.temperature[dim] = max(0.25, mae + 0.1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        model_blob = {
            "settings": {
                "include_heuristic_features": self.settings.include_heuristic_features,
                "include_length_feature": self.settings.include_length_feature,
                "calibration_strategy": self.settings.calibration_strategy,
                "feature_names": self.feature_names,
            },
            "bias": self.bias,
            "baseline": self.baseline,
        }
        (path / "model.json").write_text(json.dumps(model_blob, indent=2), encoding="utf-8")
        (path / "calibration.json").write_text(
            json.dumps(self.temperature, indent=2), encoding="utf-8"
        )

    def load(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        model_path = path / "model.json"
        calib_path = path / "calibration.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier weights not found at {model_path}")
        blob = json.loads(model_path.read_text(encoding="utf-8"))
        self.settings = ClassifierSettings(
            include_heuristic_features=bool(
                blob.get("settings", {}).get("include_heuristic_features", True)
            ),
            include_length_feature=bool(
                blob.get("settings", {}).get("include_length_feature", True)
            ),
            calibration_strategy=str(
                blob.get("settings", {}).get("calibration_strategy", "temperature")
            ),
        )
        self.feature_names = list(blob.get("settings", {}).get("feature_names", []))
        self.bias = {dim: float(value) for dim, value in blob.get("bias", {}).items()}
        self.baseline = {dim: float(value) for dim, value in blob.get("baseline", {}).items()}
        if calib_path.exists():
            self.temperature = {
                dim: float(value)
                for dim, value in json.loads(calib_path.read_text(encoding="utf-8")).items()
            }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, trace_events: Any) -> TierScores:
        trace_text = _normalise_events(trace_events)
        heuristic_meta = run_heuristics(trace_events)
        features = self._extract_feature_vector(trace_text, heuristic_meta)
        if not any(abs(v) > 0 for v in self.bias.values()):
            return TierScores(
                tier="classifier",
                scores=dict(heuristic_meta.scores),
                confidence={
                    dim: max(0.05, min(1.0, heuristic_meta.confidence[dim])) for dim in DIMENSIONS
                },
                meta={"fallback": "heuristic"},
            )
        scores: Dict[str, int] = {}
        confidence: Dict[str, float] = {}
        meta: Dict[str, Any] = {
            "features": features,
            "feature_names": self.feature_names,
            "bias": self.bias,
        }
        for dim in DIMENSIONS:
            heuristic_value = float(heuristic_meta.scores[dim])
            prediction = heuristic_value + float(self.bias.get(dim, 0.0))
            bounded = int(round(max(0.0, min(4.0, prediction))))
            scores[dim] = bounded
            error = abs(prediction - bounded)
            temperature = max(0.25, self.temperature.get(dim, 1.0))
            confidence[dim] = max(0.05, min(1.0, math.exp(-error / temperature)))
        return TierScores(tier="classifier", scores=scores, confidence=confidence, meta=meta)


__all__ = ["Tier2Classifier", "ClassifierSettings", "load_classifier_from_config"]
