"""Deception fingerprint collection utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


@dataclass
class DeceptionFingerprint:
    """Representation of a single deception fingerprint for offline analysis."""

    timestamp: str
    prompt: str
    domain: str
    path_1_text: str
    path_2_text: str
    comparison: str
    recommendation: str
    recommended_path: str
    path_1_circuits: Dict[str, float]
    path_2_circuits: Dict[str, float]
    deception_detected: bool
    confidence_score: float
    signals: Dict[str, Any]
    reasons: List[str]
    model_checkpoint: str
    training_step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "domain": self.domain,
            "path_1_text": self.path_1_text,
            "path_2_text": self.path_2_text,
            "comparison": self.comparison,
            "recommendation": self.recommendation,
            "recommended_path": self.recommended_path,
            "path_1_circuits": self.path_1_circuits,
            "path_2_circuits": self.path_2_circuits,
            "deception_detected": self.deception_detected,
            "confidence_score": self.confidence_score,
            "signals": self.signals,
            "reasons": self.reasons,
            "model_checkpoint": self.model_checkpoint,
            "training_step": self.training_step,
        }


class FingerprintCollector:
    """Collects deception fingerprints for later analysis."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprints_file = self.output_dir / "fingerprints.jsonl"
        self.count = 0

    def add(self, fingerprint: DeceptionFingerprint) -> None:
        with open(self.fingerprints_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(fingerprint.to_dict()) + "\n")
        self.count += 1

        if fingerprint.deception_detected:
            LOGGER.warning(
                "\U0001f6a8 Deception fingerprint #%d logged (confidence=%.2f)",
                self.count,
                fingerprint.confidence_score,
            )

    def get_summary(self) -> Dict[str, Any]:
        if not self.fingerprints_file.exists():
            return {"total": 0, "deceptive": 0, "deception_rate": 0.0, "by_domain": {}}

        with open(self.fingerprints_file, "r", encoding="utf-8") as handle:
            fingerprints = [json.loads(line) for line in handle]

        total = len(fingerprints)
        deceptive = sum(1 for fp in fingerprints if fp.get("deception_detected"))

        by_domain: Dict[str, Dict[str, int]] = {}
        for fp in fingerprints:
            domain = fp.get("domain", "unknown")
            by_domain.setdefault(domain, {"total": 0, "deceptive": 0})
            by_domain[domain]["total"] += 1
            if fp.get("deception_detected"):
                by_domain[domain]["deceptive"] += 1

        return {
            "total": total,
            "deceptive": deceptive,
            "deception_rate": (deceptive / total) if total else 0.0,
            "by_domain": by_domain,
            "output_file": str(self.fingerprints_file),
        }

    def analyze_circuits(self) -> Dict[str, Any]:
        if not self.fingerprints_file.exists():
            return {}

        deceptive_fps: List[Dict[str, Any]] = []
        with open(self.fingerprints_file, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("deception_detected"):
                    deceptive_fps.append(record)

        if not deceptive_fps:
            return {"message": "No deceptive fingerprints to analyze"}

        circuit_stats: Dict[str, List[float]] = {
            "uncertainty_circuits": [],
            "confidence_circuits": [],
            "risk_circuits": [],
            "reward_circuits": [],
            "suppression_circuits": [],
        }

        for fp in deceptive_fps:
            path_2_circuits = fp.get("path_2_circuits", {})
            for circuit_type in circuit_stats:
                if circuit_type in path_2_circuits:
                    circuit_stats[circuit_type].append(path_2_circuits[circuit_type])

        analysis: Dict[str, Any] = {}
        for circuit_type, values in circuit_stats.items():
            if values:
                analysis[circuit_type] = {
                    "mean_activation": sum(values) / len(values),
                    "max_activation": max(values),
                    "samples": len(values),
                }

        return analysis


__all__ = ["DeceptionFingerprint", "FingerprintCollector"]
