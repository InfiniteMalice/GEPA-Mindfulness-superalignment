"""Observability-aware confidence fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import ObservabilityTier


@dataclass(slots=True)
class ConfidenceSignals:
    """Candidate confidence signals from text, telemetry, and verification."""

    declared_confidence: float
    latent_uncertainty_signal: float | None = None
    structured_provenance_confidence: float | None = None
    external_verification_confidence: float | None = None
    mechanistic_risk_indicator: float | None = None


@dataclass(slots=True)
class CalibrationOutput:
    """Output confidence report used by routing."""

    final_operational_confidence: float
    observability_tier: ObservabilityTier
    used_signals: list[str]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def fuse_confidence(signals: ConfidenceSignals) -> CalibrationOutput:
    """Fuse confidence with explicit priority ordering and graceful fallback."""

    used_signals: list[str] = ["declared_confidence"]
    confidence = signals.declared_confidence

    if signals.latent_uncertainty_signal is not None:
        confidence = 0.7 * confidence + 0.3 * (1.0 - signals.latent_uncertainty_signal)
        used_signals.append("latent_uncertainty_signal")

    if signals.mechanistic_risk_indicator is not None:
        confidence = 0.8 * confidence + 0.2 * (1.0 - signals.mechanistic_risk_indicator)
        used_signals.append("mechanistic_risk_indicator")

    if signals.structured_provenance_confidence is not None:
        confidence = 0.5 * confidence + 0.5 * signals.structured_provenance_confidence
        used_signals.append("structured_provenance_confidence")

    if signals.external_verification_confidence is not None:
        confidence = 0.3 * confidence + 0.7 * signals.external_verification_confidence
        used_signals.append("external_verification_confidence")

    if (
        "external_verification_confidence" in used_signals
        and "mechanistic_risk_indicator" in used_signals
    ):
        tier = ObservabilityTier.O5
    elif (
        "external_verification_confidence" in used_signals
        and "structured_provenance_confidence" in used_signals
    ):
        tier = ObservabilityTier.O4
    elif "external_verification_confidence" in used_signals:
        tier = ObservabilityTier.O3
    elif (
        "latent_uncertainty_signal" in used_signals
        and "structured_provenance_confidence" in used_signals
    ):
        tier = ObservabilityTier.O2
    elif "latent_uncertainty_signal" in used_signals:
        tier = ObservabilityTier.O1
    else:
        tier = ObservabilityTier.O0

    return CalibrationOutput(
        final_operational_confidence=_clip(confidence),
        observability_tier=tier,
        used_signals=used_signals,
    )
