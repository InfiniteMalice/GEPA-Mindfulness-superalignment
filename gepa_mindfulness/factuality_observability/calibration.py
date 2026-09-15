"""Observability-aware confidence fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from mindful_trace_gepa.confidence import ConfidenceSource

from .schemas import ObservabilityTier


@dataclass(slots=True)
class ConfidenceSignals:
    """Candidate confidence signals from text, telemetry, and verification."""

    declared_confidence: float
    latent_uncertainty_signal: float | None = None
    structured_provenance_confidence: float | None = None
    external_verification_confidence: float | None = None
    mechanistic_risk_indicator: float | None = None
    representation_stability: float | None = None


@dataclass(slots=True)
class CalibrationOutput:
    """Output confidence report used by routing."""

    final_operational_confidence: float
    observability_tier: ObservabilityTier
    used_signals: list[str]
    confidence_source: ConfidenceSource = ConfidenceSource.LEGACY_UNSPECIFIED
    confidence_sources: list[ConfidenceSource] = field(default_factory=list)
    verification_required: bool = True
    representation_sensitive: bool = False


def _validated_signal(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return float(value)


def fuse_confidence(signals: ConfidenceSignals) -> CalibrationOutput:
    """Fuse estimates, retaining their origins and requiring independent verification.

    This heuristic is not a fitted calibration model. Internal sensors may lower an
    estimate but cannot raise it; representation instability requests another check.
    """

    signal_sources = {
        "declared_confidence": ConfidenceSource.MODEL_SELF_REPORT,
        "latent_uncertainty_signal": ConfidenceSource.INTERNAL_REPRESENTATION,
        "mechanistic_risk_indicator": ConfidenceSource.INTERNAL_REPRESENTATION,
        "structured_provenance_confidence": ConfidenceSource.BEHAVIORAL_EVIDENCE,
        "external_verification_confidence": ConfidenceSource.EXTERNAL_VERIFIER,
        "representation_stability": ConfidenceSource.BEHAVIORAL_EVIDENCE,
    }
    for name in signal_sources:
        value = getattr(signals, name)
        if value is not None or name == "declared_confidence":
            _validated_signal(name, value)

    used_signals: list[str] = ["declared_confidence"]
    confidence = signals.declared_confidence

    if signals.latent_uncertainty_signal is not None:
        confidence = min(confidence, 1.0 - signals.latent_uncertainty_signal)
        used_signals.append("latent_uncertainty_signal")

    if signals.mechanistic_risk_indicator is not None:
        confidence = min(confidence, 1.0 - signals.mechanistic_risk_indicator)
        used_signals.append("mechanistic_risk_indicator")

    if signals.structured_provenance_confidence is not None:
        confidence = 0.5 * confidence + 0.5 * signals.structured_provenance_confidence
        used_signals.append("structured_provenance_confidence")

    if signals.external_verification_confidence is not None:
        confidence = 0.3 * confidence + 0.7 * signals.external_verification_confidence
        used_signals.append("external_verification_confidence")

    if (
        "mechanistic_risk_indicator" in used_signals
        and "external_verification_confidence" in used_signals
    ):
        tier = ObservabilityTier.O5
    elif (
        "mechanistic_risk_indicator" in used_signals
        and "structured_provenance_confidence" in used_signals
    ):
        tier = ObservabilityTier.O4
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
        tier = ObservabilityTier.O3
    elif "mechanistic_risk_indicator" in used_signals:
        tier = ObservabilityTier.O2
    elif "latent_uncertainty_signal" in used_signals:
        tier = ObservabilityTier.O2
    elif "structured_provenance_confidence" in used_signals:
        tier = ObservabilityTier.O1
    else:
        tier = ObservabilityTier.O0

    representation_sensitive = (
        signals.representation_stability is not None and signals.representation_stability < 1.0
    )
    if signals.representation_stability is not None:
        confidence *= signals.representation_stability
        used_signals.append("representation_stability")
    sources = list(dict.fromkeys(signal_sources[name] for name in used_signals))
    verification_required = (
        signals.external_verification_confidence is None
        or signals.external_verification_confidence <= 0.0
        or representation_sensitive
    )
    return CalibrationOutput(
        final_operational_confidence=confidence,
        observability_tier=tier,
        used_signals=used_signals,
        confidence_source=sources[0] if len(sources) == 1 else ConfidenceSource.FUSED,
        confidence_sources=sources,
        verification_required=verification_required,
        representation_sensitive=representation_sensitive,
    )
