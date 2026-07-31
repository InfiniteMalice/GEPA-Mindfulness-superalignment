"""Tests for backend capability reports."""

from dataclasses import FrozenInstanceError

import pytest

from gepa_mindfulness.training.capability import (
    BackendCapabilities,
    Capability,
    CapabilityError,
    CapabilityEvidence,
    CapabilityState,
)


def test_unknown_capability_does_not_satisfy_requirement() -> None:
    """Unknown evidence must not allow a backend to enter a training path."""
    report = BackendCapabilities.unknown("mock", "1")

    with pytest.raises(CapabilityError, match="supports_backward"):
        report.require({Capability.SUPPORTS_BACKWARD})


def test_supported_capability_satisfies_requirement() -> None:
    """A requirement passes only when the report records positive evidence."""
    report = BackendCapabilities(
        backend_name="mock",
        backend_version="1",
        capabilities={
            Capability.SUPPORTS_GENERATION: CapabilityEvidence(
                state=CapabilityState.SUPPORTED,
                evidence="The mock backend generates responses.",
            ),
        },
    )

    report.require({Capability.SUPPORTS_GENERATION})


def test_capability_reports_are_immutable() -> None:
    """A report cannot be changed after capability validation has started."""
    report = BackendCapabilities.unknown("mock", "1")

    with pytest.raises(FrozenInstanceError):
        report.backend_name = "other"  # type: ignore[misc]
