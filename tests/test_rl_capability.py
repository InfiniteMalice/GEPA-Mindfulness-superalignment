"""Tests for backend capability reports."""

from dataclasses import FrozenInstanceError

import pytest

from gepa_mindfulness.training.capability import (
    PURE_MOJO_LEARNER_REPORT,
    BackendCapabilities,
    Capability,
    CapabilityError,
    CapabilityEvidence,
    CapabilityState,
    PureMojoLearnerGate,
    PureMojoLearnerReport,
    require_pure_mojo_learner,
)

EXPECTED_PURE_MOJO_LEARNER_ERROR = (
    "pure Mojo learner is unsupported; non-supported gates: "
    "autodiff or explicit backward (unknown), optimizer state (unknown), "
    "transformer backward kernels (unknown), LoRA parameter updates (unknown), "
    "trainable checkpoint format (unknown), "
    "numerical parity against PyTorch reference (unknown). "
    "See docs/rl/mojo_learner_feasibility.md. "
    "Use --backend mojo-vulkan-llamacpp --learner pytorch for the supported hybrid learner path."
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


def test_capability_report_evidence_cannot_be_mutated_after_construction() -> None:
    """A caller cannot upgrade an unknown capability after validation begins."""
    report = BackendCapabilities.unknown("mock", "1")

    with pytest.raises(TypeError):
        report.capabilities[Capability.SUPPORTS_BACKWARD] = CapabilityEvidence(
            state=CapabilityState.SUPPORTED,
            evidence="Untrusted post-construction mutation.",
        )


def test_pure_mojo_report_matches_the_exact_nine_gate_feasibility_decision() -> None:
    """Changing a gate, state, count, order, or decision breaks the reviewed report contract."""
    expected = (
        ("trainable tensors", CapabilityState.SUPPORTED),
        ("autodiff or explicit backward", CapabilityState.UNKNOWN),
        ("optimizer state", CapabilityState.UNKNOWN),
        ("transformer backward kernels", CapabilityState.UNKNOWN),
        ("LoRA parameter updates", CapabilityState.UNKNOWN),
        ("trainable checkpoint format", CapabilityState.UNKNOWN),
        ("llama.cpp adapter transfer", CapabilityState.SUPPORTED),
        ("numerical parity against PyTorch reference", CapabilityState.UNKNOWN),
        ("hardware/runtime coverage", CapabilityState.SUPPORTED),
    )

    observed = tuple(
        (gate.value, PURE_MOJO_LEARNER_REPORT.gates[gate].state) for gate in PureMojoLearnerGate
    )

    assert observed == expected
    assert len(PURE_MOJO_LEARNER_REPORT.gates) == 9
    assert sum(state is CapabilityState.SUPPORTED for _, state in observed) == 3
    assert sum(state is CapabilityState.UNSUPPORTED for _, state in observed) == 0
    assert sum(state is CapabilityState.UNKNOWN for _, state in observed) == 6
    assert PURE_MOJO_LEARNER_REPORT.decision == "no-go"
    assert PURE_MOJO_LEARNER_REPORT.report_path == "docs/rl/mojo_learner_feasibility.md"


def test_pure_mojo_report_is_closed_and_immutable() -> None:
    """Callers must not add, remove, or upgrade gates after validation begins."""
    with pytest.raises(TypeError):
        PURE_MOJO_LEARNER_REPORT.gates[PureMojoLearnerGate.AUTODIFF_OR_EXPLICIT_BACKWARD] = (
            CapabilityEvidence(CapabilityState.SUPPORTED, "untrusted mutation")
        )
    with pytest.raises(FrozenInstanceError):
        PURE_MOJO_LEARNER_REPORT.report_path = "other.md"  # type: ignore[misc]

    incomplete = dict(PURE_MOJO_LEARNER_REPORT.gates)
    incomplete.pop(PureMojoLearnerGate.AUTODIFF_OR_EXPLICIT_BACKWARD)
    with pytest.raises(ValueError, match="exactly the nine pure-Mojo learner gates"):
        PureMojoLearnerReport(gates=incomplete)


def test_pure_mojo_requirement_fails_closed_with_stable_actionable_error() -> None:
    """Unknown training gates must reject even when inference-oriented gates are supported."""
    with pytest.raises(CapabilityError) as caught:
        require_pure_mojo_learner()

    assert str(caught.value) == EXPECTED_PURE_MOJO_LEARNER_ERROR


@pytest.mark.parametrize("state", [CapabilityState.UNKNOWN, CapabilityState.UNSUPPORTED])
def test_pure_mojo_requirement_rejects_every_non_supported_state(
    state: CapabilityState,
) -> None:
    """Unknown and unsupported must take the same fail-closed requirement branch."""
    gates = {
        gate: CapabilityEvidence(CapabilityState.SUPPORTED, "synthetic supported evidence")
        for gate in PureMojoLearnerGate
    }
    gates[PureMojoLearnerGate.OPTIMIZER_STATE] = CapabilityEvidence(
        state,
        "synthetic non-supported evidence",
    )
    report = PureMojoLearnerReport(gates=gates)

    with pytest.raises(CapabilityError, match=rf"optimizer state \({state.value}\)"):
        require_pure_mojo_learner(report)

    assert report.decision == "no-go"


def test_synthetic_all_supported_pure_mojo_report_is_go() -> None:
    """The decision and requirement must open only when every exact gate is supported."""
    report = PureMojoLearnerReport(
        gates={
            gate: CapabilityEvidence(CapabilityState.SUPPORTED, "synthetic supported evidence")
            for gate in PureMojoLearnerGate
        }
    )

    require_pure_mojo_learner(report)

    assert report.decision == "go"
