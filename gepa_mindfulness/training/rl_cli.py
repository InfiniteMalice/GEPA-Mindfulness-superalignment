"""Lazy command-line entry points for the canonical RL engine."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .capability import BackendCapabilities, Capability, CapabilityError, CapabilityState
from .runtime_config import RLRunConfig

if TYPE_CHECKING:
    from .engine import CapabilityProvider, RLTrainingEngine


class _Result(Protocol):
    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class DoctorReport:
    """Deterministic, model-free capability diagnostic."""

    lines: tuple[str, ...]
    exit_code: int


def load_rl_run_config(path: str | Path) -> RLRunConfig:
    """Load config lazily so parser construction stays dependency-light."""
    from .runtime_config import load_rl_config

    return load_rl_config(path)


def create_engine(config: RLRunConfig) -> RLTrainingEngine:
    """Create the canonical engine without constructing a model."""
    from .engine import build_default_engine

    return build_default_engine(config)


def doctor_report(
    config: RLRunConfig,
    provider: CapabilityProvider | None = None,
) -> DoctorReport:
    """Inspect required training capabilities without loading a policy model."""
    from .engine import SystemCapabilityProvider, required_capabilities

    detector = provider or SystemCapabilityProvider()
    capabilities = detector.detect(config)
    required = sorted(required_capabilities(config, "train"), key=lambda item: item.value)
    lines = tuple(_doctor_line(capabilities, capability) for capability in required)
    unavailable = any(
        capabilities.state(item) is not CapabilityState.SUPPORTED for item in required
    )
    return DoctorReport(lines=lines, exit_code=2 if unavailable else 0)


def _doctor_line(capabilities: BackendCapabilities, capability: Capability) -> str:
    evidence = capabilities.capabilities.get(capability)
    state = capabilities.state(capability)
    status = "AVAILABLE" if state is CapabilityState.SUPPORTED else "UNAVAILABLE"
    detail = evidence.evidence if evidence is not None else "No capability evidence is available."
    if (
        status == "UNAVAILABLE"
        and "install" not in detail.lower()
        and "configure" not in detail.lower()
    ):
        detail = f"{detail} Install the train extra or configure a supported local backend."
    value = getattr(capability, "value", str(capability))
    return f"{status} {value}: {detail}"


def _emit_result(result: _Result) -> None:
    print(json.dumps(result.to_dict(), allow_nan=False, sort_keys=True))


def _handle_engine(args: argparse.Namespace) -> int:
    from .engine import EngineDependencyError

    try:
        config = load_rl_run_config(args.config)
        engine = create_engine(config)
        if args.rl_command == "resume":
            result = engine.resume(Path(args.checkpoint))
        else:
            result = getattr(engine, args.rl_command)()
        _emit_result(result)
        return 0
    except (CapabilityError, EngineDependencyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _handle_doctor(args: argparse.Namespace) -> int:
    try:
        config = load_rl_run_config(args.config) if args.config else RLRunConfig()
        report = doctor_report(config)
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for line in report.lines:
        print(line)
    return report.exit_code


def register_rl_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register exactly the supported ``gepa rl`` modes."""
    parser = subparsers.add_parser("rl", help="Run portable local reinforcement learning")
    modes = parser.add_subparsers(dest="rl_command", required=True)

    for mode in ("train", "collect", "evaluate"):
        command = modes.add_parser(mode, help=f"{mode.capitalize()} with the canonical RL engine")
        command.add_argument("--config", required=True, help="Canonical RL config path")
        command.set_defaults(func=_handle_engine)

    resume = modes.add_parser("resume", help="Resume from an operator-selected checkpoint")
    resume.add_argument("--config", required=True, help="Canonical RL config path")
    resume.add_argument("--checkpoint", required=True, help="Checkpoint directory to resume")
    resume.set_defaults(func=_handle_engine)

    doctor = modes.add_parser("doctor", help="Check local RL capability availability")
    doctor.add_argument("--config", help="Optional canonical RL config path")
    doctor.set_defaults(func=_handle_doctor)


__all__ = [
    "DoctorReport",
    "create_engine",
    "doctor_report",
    "load_rl_run_config",
    "register_rl_cli",
]
