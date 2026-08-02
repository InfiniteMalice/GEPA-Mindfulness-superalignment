"""Lazy command-line entry points for the canonical RL engine."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .capability import (
    BackendCapabilities,
    Capability,
    CapabilityState,
    require_pure_mojo_learner,
)
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


def create_llama_cpp_engine(config: RLRunConfig, endpoint: str) -> RLTrainingEngine:
    """Create the canonical engine with a local inference-only llama.cpp actor."""
    from .engine import build_llama_cpp_engine

    return build_llama_cpp_engine(config, endpoint)


def create_hybrid_engine(
    config: RLRunConfig,
    coordinator_command: tuple[str, ...],
) -> RLTrainingEngine:
    """Create the experimental Mojo actor and PyTorch learner engine."""
    from .engine import build_hybrid_engine

    return build_hybrid_engine(config, coordinator_command)


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


def _llama_doctor_report(endpoint: str | None) -> DoctorReport:
    from .backends.llama_cpp_vulkan import detect_llama_cpp_runtime

    capabilities = detect_llama_cpp_runtime(endpoint=endpoint)
    inspected = (
        Capability.SUPPORTS_GENERATION,
        Capability.SUPPORTS_GGUF,
        Capability.SUPPORTS_VULKAN,
        Capability.SUPPORTS_BACKWARD,
        Capability.SUPPORTS_OPTIMIZER_STEP,
        Capability.SUPPORTS_VALUE_HEAD,
        Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
        Capability.SUPPORTS_LORA_TRAINING,
        Capability.SUPPORTS_DISTRIBUTED_TRAINING,
        Capability.SUPPORTS_MIXED_PRECISION,
    )
    endpoint_evidence = "configured" if endpoint is not None else "not configured"
    lines = (
        f"INFO backend: {capabilities.backend_name} {capabilities.backend_version}",
        f"INFO endpoint: {endpoint_evidence}",
        *(_doctor_line(capabilities, capability) for capability in inspected),
    )
    unavailable = any(
        capabilities.state(capability) is not CapabilityState.SUPPORTED for capability in inspected
    )
    return DoctorReport(lines=lines, exit_code=2 if unavailable else 0)


def _emit_result(result: _Result) -> None:
    print(json.dumps(result.to_dict(), allow_nan=False, sort_keys=True))


def _handle_engine(args: argparse.Namespace) -> int:
    try:
        learner = getattr(args, "learner", None)
        if learner == "mojo":
            require_pure_mojo_learner()
        requested_backend = getattr(args, "backend", None)
        coordinator_command = getattr(args, "coordinator_command", None)
        actor_endpoint = getattr(args, "actor_endpoint", None)
        if requested_backend == "mojo-vulkan-llamacpp" and learner != "pytorch":
            raise ValueError(
                "--backend mojo-vulkan-llamacpp requires explicit --learner pytorch; "
                "pure Mojo training is unsupported"
            )
        config = load_rl_run_config(args.config)
        selected_backend = requested_backend or config.runtime.backend
        if selected_backend != config.runtime.backend:
            raise ValueError(
                f"--backend {selected_backend!r} does not match configured "
                f"runtime.backend {config.runtime.backend!r}"
            )
        config = _apply_operator_overrides(config, args)
        if selected_backend == "mojo-vulkan-llamacpp":
            if learner != "pytorch":
                raise ValueError(
                    "hybrid training requires explicit --learner pytorch; "
                    "pure Mojo training is unsupported"
                )
            if args.rl_command not in {"train", "resume"}:
                raise ValueError("mojo-vulkan-llamacpp supports hybrid train/resume only")
            if args.endpoint is not None:
                raise ValueError("use --actor-endpoint for mojo-vulkan-llamacpp")
            command = coordinator_command
            if not command:
                raise ValueError("--coordinator-command is required for hybrid training")
            command_values = tuple(command)
            if actor_endpoint is not None:
                command_values = (*command_values, "--endpoint", actor_endpoint)
            engine = create_hybrid_engine(config, command_values)
        elif selected_backend == "llama-cpp-vulkan":
            if args.rl_command != "collect":
                raise ValueError("runtime.backend='llama-cpp-vulkan' is collection only")
            if args.endpoint is None:
                raise ValueError("--endpoint is required for llama-cpp-vulkan collection")
            engine = create_llama_cpp_engine(config, args.endpoint)
        else:
            if learner is not None:
                raise ValueError("--learner is supported only by mojo-vulkan-llamacpp")
            if coordinator_command or actor_endpoint is not None:
                raise ValueError("coordinator options require --backend mojo-vulkan-llamacpp")
            if args.endpoint is not None:
                raise ValueError("--endpoint requires --backend llama-cpp-vulkan")
            engine = create_engine(config)
        if args.rl_command == "resume":
            resume_kwargs = {}
            if args.max_steps is not None:
                resume_kwargs["max_steps"] = args.max_steps
            result = engine.resume(Path(args.checkpoint), **resume_kwargs)
        elif args.rl_command == "train":
            train_kwargs = {}
            if args.max_steps is not None:
                train_kwargs["max_steps"] = args.max_steps
            result = engine.train(**train_kwargs)
        else:
            result = getattr(engine, args.rl_command)()
        _emit_result(result)
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _apply_operator_overrides(config: RLRunConfig, args: argparse.Namespace) -> RLRunConfig:
    dataset_path = getattr(args, "dataset", None)
    output_path = getattr(args, "output", None)
    if dataset_path is not None:
        config = replace(config, dataset=replace(config.dataset, train_path=dataset_path))
    if output_path is not None:
        config = replace(config, logging=replace(config.logging, log_dir=output_path))
    return config


def _handle_doctor(args: argparse.Namespace) -> int:
    try:
        if args.backend == "llama-cpp-vulkan":
            report = _llama_doctor_report(args.endpoint)
        else:
            if args.endpoint is not None:
                raise ValueError("--endpoint requires --backend llama-cpp-vulkan")
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
        _add_engine_backend_arguments(command)
        if mode == "collect":
            command.add_argument("--dataset", help="Override dataset.train_path for this run")
            command.add_argument("--output", help="Override logging.log_dir for this run")
        if mode == "train":
            command.add_argument(
                "--max-steps",
                type=int,
                help="Relative optimizer-step budget for this invocation; zero disables rollout",
            )
        command.set_defaults(func=_handle_engine)

    resume = modes.add_parser("resume", help="Resume from an operator-selected checkpoint")
    resume.add_argument("--config", required=True, help="Canonical RL config path")
    resume.add_argument("--checkpoint", required=True, help="Checkpoint directory to resume")
    _add_engine_backend_arguments(resume)
    resume.add_argument(
        "--max-steps",
        type=int,
        help="Relative optimizer-step budget for this invocation; zero disables rollout",
    )
    resume.set_defaults(func=_handle_engine)

    doctor = modes.add_parser("doctor", help="Check local RL capability availability")
    doctor.add_argument("--config", help="Optional canonical RL config path")
    doctor.add_argument(
        "--backend",
        choices=("system", "llama-cpp-vulkan"),
        default="system",
        help="Diagnostic backend; llama.cpp probing is opt-in",
    )
    doctor.add_argument(
        "--endpoint",
        help="Optional loopback llama.cpp endpoint for doctor-only metadata probing",
    )
    doctor.set_defaults(func=_handle_doctor)


def _add_engine_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("pytorch", "cuda", "llama-cpp-vulkan", "mojo-vulkan-llamacpp"),
        help="Backend selector; when provided it must match runtime.backend",
    )
    parser.add_argument(
        "--endpoint",
        help="Loopback llama.cpp endpoint; required only for llama-cpp-vulkan collection",
    )
    parser.add_argument(
        "--learner",
        choices=("pytorch", "mojo"),
        help="Explicit hybrid learner; only pytorch is supported",
    )
    parser.add_argument(
        "--coordinator-command",
        nargs="+",
        help="Operator-supplied coordinator argv; executed directly without a shell",
    )
    parser.add_argument(
        "--actor-endpoint",
        help="Optional operator-supplied actor endpoint passed to the coordinator argv",
    )


__all__ = [
    "DoctorReport",
    "create_engine",
    "create_hybrid_engine",
    "create_llama_cpp_engine",
    "doctor_report",
    "load_rl_run_config",
    "register_rl_cli",
]
