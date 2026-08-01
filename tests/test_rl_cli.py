"""Canonical RL engine and ``gepa rl`` command contract tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gepa_mindfulness.training import engine as engine_module
from gepa_mindfulness.training import rl_cli
from gepa_mindfulness.training.capability import (
    BackendCapabilities,
    Capability,
    CapabilityError,
    CapabilityEvidence,
    CapabilityState,
)
from gepa_mindfulness.training.engine import (
    RLTrainingEngine,
    SystemCapabilityProvider,
    required_capabilities,
)
from gepa_mindfulness.training.runtime_config import (
    AlgorithmConfig,
    CheckpointConfig,
    DatasetConfig,
    DistributedRuntimeConfig,
    LoggingConfig,
    PolicyConfig,
    RewardConfig,
    RLRunConfig,
    RuntimeConfig,
)
from gepa_mindfulness.training.trajectory import (
    PolicyEvaluation,
    RolloutRequest,
    Trajectory,
    TrajectoryBatch,
)
from mindful_trace_gepa.cli import build_parser


def _config(tmp_path: Path, *, max_steps: int = 1) -> RLRunConfig:
    return RLRunConfig(
        algorithm=AlgorithmConfig(max_steps=max_steps),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "checkpoints"), save_steps=1),
    )


def _write_llama_collect_config(tmp_path: Path, *, backend: str) -> Path:
    path = tmp_path / f"{backend}.json"
    path.write_text(
        json.dumps(
            {
                "runtime": {"backend": backend, "device": "cpu", "precision": "fp32"},
                "policy": {"model_name": "server-selected-model"},
                "dataset": {"format": "text", "train_path": str(tmp_path / "prompts.txt")},
                "logging": {"log_dir": str(tmp_path / "logs"), "level": "INFO"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _capabilities(*, supported: bool) -> BackendCapabilities:
    state = CapabilityState.SUPPORTED if supported else CapabilityState.UNSUPPORTED
    return BackendCapabilities(
        backend_name="fake",
        backend_version="1",
        capabilities={
            capability: CapabilityEvidence(state=state, evidence=f"fake {state.value}")
            for capability in Capability
        },
    )


class _CapabilityProvider:
    def __init__(self, events: list[str], *, supported: bool = True) -> None:
        self.events = events
        self.supported = supported

    def detect(self, config: RLRunConfig) -> BackendCapabilities:
        self.events.append("capability")
        return _capabilities(supported=self.supported)


class _Backend:
    def __init__(self, events: list[str], *, fail_generate: bool = False) -> None:
        self.events = events
        self.fail_generate = fail_generate
        self.step = 0

    def capabilities(self) -> BackendCapabilities:
        self.events.append("backend.capabilities")
        return _capabilities(supported=True)

    def generate(self, requests: object) -> tuple[Trajectory, ...]:
        self.events.append("generate")
        if self.fail_generate:
            raise RuntimeError("rollout failed")
        request = tuple(requests)[0]
        return (
            Trajectory(
                trajectory_id=f"trajectory-{self.step}",
                case_id=request.case_id,
                prompt=request.prompt,
                response="observable response",
                prompt_token_ids=(1,),
                response_token_ids=(2,),
                old_log_probs=(-0.2,),
                reference_log_probs=(-0.3,),
                value_predictions=(0.1,),
                backend_name="fake",
                backend_version="1",
                model_identifier="fake-model",
                policy_version=f"policy-{self.step}",
            ),
        )

    def evaluate(self, batch: TrajectoryBatch) -> PolicyEvaluation:
        self.events.append("evaluate")
        return PolicyEvaluation(
            log_probs="log-probs",
            reference_log_probs="reference-log-probs",
            value_predictions="values",
            entropy="entropy",
        )

    def zero_grad(self) -> None:
        self.events.append("zero_grad")

    def backward(self, loss: object) -> None:
        self.events.append("backward")

    def optimizer_step(self) -> object:
        self.events.append("optimizer_step")
        self.step += 1
        return SimpleNamespace(step=self.step)

    def close(self) -> None:
        self.events.append("close")


class _Algorithm:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def required_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.SUPPORTS_GENERATION})

    def compute_loss(self, batch: TrajectoryBatch, evaluation: PolicyEvaluation) -> object:
        self.events.append("loss")
        return SimpleNamespace(total_loss="loss")


class _Dataset:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def materialize(self, mode: str) -> tuple[RolloutRequest, ...]:
        self.events.append("dataset")
        return (RolloutRequest(prompt="prompt", case_id="case-1"),)


class _Reward:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def score(self, request: object) -> float:
        self.events.append("reward")
        return 1.0


class _BatchPreparer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def prepare(
        self,
        trajectories: tuple[Trajectory, ...],
        rewards: tuple[object, ...],
        algorithm: object,
    ) -> TrajectoryBatch:
        self.events.append("batch")
        prepared = tuple(
            replace(trajectory, advantage=(1.0,), returns=(1.0,)) for trajectory in trajectories
        )
        return TrajectoryBatch(trajectories=prepared, response_token_masks=((True,),))


class _Checkpoint:
    def __init__(self, events: list[str], backend: _Backend) -> None:
        self.events = events
        self.backend = backend

    def load(self, path: Path) -> object:
        self.events.append(f"checkpoint.load:{path.name}")
        self.backend.step = 1
        return SimpleNamespace(
            global_step=1,
            manifest=SimpleNamespace(checkpoint_id="checkpoint-00000001"),
        )

    def save(self, global_step: int, parent_checkpoint: str | None) -> object:
        self.events.append(f"checkpoint.save:{global_step}:{parent_checkpoint}")
        return SimpleNamespace(checkpoint_id=f"checkpoint-{global_step:08d}")


class _Logger:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def start(
        self,
        mode: str,
        global_step: int,
        checkpoint_parent: str | None,
        capabilities: BackendCapabilities,
    ) -> None:
        self.events.append(f"log.start:{mode}:{global_step}:{checkpoint_parent}")

    def trajectories(self, values: tuple[Trajectory, ...], global_step: int) -> None:
        self.events.append(f"log.trajectories:{global_step}")

    def metrics(self, rewards: tuple[object, ...], global_step: int) -> None:
        self.events.append(f"log.metrics:{global_step}")


def _engine(
    tmp_path: Path,
    events: list[str],
    *,
    supported: bool = True,
    max_steps: int = 1,
    fail_generate: bool = False,
) -> RLTrainingEngine:
    return RLTrainingEngine(
        _config(tmp_path, max_steps=max_steps),
        capability_provider=_CapabilityProvider(events, supported=supported),
        backend_factory=lambda config: events.append("backend.factory")
        or _Backend(events, fail_generate=fail_generate),
        algorithm_factory=lambda config, backend: events.append("algorithm.factory")
        or _Algorithm(events),
        dataset_factory=lambda config: events.append("dataset.factory") or _Dataset(events),
        reward_factory=lambda config: events.append("reward.factory") or _Reward(events),
        batch_preparer_factory=lambda config: events.append("batch.factory")
        or _BatchPreparer(events),
        checkpoint_factory=lambda config, backend: events.append("checkpoint.factory")
        or _Checkpoint(events, backend),
        logger_factory=lambda config: events.append("logger.factory") or _Logger(events),
    )


@pytest.mark.parametrize("mode", ["train", "collect", "evaluate", "resume"])
def test_capabilities_fail_before_every_mutating_mode_factory(
    tmp_path: Path,
    mode: str,
) -> None:
    events: list[str] = []
    engine = _engine(tmp_path, events, supported=False)

    with pytest.raises(CapabilityError):
        if mode == "resume":
            engine.resume(tmp_path / "operator-checkpoint")
        else:
            getattr(engine, mode)()

    assert events == ["capability"]


def test_train_orders_lifecycle_and_closes_backend(tmp_path: Path) -> None:
    events: list[str] = []

    result = _engine(tmp_path, events).train()

    assert result.mode == "train"
    assert result.global_step == 1
    assert events[0] == "capability"
    assert events.index("dataset") < events.index("log.start:train:0:None")
    assert events.index("log.start:train:0:None") < events.index("generate")
    assert events.index("optimizer_step") < events.index("checkpoint.save:1:None")
    assert events[-1] == "close"


def test_rollout_failure_closes_backend_without_checkpoint_mutation(tmp_path: Path) -> None:
    events: list[str] = []

    with pytest.raises(RuntimeError, match="rollout failed"):
        _engine(tmp_path, events, fail_generate=True).train()

    assert not any(event.startswith("checkpoint.save") for event in events)
    assert events[-1] == "close"


def test_resume_uses_operator_path_parent_and_restored_global_step(tmp_path: Path) -> None:
    events: list[str] = []
    selected = tmp_path / "operator-checkpoint"

    result = _engine(tmp_path, events, max_steps=2).resume(selected)

    assert result.global_step == 3
    assert result.checkpoint_parent == "checkpoint-00000001"
    assert "checkpoint.load:operator-checkpoint" in events
    assert "log.start:resume:1:checkpoint-00000001" in events
    assert "checkpoint.save:2:checkpoint-00000001" in events
    assert "checkpoint.save:3:checkpoint-00000002" in events


@pytest.mark.parametrize("mode", ["collect", "evaluate"])
def test_nontraining_modes_preserve_global_step_and_close_backend(
    tmp_path: Path,
    mode: str,
) -> None:
    events: list[str] = []

    result = getattr(_engine(tmp_path, events), mode)()

    assert result.global_step == 0
    assert not any(event.startswith("checkpoint.save") for event in events)
    assert events[-1] == "close"


def test_registers_exact_required_rl_modes_and_requires_nested_command() -> None:
    parser = build_parser()
    for mode in ("doctor", "train", "collect", "evaluate"):
        suffix = [] if mode == "doctor" else ["--config", "config.json"]
        args = parser.parse_args(["rl", mode, *suffix])
        assert args.rl_command == mode
        assert callable(args.func)
    args = parser.parse_args(
        ["rl", "resume", "--config", "config.json", "--checkpoint", "checkpoint"]
    )
    assert args.rl_command == "resume"
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["rl"])
    assert caught.value.code == 2
    assert parser.parse_args(["score", "--trace", "t", "--out", "o"]).command == "score"


def test_cli_dispatch_and_capability_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_engine = SimpleNamespace(
        train=lambda: calls.append("train") or SimpleNamespace(to_dict=lambda: {"mode": "train"})
    )
    monkeypatch.setattr(rl_cli, "load_rl_run_config", lambda path: _config(tmp_path))
    monkeypatch.setattr(rl_cli, "create_engine", lambda config: fake_engine)
    parser = argparse.ArgumentParser()
    rl_cli.register_rl_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["rl", "train", "--config", "config.json"])

    assert args.func(args) == 0
    assert calls == ["train"]

    def unavailable() -> object:
        raise CapabilityError("required capability unavailable")

    fake_engine.train = unavailable  # type: ignore[method-assign]
    assert args.func(args) == 2


@pytest.mark.parametrize("mode", ["train", "resume", "evaluate"])
def test_llama_backend_rejects_noncollection_modes_before_actor_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    actor_calls: list[str] = []
    config_path = _write_llama_collect_config(tmp_path, backend="llama-cpp-vulkan")
    monkeypatch.setattr(
        rl_cli,
        "create_llama_cpp_engine",
        lambda config, endpoint: actor_calls.append(endpoint),
        raising=False,
    )
    command = [
        "rl",
        mode,
        "--config",
        str(config_path),
        "--backend",
        "llama-cpp-vulkan",
        "--endpoint",
        "http://127.0.0.1:8080",
    ]
    if mode == "resume":
        command.extend(["--checkpoint", str(tmp_path / "checkpoint")])
    args = build_parser().parse_args(command)

    assert args.func(args) == 2
    assert actor_calls == []
    assert "collection only" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("config_backend", "arguments", "message"),
    [
        (
            "llama-cpp-vulkan",
            ["--backend", "llama-cpp-vulkan"],
            "--endpoint is required",
        ),
        (
            "pytorch",
            [
                "--backend",
                "llama-cpp-vulkan",
                "--endpoint",
                "http://127.0.0.1:8080",
            ],
            "does not match",
        ),
    ],
)
def test_llama_collection_rejects_missing_endpoint_and_backend_conflict_before_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config_backend: str,
    arguments: list[str],
    message: str,
) -> None:
    actor_calls: list[str] = []
    config_path = _write_llama_collect_config(tmp_path, backend=config_backend)
    monkeypatch.setattr(
        rl_cli,
        "create_llama_cpp_engine",
        lambda config, endpoint: actor_calls.append(endpoint),
        raising=False,
    )
    args = build_parser().parse_args(["rl", "collect", "--config", str(config_path), *arguments])

    assert args.func(args) == 2
    assert actor_calls == []
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "cuda:0"},
        {"precision": "fp16"},
        {
            "backend": "llama-cpp-vulkan",
            "device": "cpu",
            "distributed": DistributedRuntimeConfig(
                strategy="ddp",
                world_size=2,
                rank=0,
                local_rank=0,
            ),
        },
    ],
)
def test_llama_runtime_rejects_non_cpu_fp32_single_process_combinations(
    changes: dict[str, object],
) -> None:
    values = {"backend": "llama-cpp-vulkan", "device": "cpu", "precision": "fp32"}
    values.update(changes)

    with pytest.raises(ValueError, match="llama-cpp-vulkan|distributed"):
        RuntimeConfig(**values)  # type: ignore[arg-type]


def test_shipped_llama_collection_config_has_no_endpoint() -> None:
    path = Path(__file__).parents[1] / "configs" / "rl" / "llama_cpp_vulkan_collect.yaml"

    config = rl_cli.load_rl_run_config(path)

    assert config.runtime == RuntimeConfig(backend="llama-cpp-vulkan")
    assert config.policy.model_name == "LOCAL_GGUF_MODEL_ID"
    assert config.dataset.train_path == "data/synthetic/reward_integrity/rl_pairs_v1.jsonl"


def test_default_engine_builder_rejects_llama_config_before_framework_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    framework_calls: list[str] = []
    config = RLRunConfig(runtime=RuntimeConfig(backend="llama-cpp-vulkan"))
    monkeypatch.setattr(
        engine_module,
        "_load_local_transformers_assets",
        lambda model_name: framework_calls.append(model_name),
    )

    with pytest.raises(ValueError, match="build_llama_cpp_engine.*endpoint"):
        engine_module.build_default_engine(config)

    assert framework_calls == []


def test_llama_collection_rejects_non_loopback_endpoint_before_actor_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gepa_mindfulness.training.backends import llama_cpp_vulkan

    config_path = _write_llama_collect_config(tmp_path, backend="llama-cpp-vulkan")
    actor_calls: list[str] = []
    monkeypatch.setattr(
        llama_cpp_vulkan,
        "LlamaCppVulkanBackend",
        lambda endpoint, **kwargs: actor_calls.append(endpoint),
    )
    args = build_parser().parse_args(
        [
            "rl",
            "collect",
            "--config",
            str(config_path),
            "--backend",
            "llama-cpp-vulkan",
            "--endpoint",
            "http://example.com:8080",
        ]
    )

    assert args.func(args) == 2
    assert actor_calls == []
    assert "local loopback" in capsys.readouterr().err


@pytest.mark.parametrize("preset", ["pytorch_cpu_ppo.yaml", "pytorch_cpu_grpo.yaml"])
def test_shipped_train_commands_reach_only_the_actionable_local_model_boundary(
    preset: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        rl_cli,
        "create_engine",
        lambda config: RLTrainingEngine(
            config,
            capability_provider=_CapabilityProvider(events),
        ),
    )
    parser = build_parser()
    config_path = Path(__file__).parents[1] / "configs" / "rl" / preset
    args = parser.parse_args(["rl", "train", "--config", str(config_path)])

    assert args.func(args) == 2
    error = capsys.readouterr().err
    assert "local" in error.lower()
    assert "network" in error.lower()
    assert "pair metadata" not in error


def test_doctor_is_model_free_deterministic_actionable_and_nonzero() -> None:
    events: list[str] = []
    provider = _CapabilityProvider(events, supported=False)

    first = rl_cli.doctor_report(RLRunConfig(), provider)
    second = rl_cli.doctor_report(RLRunConfig(), provider)

    assert first.exit_code == 2
    assert first.lines == second.lines
    assert any("UNAVAILABLE" in line for line in first.lines)
    assert any("Install" in line or "configure" in line for line in first.lines)
    assert events == ["capability", "capability"]


def test_doctor_lazily_renders_llama_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gepa_mindfulness.training.backends import llama_cpp_vulkan

    endpoint = "http://127.0.0.1:8080"
    calls: list[str | None] = []
    evidence = {
        capability: CapabilityEvidence(
            state=CapabilityState.UNKNOWN,
            evidence=f"unknown {capability.value}",
        )
        for capability in Capability
    }
    evidence[Capability.SUPPORTS_GENERATION] = CapabilityEvidence(
        state=CapabilityState.UNKNOWN,
        evidence="llama-server executable reported llama.cpp build 4242",
    )
    evidence[Capability.SUPPORTS_GGUF] = CapabilityEvidence(
        state=CapabilityState.SUPPORTED,
        evidence="trusted meta.format reported GGUF",
    )
    evidence[Capability.SUPPORTS_VULKAN] = CapabilityEvidence(
        state=CapabilityState.UNKNOWN,
        evidence="vulkaninfo was not found",
    )
    for capability in (
        Capability.SUPPORTS_BACKWARD,
        Capability.SUPPORTS_OPTIMIZER_STEP,
        Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
    ):
        evidence[capability] = CapabilityEvidence(
            state=CapabilityState.UNSUPPORTED,
            evidence="llama.cpp is inference-only",
        )
    report = BackendCapabilities(
        backend_name="llama_cpp_vulkan",
        backend_version="llama.cpp build 4242",
        capabilities=evidence,
    )
    monkeypatch.setattr(
        llama_cpp_vulkan,
        "detect_llama_cpp_runtime",
        lambda endpoint=None: calls.append(endpoint) or report,
    )
    parser = build_parser()
    args = parser.parse_args(
        ["rl", "doctor", "--backend", "llama-cpp-vulkan", "--endpoint", endpoint]
    )

    assert args.func(args) == 2
    output = capsys.readouterr().out
    assert calls == [endpoint]
    assert endpoint not in output
    assert "endpoint: configured" in output
    assert "llama-server executable" in output
    assert "supports_gguf" in output
    assert "supports_vulkan" in output
    assert "supports_backward" in output
    assert "supports_optimizer_step" in output


def test_llama_doctor_never_echoes_endpoint_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint = (
        "http://secret-user:secret-password@127.0.0.1:8080" "?token=secret-query#secret-fragment"
    )
    parser = build_parser()
    args = parser.parse_args(
        ["rl", "doctor", "--backend", "llama-cpp-vulkan", "--endpoint", endpoint]
    )

    assert args.func(args) == 2
    captured = capsys.readouterr()
    rendered = f"{captured.out}\n{captured.err}"
    assert "endpoint: configured" in captured.out
    for secret in (
        endpoint,
        "secret-user",
        "secret-password",
        "secret-query",
        "secret-fragment",
    ):
        assert secret not in rendered
    assert "credentials" in rendered


def test_basic_help_does_not_import_llama_diagnostics() -> None:
    code = (
        "import sys; "
        "from mindful_trace_gepa.cli import build_parser; "
        "build_parser().format_help(); "
        "print('gepa_mindfulness.training.backends.llama_cpp_vulkan' in sys.modules)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "False"


def test_training_requires_full_weight_capability() -> None:
    assert Capability.SUPPORTS_FULL_WEIGHT_TRAINING in required_capabilities(
        RLRunConfig(),
        "train",
    )


def test_basic_cli_help_does_not_import_training_frameworks() -> None:
    code = (
        "import sys; "
        "from mindful_trace_gepa.cli import build_parser; "
        "build_parser().format_help(); "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "False False"


def test_legacy_class_warning_points_to_user_callsite(tmp_path: Path) -> None:
    from gepa_mindfulness.training.configs import TrainingConfig
    from gepa_mindfulness.training.pipeline import (
        LightweightTrainingOrchestrator,
        TrainingOrchestrator,
    )

    config = TrainingConfig.from_mapping({})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TrainingOrchestrator(config)

    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert caught[0].filename == __file__
    with warnings.catch_warnings(record=True) as lightweight:
        warnings.simplefilter("always")
        LightweightTrainingOrchestrator(config)
    assert lightweight == []


def test_legacy_trainers_warn_while_explicit_lightweight_names_do_not(tmp_path: Path) -> None:
    from gepa_mindfulness.training.config import GRPOConfig, PPOConfig
    from gepa_mindfulness.training.grpo_trainer import GRPOTrainer, LightweightGRPOTrainer
    from gepa_mindfulness.training.ppo_trainer import LightweightPPOTrainer, PPOTrainer

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"prompt": "Be mindful", "answers": ["Breathe"]}\n', encoding="utf-8")
    ppo_config = PPOConfig(dataset_path=str(dataset), output_dir=str(tmp_path / "ppo"))
    grpo_config = GRPOConfig(dataset_path=str(dataset), output_dir=str(tmp_path / "grpo"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PPOTrainer(ppo_config)
        GRPOTrainer(grpo_config)
    assert [item.category for item in caught] == [DeprecationWarning, DeprecationWarning]
    assert all(item.filename == __file__ for item in caught)

    with warnings.catch_warnings(record=True) as explicit:
        warnings.simplefilter("always")
        LightweightPPOTrainer(ppo_config)
        LightweightGRPOTrainer(grpo_config)
    assert explicit == []


def test_injected_algorithm_extra_requirement_fails_before_every_factory(tmp_path: Path) -> None:
    events: list[str] = []
    capabilities = _capabilities(supported=True)
    evidence = dict(capabilities.capabilities)
    evidence[Capability.SUPPORTS_VULKAN] = CapabilityEvidence(
        state=CapabilityState.UNSUPPORTED,
        evidence="fake unsupported Vulkan",
    )
    report = BackendCapabilities("fake", "1", evidence)

    class Provider:
        def detect(self, config: RLRunConfig) -> BackendCapabilities:
            events.append("capability")
            return report

    class AlgorithmFactory:
        def required_capabilities(self, config: RLRunConfig) -> frozenset[Capability]:
            return frozenset({Capability.SUPPORTS_VULKAN})

        def __call__(self, config: RLRunConfig, backend: object) -> _Algorithm:
            events.append("algorithm.factory")
            return _Algorithm(events)

    engine = RLTrainingEngine(
        _config(tmp_path),
        capability_provider=Provider(),
        backend_factory=lambda config: events.append("backend.factory") or _Backend(events),
        algorithm_factory=AlgorithmFactory(),
        dataset_factory=lambda config: events.append("dataset.factory") or _Dataset(events),
        reward_factory=lambda config: events.append("reward.factory") or _Reward(events),
        batch_preparer_factory=lambda config: events.append("batch.factory")
        or _BatchPreparer(events),
        checkpoint_factory=lambda config, backend: events.append("checkpoint.factory")
        or _Checkpoint(events, backend),
        logger_factory=lambda config: events.append("logger.factory") or _Logger(events),
    )

    with pytest.raises(CapabilityError, match="supports_vulkan"):
        engine.train()

    assert events == ["capability"]


def test_invalid_cuda_index_fails_before_backend_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    config = replace(_config(tmp_path), runtime=RuntimeConfig(device="cuda:3"))
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    )
    monkeypatch.setattr(engine_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(engine_module, "import_module", lambda name: fake_torch)
    engine = RLTrainingEngine(
        config,
        capability_provider=SystemCapabilityProvider(),
        backend_factory=lambda value: events.append("backend.factory") or _Backend(events),
    )

    with pytest.raises(CapabilityError, match="supports_cuda"):
        engine.train()

    assert events == []


class _PairBackend(_Backend):
    def __init__(self, events: list[str], responses: dict[str, tuple[str, ...]]) -> None:
        super().__init__(events)
        self.responses = responses
        self.request_batches: list[tuple[RolloutRequest, ...]] = []
        self.random_observations: list[float] = []

    def generate(self, requests: object) -> tuple[Trajectory, ...]:
        request_batch = tuple(requests)
        self.request_batches.append(request_batch)
        self.random_observations.append(random.random())
        trajectories: list[Trajectory] = []
        for request in request_batch:
            source_id = request.metadata.get("source_case_id")
            values = self.responses.get(
                request.case_id or "",
                self.responses.get(str(source_id), ("fallback",)),
            )
            for index in range(request.num_samples):
                response = values[index % len(values)]
                trajectories.append(
                    Trajectory(
                        trajectory_id=f"{request.case_id}-{index}-{len(self.request_batches)}",
                        case_id=request.case_id,
                        prompt=request.prompt,
                        response=response,
                        prompt_token_ids=(1,),
                        response_token_ids=(2, 3),
                        old_log_probs=(-0.2, -0.2),
                        reference_log_probs=(-0.3, -0.3),
                        value_predictions=(0.1, 0.2),
                        backend_name="fake",
                        backend_version="1",
                        model_identifier="fake-model",
                        policy_version=request.policy_version,
                        seed=request.seed,
                    )
                )
        return tuple(trajectories)

    def parameter_checksum(self) -> str:
        return f"checksum-{self.step}"


def _pair_config(tmp_path: Path, rows: list[dict[str, object]], **algorithm: object) -> RLRunConfig:
    component_names = (
        "objective_fidelity",
        "feedback_integrity",
        "skill_transfer",
        "reality_contact",
        "exploit_disclosure",
        "long_horizon_agency",
        "benign_creativity",
        "repair_quality",
    )
    normalized_rows: list[dict[str, object]] = []
    for line, row in enumerate(rows, start=1):
        source_case_id = str(row["id"])
        chosen_components = {name: 0.5 for name in component_names}
        rejected_components = {name: -0.5 for name in component_names}
        normalized_rows.append(
            {
                "record_id": f"{source_case_id}:grounded_over_proxy",
                "source_case_id": source_case_id,
                "source_case_version": "1.0",
                "source_path": "authored/source.jsonl",
                "source_line": line,
                "source_sha256": "a" * 64,
                "pair_rule": "grounded_over_proxy",
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "chosen_class": "grounded_success",
                "rejected_class": "proxy_exploitation",
                "chosen_reward_components": chosen_components,
                "rejected_reward_components": rejected_components,
                "diagnostics": {"central": "authored", "supporting": []},
                "schema_version": "reward-integrity-rl-pairs-v1",
            }
        )
    dataset = tmp_path / "pairs.jsonl"
    dataset.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    return RLRunConfig(
        algorithm=AlgorithmConfig(**algorithm),
        dataset=DatasetConfig(train_path=str(dataset)),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "checkpoints"), save_steps=1),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs")),
    )


def test_grpo_rollout_requests_propagate_validated_stochastic_generation(tmp_path: Path) -> None:
    config = RLRunConfig(
        policy=PolicyConfig(do_sample=True, temperature=0.7, top_p=0.9),
        algorithm=AlgorithmConfig(name="grpo", group_size=3),
    )
    engine = RLTrainingEngine(config)

    selected = engine._rollout_requests((RolloutRequest(prompt="p"),), 2, rollout_index=4)

    assert selected[0].num_samples == 3
    assert selected[0].sampling_parameters == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }


def test_pair_reward_keeps_base_alignment_separate_from_integrity_overlay(tmp_path: Path) -> None:
    rows = [{"id": "chosen", "prompt": "p", "chosen": "good", "rejected": "bad"}]
    base_dir = tmp_path / "base"
    enabled_dir = tmp_path / "enabled"
    base_dir.mkdir()
    enabled_dir.mkdir()
    base_config = _pair_config(base_dir, rows, max_steps=1)
    enabled_config = replace(
        _pair_config(enabled_dir, rows, max_steps=1),
        reward=RewardConfig(overlay_weight=0.5, integrity_overlay_enabled=True),
    )

    def evaluate(config: RLRunConfig) -> Trajectory:
        events: list[str] = []
        return (
            RLTrainingEngine(
                config,
                capability_provider=_CapabilityProvider(events),
                backend_factory=lambda value: _PairBackend(events, {"chosen": ("good",)}),
                logger_factory=lambda value: _Logger(events),
            )
            .evaluate()
            .trajectories[0]
        )

    default = evaluate(base_config)
    enabled = evaluate(enabled_config)
    integrity_names = {
        "objective_fidelity",
        "feedback_integrity",
        "skill_transfer",
        "reality_contact",
        "exploit_disclosure",
        "long_horizon_agency",
        "benign_creativity",
        "repair_quality",
    }

    assert (
        enabled.reward_components["gepa_alignment"] == default.reward_components["gepa_alignment"]
    )
    assert enabled.reward_components["reward_integrity_aggregate"] == pytest.approx(0.5)
    assert integrity_names.issubset(enabled.reward_components)
    assert default.reward_total is not None and enabled.reward_total is not None
    assert enabled.reward_total == pytest.approx(default.reward_total + 0.25)


def test_unmatched_pair_response_logs_all_integrity_components_with_overlay_off(
    tmp_path: Path,
) -> None:
    rows = [{"id": "other", "prompt": "p", "chosen": "good", "rejected": "bad"}]
    config = _pair_config(tmp_path, rows, max_steps=1)
    events: list[str] = []
    trajectory = (
        RLTrainingEngine(
            config,
            capability_provider=_CapabilityProvider(events),
            backend_factory=lambda value: _PairBackend(events, {"other": ("neither",)}),
            logger_factory=lambda value: _Logger(events),
        )
        .evaluate()
        .trajectories[0]
    )

    assert trajectory.reward_components["reward_integrity_aggregate"] == 0.0
    integrity_names = {
        "objective_fidelity",
        "feedback_integrity",
        "skill_transfer",
        "reality_contact",
        "exploit_disclosure",
        "long_horizon_agency",
        "benign_creativity",
        "repair_quality",
    }
    assert integrity_names.issubset(trajectory.reward_components)


def test_default_pair_reward_scores_and_binds_observable_components(tmp_path: Path) -> None:
    rows = [
        {"id": "chosen", "prompt": "p1", "chosen": "good", "rejected": "bad"},
        {"id": "rejected", "prompt": "p2", "chosen": "good", "rejected": "bad"},
    ]
    config = _pair_config(tmp_path, rows, max_steps=1)
    events: list[str] = []
    backend = _PairBackend(events, {"chosen": ("good",), "rejected": ("bad",)})
    engine = RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        logger_factory=lambda value: _Logger(events),
    )

    result = engine.evaluate()

    assert len(result.trajectories) == 2
    chosen, rejected = result.trajectories
    assert chosen.reward_total is not None and rejected.reward_total is not None
    assert chosen.reward_total > rejected.reward_total
    assert chosen.reward_components["task_success"] == 1.0
    assert rejected.reward_components["hallucination"] < 0.0
    assert rejected.reward_component_evidence["hallucination"]
    assert all(reference.is_observable for reference in rejected.evidence_references)


def test_default_dataset_is_one_immutable_snapshot_per_invocation(tmp_path: Path) -> None:
    original = {"id": "original", "prompt": "original", "chosen": "yes", "rejected": "no"}
    replacement = {"id": "changed", "prompt": "changed", "chosen": "up", "rejected": "down"}
    config = _pair_config(tmp_path, [original], max_steps=1)
    dataset_path = Path(config.dataset.train_path)
    original_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    events: list[str] = []
    backend = _PairBackend(events, {"original": ("yes",)})

    def backend_factory(value: RLRunConfig) -> _PairBackend:
        dataset_path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")
        return backend

    result = RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=backend_factory,
        logger_factory=lambda value: _Logger(events),
    ).collect()

    assert result.trajectories[0].case_id == "original:grounded_over_proxy"
    assert result.dataset_hash == original_hash


class _CaptureAlgorithm(_Algorithm):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.batches: list[TrajectoryBatch] = []

    def compute_group_advantages(self, rewards: object) -> list[float] | None:
        values = tuple(float(value) for value in rewards)
        if len(set(values)) == 1:
            return None
        mean = sum(values) / len(values)
        return [value - mean for value in values]

    def compute_loss(self, batch: TrajectoryBatch, evaluation: PolicyEvaluation) -> object:
        self.batches.append(batch)
        return SimpleNamespace(
            total_loss=2.0,
            policy_loss=1.0,
            value_loss=0.5,
            entropy=0.25,
            kl=0.125,
        )


class _ResponseReward:
    def score(self, request: object) -> float:
        return float(request.trajectory.response)


def test_grpo_preserves_groups_skips_zero_variance_and_accumulates(tmp_path: Path) -> None:
    config = _config(tmp_path, max_steps=1)
    config = replace(
        config,
        algorithm=replace(
            config.algorithm,
            name="grpo",
            batch_size=1,
            gradient_accumulation_steps=2,
            group_size=2,
            zero_variance_policy="skip",
        ),
    )
    events: list[str] = []
    backend = _PairBackend(events, {"variable": ("1", "3"), "constant": ("2", "2")})
    algorithm = _CaptureAlgorithm(events)

    class Dataset:
        def materialize(self, mode: str) -> tuple[RolloutRequest, ...]:
            return (
                RolloutRequest(prompt="p1", case_id="variable"),
                RolloutRequest(prompt="p2", case_id="constant"),
            )

    result = RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        algorithm_factory=lambda value, selected: algorithm,
        dataset_factory=lambda value: Dataset(),
        reward_factory=lambda value: _ResponseReward(),
        checkpoint_factory=lambda value, selected: _Checkpoint(events, selected),
        logger_factory=lambda value: _Logger(events),
    ).train()

    assert [batch[0].num_samples for batch in backend.request_batches] == [2, 2, 2]
    assert len(algorithm.batches) == 2
    prepared = algorithm.batches[0].trajectories
    assert [trajectory.case_id for trajectory in prepared] == ["variable", "variable"]
    assert [trajectory.advantage[0] for trajectory in prepared] == [-1.0, 1.0]
    assert events.count("backward") == 2
    assert events.count("optimizer_step") == 1
    assert result.global_step == 1


def test_ppo_terminal_rewards_gae_batching_and_gradient_accumulation(tmp_path: Path) -> None:
    config = _config(tmp_path, max_steps=1)
    config = replace(
        config,
        algorithm=replace(
            config.algorithm,
            batch_size=1,
            gradient_accumulation_steps=2,
        ),
    )
    events: list[str] = []
    backend = _PairBackend(events, {"one": ("1",), "two": ("1",)})
    algorithm = _CaptureAlgorithm(events)

    class Dataset:
        def materialize(self, mode: str) -> tuple[RolloutRequest, ...]:
            return (
                RolloutRequest(prompt="p1", case_id="one"),
                RolloutRequest(prompt="p2", case_id="two"),
            )

    RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        algorithm_factory=lambda value, selected: algorithm,
        dataset_factory=lambda value: Dataset(),
        reward_factory=lambda value: _ResponseReward(),
        checkpoint_factory=lambda value, selected: _Checkpoint(events, selected),
        logger_factory=lambda value: _Logger(events),
    ).train()

    assert [len(batch) for batch in backend.request_batches] == [1, 1]
    assert len(algorithm.batches) == 2
    trajectory = algorithm.batches[0].trajectories[0]
    assert trajectory.advantage == pytest.approx((0.8504, 0.8))
    assert trajectory.returns == pytest.approx((0.9504, 1.0))
    assert algorithm.batches[0].response_token_masks == ((True, True),)
    assert events.count("backward") == 2
    assert events.count("optimizer_step") == 1


def test_seed_is_set_before_backend_and_resume_restores_rng_before_rollout(tmp_path: Path) -> None:
    events: list[str] = []
    observed_factory_random: list[float] = []
    backend = _PairBackend(events, {"case-1": ("1",)})
    expected_seeded = random.Random(42).random()

    def backend_factory(value: RLRunConfig) -> _PairBackend:
        observed_factory_random.append(random.random())
        return backend

    engine = _engine(tmp_path, events)
    engine.backend_factory = backend_factory
    engine.train()
    assert observed_factory_random == [expected_seeded]

    restored_rng = random.Random(8675309)
    expected_restored = restored_rng.random()
    restored_state = random.Random(8675309).getstate()

    class RestoringCheckpoint(_Checkpoint):
        def load(self, path: Path) -> object:
            result = super().load(path)
            return SimpleNamespace(
                **vars(result),
                python_rng_state=restored_state,
                torch_cpu_rng_state=None,
                torch_cuda_rng_states=(),
                algorithm_state={"restored": True},
                scheduler_state=None,
            )

    resume_backend = _PairBackend(events, {"case-1": ("1",)})
    resume_engine = _engine(tmp_path, events, max_steps=2)
    resume_engine.backend_factory = lambda value: resume_backend
    resume_engine.algorithm_factory = lambda value, selected: SimpleNamespace(
        required_capabilities=lambda: frozenset({Capability.SUPPORTS_GENERATION}),
        load_state_dict=lambda state: events.append("algorithm.restore"),
        compute_loss=lambda batch, evaluation: SimpleNamespace(total_loss="loss"),
    )
    resume_engine.checkpoint_factory = lambda value, selected: RestoringCheckpoint(events, selected)
    resume_engine.resume(tmp_path / "operator-checkpoint")
    assert resume_backend.random_observations[0] == expected_restored


def test_result_and_logging_keep_update_evidence_lineage_and_scored_rollouts(
    tmp_path: Path,
) -> None:
    config = _pair_config(
        tmp_path,
        [{"id": "case", "prompt": "p", "chosen": "1", "rejected": "0"}],
        max_steps=2,
    )
    events: list[str] = []
    backend = _PairBackend(events, {"case": ("1",)})
    result = RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        algorithm_factory=lambda value, selected: _CaptureAlgorithm(events),
        checkpoint_factory=lambda value, selected: _Checkpoint(events, selected),
    ).train(max_steps=1)

    assert result.global_step == 1
    assert result.checkpoint is not None
    assert result.parameter_checksum_before == "checksum-0"
    assert result.parameter_checksum_after == "checksum-1"
    assert result.parameters_updated is True
    assert result.log_directory is not None and result.log_directory.parent == Path(
        config.logging.log_dir
    )
    assert result.run_id
    assert result.trajectories[0].reward_total is not None
    assert result.trajectories[0].policy_version == "policy-0"
    json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)
    manifest = json.loads((result.log_directory / "run_manifest.json").read_text())
    assert manifest["checkpoint_parent"] is None
    metrics = [
        json.loads(line)
        for line in (result.log_directory / "metrics.jsonl").read_text().splitlines()
    ]
    names = {name for record in metrics for name in record["metrics"]}
    assert {"policy_loss", "value_loss", "entropy", "kl", "learning_rate"} <= names


def test_output_controls_do_not_change_checkpoint_compatibility_hash(tmp_path: Path) -> None:
    first = _config(tmp_path, max_steps=1)
    second = replace(
        first,
        algorithm=replace(first.algorithm, max_steps=99),
        checkpoint=replace(first.checkpoint, output_dir=str(tmp_path / "elsewhere")),
        logging=replace(first.logging, log_dir=str(tmp_path / "other-logs")),
    )
    assert engine_module._config_hash(first) == engine_module._config_hash(second)


def test_unique_child_log_directories_and_expected_cli_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(
        tmp_path,
        [{"id": "case", "prompt": "p", "chosen": "yes", "rejected": "no"}],
        max_steps=1,
    )
    results = []
    for _ in range(2):
        events: list[str] = []
        backend = _PairBackend(events, {"case": ("yes",)})
        results.append(
            RLTrainingEngine(
                config,
                capability_provider=_CapabilityProvider(events),
                backend_factory=lambda value, selected=backend: selected,
            ).collect()
        )
    assert results[0].log_directory != results[1].log_directory
    assert all(result.log_directory.parent == Path(config.logging.log_dir) for result in results)

    monkeypatch.setattr(
        rl_cli,
        "load_rl_run_config",
        lambda path: (_ for _ in ()).throw(FileNotFoundError("missing config.json")),
    )
    parser = argparse.ArgumentParser()
    rl_cli.register_rl_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["rl", "train", "--config", "missing.json"])
    assert args.func(args) == 2


def test_close_failure_preserves_primary_rollout_error(tmp_path: Path) -> None:
    events: list[str] = []

    class BrokenCloseBackend(_Backend):
        def generate(self, requests: object) -> tuple[Trajectory, ...]:
            raise RuntimeError("primary rollout failure")

        def close(self) -> None:
            raise RuntimeError("secondary close failure")

    engine = _engine(tmp_path, events)
    engine.backend_factory = lambda value: BrokenCloseBackend(events)

    with pytest.raises(RuntimeError, match="primary rollout failure") as caught:
        engine.collect()

    assert any("secondary close failure" in note for note in getattr(caught.value, "__notes__", ()))


@pytest.mark.parametrize("selector", ["cuda:-1", "cuda:", "cuda:x", "gpu:0"])
def test_runtime_config_direct_construction_rejects_invalid_device_selectors(
    selector: str,
) -> None:
    with pytest.raises(ValueError, match="runtime.device"):
        RuntimeConfig(device=selector)


@pytest.mark.parametrize("selector", ["cuda:-1", "cuda:"])
def test_malformed_cuda_selector_fails_preflight_before_seed_or_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
) -> None:
    events: list[str] = []
    runtime = object.__new__(RuntimeConfig)
    object.__setattr__(runtime, "backend", "pytorch")
    object.__setattr__(runtime, "device", selector)
    config = replace(_config(tmp_path), runtime=runtime)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    )
    monkeypatch.setattr(engine_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(engine_module, "import_module", lambda name: fake_torch)
    monkeypatch.setattr(
        engine_module,
        "_seed_process",
        lambda value: events.append("seed"),
    )
    engine = RLTrainingEngine(
        config,
        capability_provider=SystemCapabilityProvider(),
        backend_factory=lambda value: events.append("backend.factory") or _Backend(events),
        dataset_factory=lambda value: events.append("dataset.factory") or _Dataset(events),
    )

    with pytest.raises(CapabilityError, match="supports_cuda"):
        engine.train()

    assert events == []


@pytest.mark.parametrize("fault", ["missing", "extra", "unknown_case", "wrong_prompt"])
def test_grpo_rejects_backend_outputs_that_break_requested_group_identity(
    tmp_path: Path,
    fault: str,
) -> None:
    events: list[str] = []
    config = replace(
        _config(tmp_path),
        algorithm=AlgorithmConfig(name="grpo", group_size=2, max_steps=1),
    )

    class FaultyBackend(_PairBackend):
        def generate(self, requests: object) -> tuple[Trajectory, ...]:
            generated = super().generate(requests)
            if fault == "missing":
                return generated[:1]
            if fault == "extra":
                return generated + (replace(generated[0], trajectory_id="unexpected-extra"),)
            if fault == "unknown_case":
                return (replace(generated[0], case_id="unknown"), generated[1])
            return (replace(generated[0], prompt="wrong prompt"), generated[1])

    class TrackingReward(_ResponseReward):
        def score(self, request: object) -> float:
            events.append("reward.score")
            return super().score(request)

    backend = FaultyBackend(events, {"case": ("1", "2")})
    engine = RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        algorithm_factory=lambda value, selected: _CaptureAlgorithm(events),
        dataset_factory=lambda value: SimpleNamespace(
            materialize=lambda mode: (RolloutRequest(prompt="prompt", case_id="case"),)
        ),
        reward_factory=lambda value: TrackingReward(),
        checkpoint_factory=lambda value, selected: _Checkpoint(events, selected),
        logger_factory=lambda value: _Logger(events),
    )

    with pytest.raises(ValueError, match="GRPO backend output"):
        engine.train()

    assert "reward.score" not in events
    assert "backward" not in events


def test_checkpoint_compatibility_hash_keeps_format_but_ignores_dataset_paths(
    tmp_path: Path,
) -> None:
    first = replace(
        _config(tmp_path),
        dataset=DatasetConfig(
            train_path=str(tmp_path / "one.jsonl"),
            validation_path=str(tmp_path / "one-validation.jsonl"),
            format="jsonl",
        ),
    )
    moved = replace(
        first,
        dataset=replace(
            first.dataset,
            train_path=str(tmp_path / "moved.jsonl"),
            validation_path=str(tmp_path / "moved-validation.jsonl"),
        ),
    )
    different_format = replace(moved, dataset=replace(moved.dataset, format="text"))

    assert engine_module._config_hash(first) == engine_module._config_hash(moved)
    assert engine_module._config_hash(first) != engine_module._config_hash(different_format)


@pytest.mark.parametrize("state_kind", ["algorithm", "scheduler"])
def test_resume_rejects_nonempty_training_state_without_compatible_loader_before_rollout(
    tmp_path: Path,
    state_kind: str,
) -> None:
    events: list[str] = []

    class AlgorithmWithOnlyStateLoader(_Algorithm):
        def load_state_dict(self, state: object) -> None:
            events.append("algorithm.restore")

    class StatefulCheckpoint(_Checkpoint):
        def load(self, path: Path) -> object:
            restored = super().load(path)
            return SimpleNamespace(
                **vars(restored),
                algorithm_state={"updates": 1} if state_kind == "algorithm" else {},
                scheduler_state={"epoch": 1} if state_kind == "scheduler" else None,
                python_rng_state=None,
                torch_cpu_rng_state=None,
                torch_cuda_rng_states=(),
            )

    engine = _engine(tmp_path, events, max_steps=2)
    engine.algorithm_factory = lambda value, backend: (
        _Algorithm(events) if state_kind == "algorithm" else AlgorithmWithOnlyStateLoader(events)
    )
    engine.checkpoint_factory = lambda value, backend: StatefulCheckpoint(events, backend)

    with pytest.raises(ValueError, match=f"{state_kind}.*load_state_dict"):
        engine.resume(tmp_path / "checkpoint")

    assert "generate" not in events


def test_resume_rejects_empty_scheduler_mapping_without_scheduler_before_rollout(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class EmptySchedulerCheckpoint(_Checkpoint):
        def load(self, path: Path) -> object:
            restored = super().load(path)
            return SimpleNamespace(
                **vars(restored),
                algorithm_state={},
                scheduler_state={},
                python_rng_state=None,
                torch_cpu_rng_state=None,
                torch_cuda_rng_states=(),
            )

    engine = _engine(tmp_path, events)
    engine.checkpoint_factory = lambda value, backend: EmptySchedulerCheckpoint(events, backend)

    with pytest.raises(ValueError, match="scheduler.*load_state_dict"):
        engine.resume(tmp_path / "checkpoint")

    assert "generate" not in events


def test_resume_accepts_empty_algorithm_state_and_absent_scheduler(tmp_path: Path) -> None:
    events: list[str] = []

    class StatelessCheckpoint(_Checkpoint):
        def load(self, path: Path) -> object:
            restored = super().load(path)
            return SimpleNamespace(
                **vars(restored),
                algorithm_state={},
                scheduler_state=None,
                python_rng_state=None,
                torch_cpu_rng_state=None,
                torch_cuda_rng_states=(),
            )

    engine = _engine(tmp_path, events)
    engine.checkpoint_factory = lambda value, backend: StatelessCheckpoint(events, backend)

    result = engine.resume(tmp_path / "checkpoint")

    assert result.global_step == 2
    assert "generate" in events


def test_resume_restores_algorithm_then_scheduler_before_rollout(tmp_path: Path) -> None:
    events: list[str] = []

    class Scheduler:
        def load_state_dict(self, state: object) -> None:
            events.append("scheduler.restore")

    class StatefulAlgorithm(_Algorithm):
        def __init__(self) -> None:
            super().__init__(events)
            self.scheduler = Scheduler()

        def load_state_dict(self, state: object) -> None:
            events.append("algorithm.restore")

    class StatefulCheckpoint(_Checkpoint):
        def load(self, path: Path) -> object:
            restored = super().load(path)
            return SimpleNamespace(
                **vars(restored),
                algorithm_state={"updates": 1},
                scheduler_state={"epoch": 1},
                python_rng_state=None,
                torch_cpu_rng_state=None,
                torch_cuda_rng_states=(),
            )

    engine = _engine(tmp_path, events, max_steps=2)
    engine.algorithm_factory = lambda value, backend: StatefulAlgorithm()
    engine.checkpoint_factory = lambda value, backend: StatefulCheckpoint(events, backend)

    engine.resume(tmp_path / "checkpoint")

    assert events.index("algorithm.restore") < events.index("scheduler.restore")
    assert events.index("scheduler.restore") < events.index("generate")


def test_resume_rolls_back_backend_algorithm_scheduler_and_rng_as_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class TransactionalBackend(_Backend):
        def capture_checkpoint_restore_state(self) -> int:
            return self.step

        def rollback_checkpoint_restore_state(self, state: object) -> None:
            self.step = int(state)

    class Scheduler:
        def __init__(self) -> None:
            self.epoch = 0

        def state_dict(self) -> dict[str, int]:
            return {"epoch": self.epoch}

        def load_state_dict(self, state: object) -> None:
            self.epoch = int(state["epoch"])

    class StatefulAlgorithm(_Algorithm):
        def __init__(self) -> None:
            super().__init__(events)
            self.updates = 0
            self.scheduler = Scheduler()

        def state_dict(self) -> dict[str, int]:
            return {"updates": self.updates}

        def load_state_dict(self, state: object) -> None:
            self.updates = int(state["updates"])

    class FailingCUDA:
        def set_rng_state_all(self, states: object) -> None:
            if tuple(states):
                raise RuntimeError("late CUDA RNG restore failed")

        def get_rng_state_all(self) -> list[object]:
            return []

    class FakeTorch:
        def __init__(self) -> None:
            self.state = "torch-before"
            self.cuda = FailingCUDA()

        def get_rng_state(self) -> str:
            return self.state

        def set_rng_state(self, state: object) -> None:
            self.state = str(state)

        def manual_seed(self, seed: int) -> None:
            self.state = f"seed-{seed}"

    fake_torch = FakeTorch()
    real_import = engine_module.import_module
    monkeypatch.setattr(
        engine_module,
        "import_module",
        lambda name: fake_torch if name == "torch" else real_import(name),
    )
    backend = TransactionalBackend(events)
    algorithm = StatefulAlgorithm()
    python_before = random.Random(42).getstate()

    class StatefulCheckpoint(_Checkpoint):
        def load(self, path: Path) -> object:
            restored = super().load(path)
            return SimpleNamespace(
                **vars(restored),
                algorithm_state={"updates": 9},
                scheduler_state={"epoch": 8},
                python_rng_state=random.Random(999).getstate(),
                torch_cpu_rng_state="torch-restored",
                torch_cuda_rng_states=("cuda-restored",),
            )

    engine = RLTrainingEngine(
        _config(tmp_path),
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        algorithm_factory=lambda value, selected: algorithm,
        dataset_factory=lambda value: _Dataset(events),
        reward_factory=lambda value: _Reward(events),
        batch_preparer_factory=lambda value: _BatchPreparer(events),
        checkpoint_factory=lambda value, selected: StatefulCheckpoint(events, backend),
        logger_factory=lambda value: _Logger(events),
    )

    with pytest.raises(RuntimeError, match="late CUDA RNG restore failed"):
        engine.resume(tmp_path / "checkpoint")

    assert backend.step == 0
    assert algorithm.updates == 0
    assert algorithm.scheduler.epoch == 0
    assert random.getstate() == python_before
    assert fake_torch.state == "seed-42"


def test_all_skipped_grpo_groups_still_emit_group_and_response_evidence(tmp_path: Path) -> None:
    config = _pair_config(
        tmp_path,
        [{"id": "constant", "prompt": "p", "chosen": "1", "rejected": "0"}],
        name="grpo",
        group_size=2,
        zero_variance_policy="skip",
        max_steps=1,
    )
    events: list[str] = []
    backend = _PairBackend(events, {"constant": ("1", "1")})
    result = RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        algorithm_factory=lambda value, selected: _CaptureAlgorithm(events),
        reward_factory=lambda value: _ResponseReward(),
        checkpoint_factory=lambda value, selected: _Checkpoint(events, selected),
    ).train()

    assert result.global_step == 0
    assert result.log_directory is not None
    metrics = [
        json.loads(line)
        for line in (result.log_directory / "metrics.jsonl").read_text().splitlines()
    ]
    group_records = [record for record in metrics if record["scope"] == "group"]
    assert len(group_records) == 1
    assert group_records[0]["metrics"]["group_skipped"] == 1.0
    names = {name for record in metrics for name in record["metrics"]}
    assert "response_reward" in names
    assert "optimizer_step" not in names
    assert "policy_loss" not in names
    trajectories = (result.log_directory / "trajectories.jsonl").read_text().splitlines()
    assert len(trajectories) == 2


def test_repeated_preupdate_grpo_rollouts_get_unique_log_record_ids(tmp_path: Path) -> None:
    config = _pair_config(
        tmp_path,
        [{"id": "repeat", "prompt": "p", "chosen": "2", "rejected": "1"}],
        name="grpo",
        group_size=2,
        gradient_accumulation_steps=2,
        max_steps=1,
    )
    events: list[str] = []

    class StableIdentityBackend(_PairBackend):
        def generate(self, requests: object) -> tuple[Trajectory, ...]:
            generated = super().generate(requests)
            return tuple(
                replace(trajectory, trajectory_id=f"stable-{index}")
                for index, trajectory in enumerate(generated)
            )

    backend = StableIdentityBackend(events, {"repeat": ("1", "2")})
    result = RLTrainingEngine(
        config,
        capability_provider=_CapabilityProvider(events),
        backend_factory=lambda value: backend,
        algorithm_factory=lambda value, selected: _CaptureAlgorithm(events),
        reward_factory=lambda value: _ResponseReward(),
        checkpoint_factory=lambda value, selected: _Checkpoint(events, selected),
    ).train()

    assert result.log_directory is not None
    records = [
        json.loads(line)
        for line in (result.log_directory / "trajectories.jsonl").read_text().splitlines()
    ]
    record_ids = [record["record_id"] for record in records]
    assert len(record_ids) == 4
    assert len(set(record_ids)) == 4


def test_resume_max_steps_is_relative_to_restored_global_step(tmp_path: Path) -> None:
    events: list[str] = []

    class StepFiveCheckpoint(_Checkpoint):
        def load(self, path: Path) -> object:
            self.events.append(f"checkpoint.load:{path.name}")
            self.backend.step = 5
            return SimpleNamespace(
                global_step=5,
                manifest=SimpleNamespace(checkpoint_id="checkpoint-00000005"),
                algorithm_state={},
                scheduler_state=None,
                python_rng_state=None,
                torch_cpu_rng_state=None,
                torch_cuda_rng_states=(),
            )

    engine = _engine(tmp_path, events, max_steps=99)
    engine.checkpoint_factory = lambda value, backend: StepFiveCheckpoint(events, backend)

    result = engine.resume(tmp_path / "checkpoint", max_steps=2)

    assert result.global_step == 7
    assert events.count("optimizer_step") == 2
    assert "checkpoint.save:7:checkpoint-00000006" in events


def test_zero_invocation_step_budget_performs_no_rollout_or_update(tmp_path: Path) -> None:
    events: list[str] = []

    result = _engine(tmp_path, events, max_steps=9).train(max_steps=0)

    assert result.global_step == 0
    assert "generate" not in events
    assert "optimizer_step" not in events


def test_configured_step_budget_must_remain_positive(tmp_path: Path) -> None:
    events: list[str] = []
    engine = _engine(tmp_path, events)
    with pytest.raises(ValueError, match="algorithm.max_steps"):
        replace(engine.config.algorithm, max_steps=0)

    assert events == []


def test_cli_reports_operational_runtime_errors_without_swallowing_base_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_engine = SimpleNamespace(
        train=lambda: (_ for _ in ()).throw(RuntimeError("generation failed operationally"))
    )
    monkeypatch.setattr(rl_cli, "load_rl_run_config", lambda path: _config(tmp_path))
    monkeypatch.setattr(rl_cli, "create_engine", lambda config: fake_engine)
    parser = argparse.ArgumentParser()
    rl_cli.register_rl_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["rl", "train", "--config", "config.json"])

    assert args.func(args) == 2
    assert capsys.readouterr().err.strip() == "generation failed operationally"

    fake_engine.train = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        args.func(args)
