"""Canonical RL engine and ``gepa rl`` command contract tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gepa_mindfulness.training import rl_cli
from gepa_mindfulness.training.capability import (
    BackendCapabilities,
    Capability,
    CapabilityError,
    CapabilityEvidence,
    CapabilityState,
)
from gepa_mindfulness.training.engine import RLTrainingEngine, required_capabilities
from gepa_mindfulness.training.runtime_config import (
    AlgorithmConfig,
    CheckpointConfig,
    RLRunConfig,
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

    assert result.global_step == 2
    assert result.checkpoint_parent == "checkpoint-00000001"
    assert "checkpoint.load:operator-checkpoint" in events
    assert "log.start:resume:1:checkpoint-00000001" in events
    assert "checkpoint.save:2:checkpoint-00000001" in events


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
