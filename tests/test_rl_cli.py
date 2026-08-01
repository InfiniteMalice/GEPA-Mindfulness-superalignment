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
    LoggingConfig,
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
