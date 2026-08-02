# Phase 2 Portable PyTorch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement real CPU-capable PPO and GRPO, stable checkpoints, common logging, resume, and the canonical `gepa rl` CLI.

**Architecture:** `RLTrainingEngine` coordinates backend-neutral algorithms with one `TorchPolicyBackend`. Algorithms use a small differentiable-operations protocol; the backend owns models, devices, optimizers, generation, gradient work, and serialization.

**Tech Stack:** Python 3.10+, PyTorch 2.9, Transformers 4.57+, PEFT 0.17+, argparse, pytest.

## Global Constraints

- Python source lines are at most 100 characters.
- CPU acceptance tests are offline and construct tiny models from local configuration.
- PPO and GRPO must perform backward and optimizer steps.
- A full-RL claim requires a parameter change and successful checkpoint reload.
- Existing configuration, trainer, and CLI APIs remain available with deprecation warnings.
- Use TDD and commit each independently reviewable task.

---

### Task 1: Canonical runtime configuration and compatibility translation

**Files:**
- Create: `gepa_mindfulness/training/runtime_config.py`
- Modify: `gepa_mindfulness/training/config.py`
- Modify: `gepa_mindfulness/training/configs.py`
- Create: `configs/rl/pytorch_cpu_ppo.yaml`
- Create: `configs/rl/pytorch_cpu_grpo.yaml`
- Test: `tests/test_rl_runtime_config.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_configs.py`

**Interfaces:**
- Produces: `RLRunConfig.from_mapping()`, `load_rl_config()`, `translate_legacy_config()`.

- [ ] Write failing canonical parsing, device validation, and legacy translation tests.

```python
def test_legacy_grpo_config_translates_to_canonical() -> None:
    config = translate_legacy_config({"grpo": {"group_size": 4}, "device": "cpu"})
    assert config.runtime.backend == "pytorch"
    assert config.algorithm.name == "grpo"
    assert config.algorithm.group_size == 4
```

- [ ] Run `pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q` and observe missing canonical types.
- [ ] Implement frozen nested config dataclasses for runtime, policy, algorithm, reward, dataset, checkpoint, and logging sections.

```python
@dataclass(frozen=True)
class RLRunConfig:
    runtime: RuntimeConfig
    policy: PolicyConfig
    algorithm: AlgorithmConfig
    reward: RewardConfig
    dataset: DatasetConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    seed: int = 42
```

- [ ] Emit `DeprecationWarning(stacklevel=2)` only when a legacy loader or wrapper is used.
- [ ] Rerun the focused tests and commit with `feat: add canonical RL runtime config`.

### Task 2: PPO and GRPO tensor mathematics

**Files:**
- Create: `gepa_mindfulness/training/algorithms/__init__.py`
- Create: `gepa_mindfulness/training/algorithms/base.py`
- Create: `gepa_mindfulness/training/algorithms/ops.py`
- Create: `gepa_mindfulness/training/algorithms/ppo.py`
- Create: `gepa_mindfulness/training/algorithms/grpo.py`
- Test: `tests/test_rl_ppo.py`
- Test: `tests/test_rl_grpo.py`

**Interfaces:**
- Produces: `compute_gae()`, `compute_ppo_loss()`, `compute_group_advantages()`, `compute_grpo_loss()`.
- Produces: `PPOAlgorithm`, `GRPOAlgorithm`, and `AlgorithmLoss`.

- [ ] Write hand-computed tests for clipping, value clipping, entropy, KL, masking, GAE, group normalization, and zero variance.

```python
def test_ppo_clipped_surrogate_matches_manual_tensor() -> None:
    result = compute_ppo_loss(ops, batch, evaluation, PPOAlgorithmConfig(clip_range=0.2))
    expected = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
    assert result.policy_loss.item() == pytest.approx(expected.item())


def test_identical_group_rewards_produce_zero_advantages() -> None:
    advantages = compute_group_advantages([2.0, 2.0], zero_variance="zero")
    assert advantages == pytest.approx([0.0, 0.0])
```

- [ ] Run `pytest tests/test_rl_ppo.py tests/test_rl_grpo.py -q` and observe import failures.
- [ ] Implement formulas without device strings, model calls, or filesystem calls.

```python
ratio = ops.exp(evaluation.log_probs - batch.old_log_probs)
unclipped = ratio * batch.advantages
clipped = ops.clip(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
policy_loss = -ops.masked_mean(ops.minimum(unclipped, clipped * batch.advantages), batch.mask)
```

- [ ] Run focused tests and verify expected scalar values with `pytest.approx`.
- [ ] Commit with `feat: implement PPO and GRPO objectives`.

### Task 3: Shared PyTorch policy backend

**Files:**
- Create: `gepa_mindfulness/training/backends/__init__.py`
- Create: `gepa_mindfulness/training/backends/base.py`
- Create: `gepa_mindfulness/training/backends/torch_policy.py`
- Create: `gepa_mindfulness/training/backends/torch_portable.py`
- Test: `tests/test_rl_torch_backend.py`

**Interfaces:**
- Produces: `TorchPolicyBackend.generate()`, `evaluate()`, `backward()`, `optimizer_step()`.
- Produces: `TorchTensorOps` and `create_portable_backend()`.

- [ ] Write a tiny local tokenizer/model fixture and failing generation/log-probability/update tests.

```python
def test_optimizer_step_changes_policy_parameter(tiny_backend: TorchPolicyBackend) -> None:
    before = clone_trainable_parameters(tiny_backend)
    train_one_backend_step(tiny_backend)
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters(tiny_backend)))
```

- [ ] Run `pytest tests/test_rl_torch_backend.py -q` and observe missing backend imports.
- [ ] Implement response-only token masking, policy/reference log probabilities, value head, full-model mode, and optional PEFT LoRA mode.

```python
token_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
selected = token_log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
response_log_probs = selected[:, prompt_length - 1 :]
```

- [ ] Verify reference parameters stay frozen and at least one policy parameter changes after a step.
- [ ] Commit with `feat: add portable PyTorch policy backend`.

### Task 4: Checkpoint and logging schemas

**Files:**
- Create: `gepa_mindfulness/training/checkpointing.py`
- Create: `gepa_mindfulness/training/run_logging.py`
- Test: `tests/test_rl_checkpointing.py`
- Test: `tests/test_rl_logging.py`

**Interfaces:**
- Produces: `LocalCheckpointStore.save()`, `load()`, `CheckpointManifest`.
- Produces: `JSONLLoggingSink.start_run()`, `log_trajectory()`, `log_metrics()`.

- [ ] Write failing atomic-save, hash-validation, RNG, resume-step, manifest-field, and metric-component tests.

```python
def test_checkpoint_round_trip_restores_step(store: LocalCheckpointStore) -> None:
    manifest = store.save(snapshot(global_step=3))
    restored = store.load(manifest.path)
    assert restored.global_step == 3
    assert restored.artifact_hashes == manifest.artifact_hashes
```

- [ ] Run focused tests and observe import failures.
- [ ] Implement sibling temporary-directory saves, hash verification, atomic rename, and rank-safe JSONL writes.

```python
temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
write_snapshot(temporary, snapshot)
verify_artifact_hashes(temporary)
temporary.replace(destination)
```

- [ ] Verify corrupted artifacts fail before backend restoration.
- [ ] Commit with `feat: add RL checkpoints and structured logging`.

### Task 5: Engine and CLI

**Files:**
- Create: `gepa_mindfulness/training/engine.py`
- Create: `gepa_mindfulness/training/rl_cli.py`
- Modify: `src/mindful_trace_gepa/cli.py`
- Modify: `gepa_mindfulness/training/cli.py`
- Modify: `gepa_mindfulness/training/train.py`
- Modify: `gepa_mindfulness/training/ppo_trainer.py`
- Modify: `gepa_mindfulness/training/grpo_trainer.py`
- Modify: `gepa_mindfulness/training/pipeline.py`
- Test: `tests/test_rl_cli.py`

**Interfaces:**
- Produces: `RLTrainingEngine.train()`, `resume()`, `collect()`, `evaluate()`.
- Produces: `register_rl_cli(subparsers)` and `gepa rl doctor`.

- [ ] Write failing capability-before-load, CLI parsing, doctor, and deprecation tests.

```python
def test_unsupported_backend_fails_before_model_factory(monkeypatch) -> None:
    factory = Mock()
    with pytest.raises(CapabilityError):
        RLTrainingEngine(config, backend_factory=factory).train()
    factory.assert_not_called()
```

- [ ] Run `pytest tests/test_rl_cli.py -q` and observe failures.
- [ ] Implement engine lifecycle and register `train`, `collect`, `evaluate`, `resume`, and `doctor` under `gepa rl`.

```python
def register_rl_cli(subparsers: argparse._SubParsersAction) -> None:
    rl_parser = subparsers.add_parser("rl", help="Reinforcement-learning workflows")
    modes = rl_parser.add_subparsers(dest="rl_command", required=True)
    for name in ("train", "collect", "evaluate", "resume", "doctor"):
        register_mode(modes, name)
```

- [ ] Add explicit lightweight class names and deprecated compatibility wrappers.
- [ ] Run existing CLI/trainer tests plus `tests/test_rl_cli.py` and commit with `feat: add canonical RL engine and CLI`.

### Task 6: Offline CPU proof, dependencies, CI, and documentation

**Files:**
- Test: `tests/test_rl_engine_cpu.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `gepa_mindfulness/training/README.md`
- Modify: `docs/execution_flow.md`
- Modify: `docs/NEWCOMER_GUIDE.md`
- Create: `docs/rl/README.md`

**Interfaces:**
- Consumes: all Phase 1 and Phase 2 public interfaces.
- Produces: offline CPU acceptance evidence and exact runnable commands.

- [ ] Write the failing end-to-end test that records parameters, trains one step, reloads, verifies state, and resumes the global step.

```python
def test_cpu_rl_updates_reloads_and_resumes(tmp_path: Path) -> None:
    run = build_offline_cpu_run(tmp_path)
    before = run.backend.parameter_checksum()
    result = run.engine.train(max_steps=1)
    assert result.parameter_checksum != before
    restored = run.engine.resume(result.checkpoint)
    assert restored.parameter_checksum == result.parameter_checksum
    assert restored.global_step == 1
```

- [ ] Run `pytest tests/test_rl_engine_cpu.py -q` and confirm the acceptance assertion fails before integration is complete.
- [ ] Wire reward pipeline, algorithm, backend, checkpoint, and logging through the engine until the test passes offline.
- [ ] Add `rl` and `rl-dev` extras, retain the synchronized `train` extra, and omit TRL, datasets, Accelerate, and CUDA wheel pins.
- [ ] Correct all three documentation blocks and add exact CPU, resume, and doctor commands.
- [ ] Run Phase 2 tests, existing training tests, Ruff, Black, MyPy targets, and wheel build.
- [ ] Commit with `feat: prove portable model-weight RL` only after parameter and reload assertions pass.
