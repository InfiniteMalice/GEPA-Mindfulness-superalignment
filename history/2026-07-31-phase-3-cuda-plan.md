# Phase 3 CUDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add honest CUDA specialization while sharing all model, optimizer, algorithm, trajectory, reward, logging, and checkpoint code with portable PyTorch.

**Architecture:** `torch_cuda.py` validates and configures `TorchPolicyBackend`; it does not implement PPO, GRPO, model evaluation, or serialization again. Hardware tests are marked and skip cleanly when CUDA is absent.

**Tech Stack:** PyTorch 2.9 CUDA APIs, torch.amp, torch.distributed DDP/FSDP, pytest.

## Global Constraints

- Python source lines are at most 100 characters.
- Do not encode a CUDA-specific PyTorch wheel in `pyproject.toml`.
- Single-GPU CUDA must prove parameter change and checkpoint reload before production labeling.
- Distributed metrics and manifests are rank-safe and non-duplicated.
- Use TDD and keep hardware-dependent tests explicitly marked.

---

### Task 1: CUDA capability, precision, and OOM validation

**Files:**
- Create: `gepa_mindfulness/training/backends/torch_cuda.py`
- Create: `configs/rl/cuda_single_gpu.yaml`
- Test: `tests/test_rl_cuda.py`

**Interfaces:**
- Consumes: `TorchPolicyBackend`, `RLRunConfig`, `BackendCapabilities`.
- Produces: `detect_cuda_capabilities()` and `create_cuda_backend()`.

- [ ] Write failing tests for unavailable CUDA, invalid device index, BF16 rejection, AMP selection, and structured OOM errors.

```python
def test_cuda_factory_rejects_unavailable_runtime(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CapabilityError, match="CUDA is unavailable"):
        create_cuda_backend(config, model_factory)
```

- [ ] Run `pytest tests/test_rl_cuda.py -q` and observe missing backend imports.
- [ ] Implement `detect_cuda_capabilities()` and `create_cuda_backend()` as factories over `TorchPolicyBackend`.

```python
def create_cuda_backend(config: RLRunConfig, model_factory: ModelFactory) -> TorchPolicyBackend:
    capabilities = detect_cuda_capabilities(config.runtime.device)
    capabilities.require(required_cuda_capabilities(config))
    return TorchPolicyBackend(config, model_factory, torch.device(config.runtime.device))
```

- [ ] Run non-hardware tests and confirm hardware tests skip when `torch.cuda.is_available()` is false.
- [ ] Commit with `feat: specialize PyTorch RL for CUDA`.

### Task 2: Single-GPU acceptance test

**Files:**
- Modify: `tests/test_rl_cuda.py`
- Modify: `docs/rl/README.md`

**Interfaces:**
- Consumes: `create_cuda_backend()` and `LocalCheckpointStore`.
- Produces: hardware-marked parameter-update evidence.

- [ ] Add `@pytest.mark.cuda` test that records a checksum, performs one optimizer step under each supported precision, reloads the checkpoint, and compares state.

```python
@pytest.mark.cuda
def test_cuda_parameter_update_and_checkpoint(cuda_run) -> None:
    before = cuda_run.backend.parameter_checksum()
    result = cuda_run.engine.train(max_steps=1)
    assert result.parameter_checksum != before
    assert cuda_run.reload(result.checkpoint).parameter_checksum == result.parameter_checksum
```

- [ ] Run `pytest -m cuda tests/test_rl_cuda.py -q` on CUDA hardware or record a clean skip on non-CUDA hosts.
- [ ] Document exact install, train, OOM-diagnosis, and checkpoint-resume commands.
- [ ] Commit with `test: prove CUDA parameter updates` only on evidence-producing hardware; otherwise commit the gated test without claiming it passed.

### Task 3: Distributed validation and rank-safe output

**Files:**
- Create: `configs/rl/cuda_ddp.yaml`
- Modify: `gepa_mindfulness/training/backends/torch_cuda.py`
- Modify: `gepa_mindfulness/training/checkpointing.py`
- Modify: `gepa_mindfulness/training/run_logging.py`
- Modify: `tests/test_rl_cuda.py`

**Interfaces:**
- Consumes: PyTorch `DistributedDataParallel`, FSDP, common checkpoint and logging sinks.
- Produces: `DistributedRuntimeConfig` validation and rank-safe writers.

- [ ] Write failing configuration tests for DDP/FSDP world size, rank, sharded optimizer, and unsupported device combinations.

```python
def test_ddp_requires_world_size_greater_than_one() -> None:
    with pytest.raises(ValueError, match="world_size"):
        DistributedRuntimeConfig(strategy="ddp", world_size=1)
```

- [ ] Implement DDP/FSDP wrapping through PyTorch primitives and rank-zero manifest/metric writers.
- [ ] Add a marked two-process smoke test that asserts one manifest and no duplicated trajectory IDs.
- [ ] Run configuration tests everywhere and distributed smoke only where two CUDA devices exist.
- [ ] Commit with `feat: add optional distributed CUDA RL`.
