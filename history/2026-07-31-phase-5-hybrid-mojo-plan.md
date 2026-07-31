# Phase 5 Hybrid Mojo/Vulkan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect versioned Mojo/Vulkan/llama.cpp actors to the portable PyTorch learner with explicit staleness and atomic adapter publication.

**Architecture:** A newline-delimited JSON actor protocol separates the Python engine from a Mojo coordinator executable. Policy versions are mandatory, and publication never performs implicit format conversion.

**Tech Stack:** Python 3.10+, Mojo external toolchain, JSONL process protocol, PyTorch learner, pytest.

## Global Constraints

- Python source lines are at most 100 characters.
- Mojo, Vulkan, llama.cpp, and GGUF tooling remain external runtime dependencies.
- Every actor request and trajectory has a policy version.
- Default stale behavior is rejection; down-weighting is explicit and logged.
- Adapter publication verifies hashes before atomic manifest replacement.
- Use TDD; mock the process boundary when Mojo is unavailable.

---

### Task 1: Policy-version validation

**Files:**
- Create: `gepa_mindfulness/training/policy_versions.py`
- Test: `tests/test_rl_policy_versions.py`

**Interfaces:**
- Produces: `PolicyVersion`, `StalenessPolicy`, `StalenessDecision`, `evaluate_staleness()`.

- [ ] Write failing tests for missing versions, exact match, allowed lag, rejection, and deterministic down-weighting.

```python
def test_stale_trajectory_is_rejected_by_default() -> None:
    decision = evaluate_staleness(PolicyVersion(3), PolicyVersion(1), max_lag=1)
    assert decision.accepted is False
    assert decision.weight == 0.0
```

- [ ] Implement `PolicyVersion`, `StalenessPolicy`, and `evaluate_staleness()`.
- [ ] Run focused tests and commit with `feat: validate actor policy versions`.

### Task 2: Atomic adapter publication

**Files:**
- Create: `gepa_mindfulness/training/adapter_publication.py`
- Test: `tests/test_rl_adapter_publication.py`

**Interfaces:**
- Produces: `AdapterManifest` and `LocalAdapterPublisher.publish()`.

- [ ] Write failing tests for checksum mismatch, parent mismatch, interrupted temporary output, and successful atomic publication.

```python
def test_failed_publication_preserves_current_manifest(publisher) -> None:
    current = publisher.publish(valid_candidate("v1"))
    with pytest.raises(ArtifactHashError):
        publisher.publish(corrupt_candidate("v2", parent="v1"))
    assert publisher.current() == current
```

- [ ] Implement `AdapterManifest` and `LocalAdapterPublisher.publish()` without model conversion.
- [ ] Verify failed publication leaves the previous manifest readable.
- [ ] Commit with `feat: publish versioned adapters atomically`.

### Task 3: Mojo coordinator transport

**Files:**
- Create: `gepa_mindfulness/training/backends/mojo_coordinator.py`
- Create: `mojo/rl_coordinator/main.mojo`
- Modify: `gepa_mindfulness/training/contracts.py`
- Test: `tests/test_rl_hybrid.py`

**Interfaces:**
- Consumes: `ActorTransport`, `RolloutBackend`, JSONL message envelopes.
- Produces: `MojoProcessTransport` and `MojoCoordinatorBackend`.

- [ ] Write a fake-executable JSONL test for startup handshake, generate request, response validation, timeout, crash, and shutdown.

```python
def test_mojo_transport_requires_handshake(fake_coordinator) -> None:
    transport = MojoProcessTransport(fake_coordinator.command)
    assert transport.start().protocol_version == "gepa-actor-v1"
```

- [ ] Implement `ActorTransport` and `MojoCoordinatorBackend` with bounded process I/O and stderr capture.
- [ ] Implement the Mojo executable protocol with `hello`, `generate`, and `close` message types.
- [ ] Run Python mock tests; compile and run the Mojo protocol test only when `mojo` is installed.
- [ ] Commit with `feat: add Mojo actor coordinator protocol`.

### Task 4: Hybrid engine integration

**Files:**
- Create: `configs/rl/hybrid_vulkan_grpo.yaml`
- Modify: `gepa_mindfulness/training/engine.py`
- Modify: `gepa_mindfulness/training/rl_cli.py`
- Modify: `docs/rl/README.md`
- Modify: `tests/test_rl_hybrid.py`

**Interfaces:**
- Consumes: Mojo actor, PyTorch learner, GRPO algorithm, staleness and publication policies.
- Produces: hybrid `RLTrainingEngine` execution.

- [ ] Write a failing mocked end-to-end test: actor trajectories, integrity rewards, PyTorch GRPO step, parameter change, checkpoint, and versioned publication.

```python
def test_hybrid_step_updates_pytorch_and_publishes_version(hybrid_run) -> None:
    before = hybrid_run.learner.parameter_checksum()
    result = hybrid_run.engine.train(max_steps=1)
    assert result.parameter_checksum != before
    assert result.published_adapter.policy_version == "2"
```

- [ ] Wire actor and learner backends separately in the run manifest and engine.
- [ ] Reject `--learner mojo` before actor startup.
- [ ] Run hybrid tests and commit with `feat: add versioned hybrid RL` only after the PyTorch parameter-update assertion passes.
