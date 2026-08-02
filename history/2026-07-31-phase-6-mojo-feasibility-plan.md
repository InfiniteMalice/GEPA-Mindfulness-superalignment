# Phase 6 Pure Mojo Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-backed pure-Mojo learner decision and keep unsupported training impossible to invoke accidentally.

**Architecture:** The feasibility report evaluates each required learner capability against reproducible evidence. Code changes are limited to capability reporting and fail-fast behavior unless every gate is proven.

**Tech Stack:** Mojo external toolchain documentation and probes, PyTorch numerical reference, pytest.

## Global Constraints

- Python source lines are at most 100 characters.
- Absence of evidence is not support.
- A pure Mojo learner requires backward, optimizer, LoRA update, checkpoint, transfer, parity, and hardware evidence.
- `--learner mojo` fails before actor startup unless every capability is supported.
- Use TDD for all capability and CLI behavior.

---

### Task 1: Feasibility evidence matrix

**Files:**
- Create: `docs/rl/mojo_learner_feasibility.md`

**Interfaces:**
- Produces: a nine-gate evidence table with one `supported`, `unsupported`, or `unknown` verdict per gate.

- [ ] Record the tested Mojo version, commands, hardware, and dates.
- [ ] Evaluate trainable tensors, autodiff or explicit backward, optimizer state, transformer backward kernels, LoRA updates, trainable checkpoint format, llama.cpp adapter transfer, numerical parity, and hardware coverage.
- [ ] Assign each gate `supported`, `unsupported`, or `unknown` and cite the observable evidence.
- [ ] Conclude `go` only when every gate is supported; otherwise conclude `no-go` and retain hybrid RL.
- [ ] Run the documentation precision gate and commit with `docs: assess pure Mojo learner feasibility`.

### Task 2: Capability and CLI enforcement

**Files:**
- Modify: `gepa_mindfulness/training/capability.py`
- Modify: `gepa_mindfulness/training/rl_cli.py`
- Modify: `tests/test_rl_capability.py`
- Modify: `tests/test_rl_cli.py`

**Interfaces:**
- Consumes: the report conclusion and backend capability registry.
- Produces: fail-fast pure-Mojo learner validation.

- [ ] Write failing tests that prove `--learner mojo` rejects unsupported and unknown gates before actor construction.

```python
def test_mojo_learner_rejected_before_actor_start(monkeypatch) -> None:
    actor_factory = Mock()
    with pytest.raises(CapabilityError, match="pure Mojo learner is unsupported"):
        run_train(config_with_mojo_learner(), actor_factory=actor_factory)
    actor_factory.assert_not_called()
```

- [ ] Implement the report-backed capability result and actionable error message.
- [ ] Run focused tests and confirm no inference-only path can satisfy train mode.
- [ ] Commit with `fix: enforce pure Mojo learner capability gate`.

### Task 3: Program-wide completion gate

**Files:**
- Modify: `docs/rl/README.md`
- Modify: `gepa_mindfulness/training/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all phase test evidence and documentation maturity labels.
- Produces: final verified user-facing status.

- [ ] Run all curriculum, reward, algorithm, CPU, CLI, mock actor, hybrid, and regression tests.
- [ ] Run CUDA, Vulkan, llama.cpp, and Mojo tests where hardware/runtime prerequisites exist; record clean skips otherwise.
- [ ] Run Ruff, Black, MyPy targets, `git diff --check`, wheel build, and wheel smoke.
- [ ] Verify documentation labels every path as production-ready, experimental, hybrid, or unsupported using actual test evidence.
- [ ] Commit final documentation corrections only after the documentation precision gate has no `BLOCK` findings.
