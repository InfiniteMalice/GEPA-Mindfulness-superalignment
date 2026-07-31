# Phase 4 llama.cpp Vulkan Actor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect and evaluate backend-neutral trajectories from a local llama.cpp server while reporting inference and Vulkan capabilities accurately.

**Architecture:** A standard-library HTTP client targets the llama.cpp server boundary. Server responses populate only fields that are actually returned; backward, optimizer, and full-training capabilities remain unsupported.

**Tech Stack:** Python standard library HTTP/JSON, llama-server REST API, pytest mock server.

## Global Constraints

- Python source lines are at most 100 characters.
- Do not add `requests`, an OpenAI client, llama.cpp bindings, or a Vulkan Python wheel.
- Missing token IDs or log probabilities serialize as `null`.
- Vulkan is supported only with positive evidence; reachability alone reports `unknown`.
- Use TDD and keep local native-runtime tests optional.

---

### Task 1: Server client and grouped rollout contract

**Files:**
- Create: `gepa_mindfulness/training/backends/llama_cpp_vulkan.py`
- Test: `tests/test_llama_cpp_vulkan_backend.py`

**Interfaces:**
- Consumes: `RolloutRequest`, `Trajectory`, `BackendCapabilities`.
- Produces: `LlamaCppVulkanBackend.generate()` and `LlamaCppServerClient`.

- [ ] Write a local mock HTTP server test for health, models, grouped completions, seeds, tokens, probabilities, timeout, and malformed responses.

```python
def test_missing_server_probabilities_remain_null(mock_llama_server) -> None:
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)
    trajectory = backend.generate([RolloutRequest(prompt="hello")])[0]
    assert trajectory.old_log_probs is None
    assert trajectory.response_token_ids is None
```

- [ ] Run the test and observe missing backend imports.
- [ ] Implement `LlamaCppVulkanBackend.generate()` with `urllib.request`, bounded timeouts, and response validation.

```python
request = urllib.request.Request(
    f"{self.endpoint}/completion",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
    body = json.loads(response.read().decode("utf-8"))
```

- [ ] Assert `supports_backward`, `supports_optimizer_step`, and `supports_full_weight_training` are always unsupported.
- [ ] Run focused tests and commit with `feat: add llama.cpp rollout backend`.

### Task 2: Doctor evidence and optional Vulkan integration

**Files:**
- Modify: `gepa_mindfulness/training/backends/llama_cpp_vulkan.py`
- Modify: `gepa_mindfulness/training/rl_cli.py`
- Create: `tests/test_llama_cpp_vulkan_integration.py`

**Interfaces:**
- Produces: `detect_llama_cpp_runtime()` and `detect_vulkan_evidence()`.

- [ ] Write failing tests for executable discovery, `vulkaninfo` evidence, endpoint-only unknown status, and unsupported training combinations.

```python
def test_reachable_endpoint_does_not_prove_vulkan(mock_llama_server) -> None:
    report = detect_llama_cpp_runtime(endpoint=mock_llama_server.endpoint)
    assert report.state(Capability.SUPPORTS_VULKAN) is CapabilityState.UNKNOWN
```

- [ ] Implement read-only subprocess probes and evidence strings without starting external services.
- [ ] Add an opt-in integration test controlled by `GEPA_LLAMA_CPP_ENDPOINT`.
- [ ] Run mock tests; run or cleanly skip the integration test.
- [ ] Commit with `feat: diagnose llama.cpp Vulkan actors`.

### Task 3: Collection CLI and documentation

**Files:**
- Create: `configs/rl/llama_cpp_vulkan_collect.yaml`
- Modify: `gepa_mindfulness/training/engine.py`
- Modify: `gepa_mindfulness/training/rl_cli.py`
- Modify: `docs/rl/README.md`
- Modify: `tests/test_rl_cli.py`

**Interfaces:**
- Consumes: common engine, logging sink, and llama.cpp backend.
- Produces: `gepa rl collect --backend llama-cpp-vulkan`.

- [ ] Write a failing `gepa rl collect` test that checks trajectory JSONL and nullable probability fields.

```python
def test_collect_writes_backend_neutral_trajectory(cli_runner, mock_llama_server) -> None:
    result = cli_runner.collect(mock_llama_server.endpoint)
    assert result.returncode == 0
    assert read_jsonl(result.output / "trajectories.jsonl")[0]["old_log_probs"] is None
```

- [ ] Wire collection and evaluation modes through the common engine and logging sink.
- [ ] Document Vulkan build, server launch, doctor, collect, and output-verification commands.
- [ ] Run Phase 4 tests and commit with `feat: collect Vulkan actor trajectories`.
