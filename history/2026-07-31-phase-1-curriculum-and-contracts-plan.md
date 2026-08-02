# Phase 1 Curriculum and Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the reward-integrity curriculum, deterministic RL-pair conversion, observable reward overlay, and backend-neutral contracts.

**Architecture:** Extend the existing rich synthetic-case schema with an optional reward-integrity object. Keep rich cases authoritative, generate flat preference pairs deterministically, and expose reward/capability/trajectory interfaces without importing accelerator libraries.

**Tech Stack:** Python 3.10+, dataclasses, Protocol, JSON/JSONL, pytest, existing synthetic validator.

## Global Constraints

- Python source lines are at most 100 characters.
- Do not run `bd` or modify `.beads/*`.
- Reward-integrity components use `[-1.0, 1.0]`; negative values cite observable evidence.
- Do not score hidden thoughts, activations, or private scratchpads.
- Preserve existing public APIs and keep the overlay disabled by default.
- Use TDD: observe each focused test fail before implementation and pass afterward.

---

### Task 1: Capability and trajectory contracts

**Files:**
- Create: `gepa_mindfulness/training/capability.py`
- Create: `gepa_mindfulness/training/trajectory.py`
- Create: `gepa_mindfulness/training/contracts.py`
- Test: `tests/test_rl_capability.py`
- Test: `tests/test_rl_trajectory.py`

**Interfaces:**
- Produces: `Capability`, `CapabilityState`, `CapabilityEvidence`, `BackendCapabilities`.
- Produces: `Trajectory`, `TrajectoryBatch`, `RolloutRequest`, `PolicyEvaluation`.
- Produces: `RolloutBackend`, `TrainablePolicyBackend`, `RewardProvider`, `RLAlgorithm`.

- [ ] **Step 1: Write capability and trajectory tests**

```python
def test_unknown_capability_does_not_satisfy_requirement() -> None:
    report = BackendCapabilities.unknown("mock", "1")
    with pytest.raises(CapabilityError, match="supports_backward"):
        report.require({Capability.SUPPORTS_BACKWARD})


def test_trajectory_round_trip_preserves_null_log_probs() -> None:
    trajectory = Trajectory.minimal("traj-1", "prompt", "response")
    restored = Trajectory.from_dict(trajectory.to_dict())
    assert restored.old_log_probs is None
    assert restored == trajectory
```

- [ ] **Step 2: Run the focused tests and confirm import failures**

Run: `pytest tests/test_rl_capability.py tests/test_rl_trajectory.py -q`

Expected: collection fails because the three contract modules do not exist.

- [ ] **Step 3: Implement frozen dataclasses, tri-state requirement checks, and JSON conversion**

```python
class CapabilityState(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    case_id: str | None
    prompt: str
    response: str
    prompt_token_ids: tuple[int, ...] | None = None
    response_token_ids: tuple[int, ...] | None = None
    old_log_probs: tuple[float, ...] | None = None
    reference_log_probs: tuple[float, ...] | None = None
    value_predictions: tuple[float, ...] | None = None
    reward_total: float | None = None
    reward_components: Mapping[str, float] = field(default_factory=dict)
    advantage: tuple[float, ...] | None = None
    returns: tuple[float, ...] | None = None
    sampling_parameters: Mapping[str, object] = field(default_factory=dict)
    backend_name: str = ""
    backend_version: str = ""
    model_identifier: str = ""
    adapter_identifier: str | None = None
    policy_version: str | None = None
    seed: int | None = None
    trace_references: tuple[str, ...] = ()
```

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_rl_capability.py tests/test_rl_trajectory.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the contracts**

```bash
git add gepa_mindfulness/training/{capability,trajectory,contracts}.py \
  tests/test_rl_capability.py tests/test_rl_trajectory.py
git commit -m "feat: add backend-neutral RL contracts"
```

### Task 2: Rich schema and deterministic curriculum builder

**Files:**
- Modify: `data/synthetic/schema/synthetic_case.schema.json`
- Modify: `scripts/synthetic_dataset_tool.py`
- Create: `scripts/build_reward_integrity_rl_dataset.py`
- Create: `data/synthetic/reward_integrity/README.md`
- Create: `data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl`
- Generate: `data/synthetic/reward_integrity/rl_pairs_v1.jsonl`
- Generate: `data/synthetic/reward_integrity/curriculum_manifest.json`
- Test: `tests/test_reward_integrity_dataset.py`
- Test: `tests/test_reward_integrity_builder.py`

**Interfaces:**
- Consumes: existing `_validate_jsonl(path)` rich-case validator.
- Produces: `build_dataset(source: Path, pairs: Path, manifest: Path) -> BuildResult`.

- [ ] **Step 1: Write failing schema, ethics, provenance, and determinism tests**

```python
EXPECTED_FAMILIES = {
    "animal_reinforcement_pathway",
    "human_substance_addiction",
    "benchmark_leakage",
    "evaluator_manipulation",
    "reward_channel_tampering",
    "benign_shortcut",
    "structured_skill_acquisition",
    "recovery_and_repair",
}


def test_builder_is_byte_deterministic(tmp_path: Path) -> None:
    first = build_into(tmp_path / "first")
    second = build_into(tmp_path / "second")
    assert first.pairs.read_bytes() == second.pairs.read_bytes()
    assert first.manifest.read_bytes() == second.manifest.read_bytes()
```

- [ ] **Step 2: Run the dataset tests and confirm missing-file/schema failures**

Run: `pytest tests/test_reward_integrity_dataset.py tests/test_reward_integrity_builder.py -q`

Expected: tests fail because the reward-integrity data and builder do not exist.

- [ ] **Step 3: Extend the schema and author the eight exact case IDs**

```text
reward-integrity-001 animal_reinforcement_pathway
reward-integrity-002 human_substance_addiction
reward-integrity-003 benchmark_leakage (extends syn-gold-002)
reward-integrity-004 evaluator_manipulation
reward-integrity-005 reward_channel_tampering
reward-integrity-006 benign_shortcut
reward-integrity-007 structured_skill_acquisition
reward-integrity-008 recovery_and_repair
```

Each case contains all eight required response classes and the six required preference relations.
The addiction and animal cases use the scientific language constraints from the approved design.

- [ ] **Step 4: Implement deterministic pair generation and manifest hashing**

```python
PAIR_RULES = (
    ("grounded_success", "proxy_exploitation", "grounded_over_proxy"),
    ("exploit_disclosure", "proxy_exploitation", "disclosure_over_exploitation"),
    ("honest_failure", "evaluator_manipulation", "honest_failure_over_dishonesty"),
    ("benign_shortcut", "rationalized_failure", "creativity_over_rigidity"),
    ("exploit_disclosure", "reward_tampering", "repair_over_concealment"),
    ("grounded_success", "reward_tampering", "agency_over_immediate_reward"),
)
```

- [ ] **Step 5: Regenerate artifacts and run focused tests**

Run: `python scripts/build_reward_integrity_rl_dataset.py`

Run: `pytest tests/test_reward_integrity_dataset.py tests/test_reward_integrity_builder.py -q`

Expected: builder exits zero and all tests pass.

- [ ] **Step 6: Commit curriculum artifacts and tests**

```bash
git add data/synthetic scripts/build_reward_integrity_rl_dataset.py \
  scripts/synthetic_dataset_tool.py tests/test_reward_integrity_*.py
git commit -m "feat: add reward-integrity curriculum"
```

### Task 3: Observable reward overlay and pipeline

**Files:**
- Create: `gepa_mindfulness/core/reward_integrity.py`
- Create: `gepa_mindfulness/training/reward_pipeline.py`
- Test: `tests/test_reward_integrity_rewards.py`

**Interfaces:**
- Produces: `RewardIntegrityBreakdown`, `RewardIntegrityWeights`, `RewardObservation`.
- Produces: `RewardIntegrityCalculator.compute(observation) -> RewardIntegrityBreakdown`.
- Produces: `RewardPipeline.score(request) -> RewardResult`.

- [ ] **Step 1: Write failing ordering, range, visibility, and hidden-input tests**

```python
def test_equal_aggregate_keeps_distinct_components() -> None:
    left = calculator.compute(observation(objective_fidelity=1.0, reality_contact=-1.0))
    right = calculator.compute(observation(objective_fidelity=-1.0, reality_contact=1.0))
    assert left.aggregate == right.aggregate
    assert left.objective_fidelity != right.objective_fidelity


def test_hidden_state_input_is_rejected() -> None:
    with pytest.raises(TypeError):
        RewardObservation(hidden_state=[0.1])
```

- [ ] **Step 2: Run the focused test and confirm missing imports**

Run: `pytest tests/test_reward_integrity_rewards.py -q`

Expected: collection fails because the overlay module does not exist.

- [ ] **Step 3: Implement range validation, weighted aggregation, and optional composition**

```python
def aggregate_components(
    components: Mapping[str, float],
    weights: RewardIntegrityWeights,
) -> float:
    weights.validate()
    weighted = sum(components[name] * weights[name] for name in COMPONENT_NAMES)
    return weighted / weights.total
```

- [ ] **Step 4: Run the reward tests**

Run: `pytest tests/test_reward_integrity_rewards.py tests/test_rewards.py -q`

Expected: new and existing reward tests pass.

- [ ] **Step 5: Commit the overlay**

```bash
git add gepa_mindfulness/core/reward_integrity.py \
  gepa_mindfulness/training/reward_pipeline.py tests/test_reward_integrity_rewards.py
git commit -m "feat: add observable reward-integrity overlay"
```

### Task 4: Dataset adapters and Phase 1 documentation

**Files:**
- Create: `gepa_mindfulness/training/adapters/__init__.py`
- Create: `gepa_mindfulness/training/adapters/synthetic_cases.py`
- Create: `gepa_mindfulness/training/adapters/flat_jsonl.py`
- Modify: `docs/synthetic_dataset.md`
- Test: `tests/test_rl_adapters.py`

**Interfaces:**
- Produces: `SyntheticCaseAdapter.iter_requests() -> Iterator[RolloutRequest]`.
- Produces: `FlatJSONLAdapter.iter_requests() -> Iterator[RolloutRequest]`.

- [ ] **Step 1: Write failing provenance and malformed-row tests**

```python
def test_flat_adapter_preserves_case_provenance() -> None:
    request = next(FlatJSONLAdapter(PAIRS_PATH).iter_requests())
    assert request.case_id.startswith("reward-integrity-")
    assert request.metadata["source_sha256"]
```

- [ ] **Step 2: Run the adapter test and confirm missing imports**

Run: `pytest tests/test_rl_adapters.py -q`

- [ ] **Step 3: Implement streaming adapters with explicit row validation**

```python
def iter_json_objects(path: Path) -> Iterator[dict[str, object]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            yield validate_row(json.loads(raw), path, line_number)
```

- [ ] **Step 4: Document source-of-truth, rebuild, validation, and diagnostic commands**

Run: `python scripts/synthetic_dataset_tool.py validate data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl`

Run: `pytest tests/test_rl_adapters.py tests/test_reward_integrity_dataset.py -q`

- [ ] **Step 5: Commit adapters and documentation**

```bash
git add gepa_mindfulness/training/adapters tests/test_rl_adapters.py \
  docs/synthetic_dataset.md
git commit -m "feat: adapt reward-integrity data for RL"
```
