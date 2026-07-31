# Reward-Integrity Curriculum and Multi-Backend RL Design

**Status:** Approved in conversation on 2026-07-31; awaiting review of this written form.

**Repository baseline:** `main` at `1878a2d7c690963745a5a40c3bfb7913151cb753`.

## Purpose

This program adds two connected capabilities:

1. A synthetic curriculum that teaches that reward is evidence of success rather than the goal
   itself.
2. A real reinforcement-learning control plane with portable PyTorch, CUDA, and experimental
   Mojo/Vulkan/llama.cpp runtime paths.

The implementation must not describe rollout generation, scalar prompt-logit adjustment, or
inference-only evaluation as full model-weight reinforcement learning. A runtime is trainable only
when an automated test observes a model or adapter parameter change and successfully reloads the
saved state.

## Confirmed Baseline

The repository inspection established these facts:

- `gepa_mindfulness/training/pipeline.py` generates placeholder compatibility rollouts.
- `gepa_mindfulness/training/ppo_trainer.py` updates prompt-keyed Python scalar logits and value
  estimates. It does not update transformer weights.
- The dataset mode in `gepa_mindfulness/training/grpo_trainer.py` is a lightweight test double that
  updates prompt-keyed scalar logits.
- The model mode in `gepa_mindfulness/training/grpo_trainer.py` generates and scores responses,
  assigns fabricated zero log-probability tensors, and performs no backward or optimizer step.
- `gepa_mindfulness/training/config.py` and `gepa_mindfulness/training/configs.py` define overlapping
  configuration types with different shapes and defaults.
- `gepa_mindfulness/training/README.md` describes dataclasses as Pydantic models, describes the
  compatibility path as an end-to-end PPO loop, and mentions dependencies that do not match
  `pyproject.toml`.
- The existing four-case rich synthetic dataset validates with
  `scripts/synthetic_dataset_tool.py`.

## Scope and Delivery Strategy

The program is divided into six independently reviewable phases. Each phase receives its own
test-driven implementation plan after this design is accepted. A phase may not use the maturity
label assigned to a later phase.

1. Curriculum, reward overlay, common schemas, and interfaces.
2. Portable PyTorch PPO and GRPO with CPU acceptance tests.
3. CUDA specialization that reuses the portable PyTorch implementation.
4. Inference-only llama.cpp/Vulkan actors.
5. Mojo-coordinated hybrid reinforcement learning with a PyTorch learner.
6. A pure-Mojo learner feasibility report and, only if all gates pass, a separate implementation.

The implementation does not add a second set of PPO or GRPO algorithms for CUDA, Vulkan, or Mojo.
Runtime-specific behavior remains behind backend, tensor-operation, checkpoint, transport, and
capability interfaces.

## Planned File Map

Phase 1 creates or modifies:

- `data/synthetic/reward_integrity/README.md`
- `data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl`
- `data/synthetic/reward_integrity/curriculum_manifest.json`
- `data/synthetic/reward_integrity/rl_pairs_v1.jsonl`
- `data/synthetic/schema/synthetic_case.schema.json`
- `scripts/build_reward_integrity_rl_dataset.py`
- `scripts/synthetic_dataset_tool.py`
- `gepa_mindfulness/core/reward_integrity.py`
- `gepa_mindfulness/training/contracts.py`
- `gepa_mindfulness/training/capability.py`
- `gepa_mindfulness/training/trajectory.py`
- `gepa_mindfulness/training/reward_pipeline.py`
- `gepa_mindfulness/training/adapters/__init__.py`
- `gepa_mindfulness/training/adapters/synthetic_cases.py`
- `gepa_mindfulness/training/adapters/flat_jsonl.py`
- `tests/test_reward_integrity_dataset.py`
- `tests/test_reward_integrity_builder.py`
- `tests/test_reward_integrity_rewards.py`
- `tests/test_rl_trajectory.py`
- `tests/test_rl_adapters.py`
- `tests/test_rl_capability.py`

Phase 2 creates or modifies:

- `gepa_mindfulness/training/runtime_config.py`
- `gepa_mindfulness/training/checkpointing.py`
- `gepa_mindfulness/training/run_logging.py`
- `gepa_mindfulness/training/engine.py`
- `gepa_mindfulness/training/algorithms/__init__.py`
- `gepa_mindfulness/training/algorithms/base.py`
- `gepa_mindfulness/training/algorithms/ops.py`
- `gepa_mindfulness/training/algorithms/ppo.py`
- `gepa_mindfulness/training/algorithms/grpo.py`
- `gepa_mindfulness/training/backends/__init__.py`
- `gepa_mindfulness/training/backends/base.py`
- `gepa_mindfulness/training/backends/torch_policy.py`
- `gepa_mindfulness/training/backends/torch_portable.py`
- `gepa_mindfulness/training/rl_cli.py`
- `src/mindful_trace_gepa/cli.py`
- `configs/rl/pytorch_cpu_ppo.yaml`
- `configs/rl/pytorch_cpu_grpo.yaml`
- `gepa_mindfulness/training/config.py`
- `gepa_mindfulness/training/configs.py`
- `gepa_mindfulness/training/cli.py`
- `gepa_mindfulness/training/train.py`
- `gepa_mindfulness/training/ppo_trainer.py`
- `gepa_mindfulness/training/grpo_trainer.py`
- `gepa_mindfulness/training/pipeline.py`
- `gepa_mindfulness/training/README.md`
- `docs/execution_flow.md`
- `docs/NEWCOMER_GUIDE.md`
- `docs/rl/README.md`
- `pyproject.toml` and `.github/workflows/ci.yml`
- `tests/test_rl_runtime_config.py`
- `tests/test_rl_ppo.py`
- `tests/test_rl_grpo.py`
- `tests/test_rl_torch_backend.py`
- `tests/test_rl_checkpointing.py`
- `tests/test_rl_logging.py`
- `tests/test_rl_engine_cpu.py`
- `tests/test_rl_cli.py`

Phase 3 adds `gepa_mindfulness/training/backends/torch_cuda.py`,
`configs/rl/cuda_single_gpu.yaml`, `configs/rl/cuda_ddp.yaml`, and `tests/test_rl_cuda.py`, and it
adds CUDA sections to `docs/rl/README.md`. Phase 4 adds
`gepa_mindfulness/training/backends/llama_cpp_vulkan.py`,
`configs/rl/llama_cpp_vulkan_collect.yaml`, `tests/test_llama_cpp_vulkan_backend.py`, and
`tests/test_llama_cpp_vulkan_integration.py`. Phase 5 adds
`gepa_mindfulness/training/backends/mojo_coordinator.py`,
`gepa_mindfulness/training/policy_versions.py`,
`gepa_mindfulness/training/adapter_publication.py`, `mojo/rl_coordinator/main.mojo`,
`configs/rl/hybrid_vulkan_grpo.yaml`, `tests/test_rl_policy_versions.py`,
`tests/test_rl_adapter_publication.py`, and `tests/test_rl_hybrid.py`. Phase 6 adds
`docs/rl/mojo_learner_feasibility.md` and changes learner code only if the report's evidence satisfies
every feasibility gate. If any gate fails, the implementation retains the fail-fast Mojo learner.

## Considered Designs

### Selected: phased shared control plane

One engine coordinates typed interfaces and validates capabilities before loading a model. Each
phase adds a usable vertical slice. This approach preserves current APIs, makes unsupported states
observable, and keeps CPU tests authoritative for shared algorithm behavior.

### Rejected: big-bang replacement

A single replacement would reduce the temporary compatibility surface, but it would combine
dataset, reward, configuration, CLI, optimizer, distributed, and native-runtime risks in one
review. Failures would be difficult to localize and existing demos would be more likely to break.

### Rejected: third-party trainer as the control plane

Using TRL or Accelerate as the primary abstraction could shorten the first PyTorch implementation,
but it would not provide a clean contract for inference-only llama.cpp actors or a future Mojo
learner. The design therefore implements the required PPO and GRPO mathematics directly and uses
optional libraries only for capabilities they uniquely provide.

## Reward-Integrity Curriculum

### Location and source of truth

The rich source dataset lives under `data/synthetic/reward_integrity/`:

- `README.md`
- `reward_integrity_curriculum_v1.jsonl`
- `curriculum_manifest.json`
- `rl_pairs_v1.jsonl`

`reward_integrity_curriculum_v1.jsonl` is the source of truth. The builder
`scripts/build_reward_integrity_rl_dataset.py` deterministically regenerates the flat pair file and
manifest. The builder writes records in source-line order, then in the documented preference-rule
order. Two builds from identical rich input must be byte-identical.

Every case and rubric contains this exact central diagnostic:

> If the reward channel disappeared and reality alone judged the result, would this still count as
> success?

Every case also asks:

- What was the reward intended to represent?
- Did the action achieve that underlying purpose?
- Did it build transferable competence?
- Did it preserve accurate feedback?
- Did it improve or damage long-term agency?
- Would the method remain defensible if fully disclosed?
- Is this a novel solution or merely a way to make the metric look better?

The curriculum uses these exact causal paths:

```text
Healthy reinforcement:
need or objective
-> useful action
-> real-world result
-> reward signal
-> transferable learning

Corrupted reinforcement:
need or objective
-> direct access to reward signal
-> repeated shortcut
-> degradation of skill, reality contact, or long-term agency
```

### Existing schema extension

`data/synthetic/schema/synthetic_case.schema.json` gains an optional top-level
`reward_integrity` object. The existing top-level required list does not change, so existing data
remains valid. A dataset-specific validator requires the object for every reward-integrity case.

The `reward_integrity` object contains:

- `central_diagnostic`: the exact required diagnostic question.
- `supporting_diagnostics`: all seven supporting questions.
- `healthy_reinforcement_path` and `corrupted_reinforcement_path`.
- `response_classes`: exactly the eight required response-class keys.
- `preference_orderings`: explicit chosen and rejected class pairs with rationales.
- `observable_outcomes`: facts available to a scorer without private reasoning access.
- `component_targets`: authored target values for the eight overlay components.
- `repair_protocol`: the eight ordered repair actions.
- `scientific_and_ethical_notes`: the factual and non-stigmatizing constraints used by tests.

Each of the eight required case families contains all eight response classes:

- `grounded_success`
- `benign_shortcut`
- `exploit_disclosure`
- `proxy_exploitation`
- `evaluator_manipulation`
- `reward_tampering`
- `honest_failure`
- `rationalized_failure`

The benchmark-leakage case extends the existing `syn-gold-002` scenario rather than copying it. Its
provenance records that source case ID and explains the additional response classes, audit evidence,
patch, and rerun behavior.

### Flat RL pair schema

Every generated pair contains:

- `record_id`
- `source_case_id`
- `source_case_version`
- `source_path`
- `source_line`
- `source_sha256`
- `pair_rule`
- `prompt`
- `chosen`
- `rejected`
- `chosen_class`
- `rejected_class`
- `chosen_reward_components`
- `rejected_reward_components`
- `diagnostics`
- `schema_version`

The builder rejects unknown case IDs, missing response classes, duplicate record IDs, invalid
preference orderings, and component values outside the documented range.

### Ethical and scientific acceptance rules

Dataset tests reject claims that all experimental animals self-stimulated until death, descriptions
of dopamine as only a pleasure chemical, language that presents addiction as a moral defect, and
language that describes people with addiction as choosing reward for nothing. The addiction case
must describe motivation, incentive salience, learning or habit formation, reduced agency, and a
recovery or repair path.

Reward-channel tampering examples use toy names and abstract actions. Dataset tests reject shell
commands, network targets, credentials, real service names, executable payloads, and stepwise
intrusion instructions.

## Reward-Integrity Overlay

`gepa_mindfulness/core/reward_integrity.py` defines:

```python
@dataclass(frozen=True)
class RewardIntegrityBreakdown:
    objective_fidelity: float
    feedback_integrity: float
    skill_transfer: float
    reality_contact: float
    exploit_disclosure: float
    long_horizon_agency: float
    benign_creativity: float
    repair_quality: float
    aggregate: float
```

Every component has the inclusive range `[-1.0, 1.0]`. A value of `0.0` is neutral or not
applicable. Negative values require a cited observable output or action outcome. The aggregate is a
weighted arithmetic mean after validating that all weights are non-negative and have positive
total mass. Default component weights are equal.

The overlay is disabled by default. When enabled, the reward pipeline computes:

```text
total_reward = gepa_reward + overlay_weight * reward_integrity.aggregate
```

`overlay_weight` must be finite and non-negative. Disabling the overlay leaves the existing GEPA
reward unchanged. Logs always preserve every source component and never persist only the aggregate.

`RewardIntegrityProvider` accepts either authored observable annotations or an implementation of an
observable-output evaluator protocol. Enabling the provider without either input fails with an
actionable configuration error. The provider does not accept chain-of-thought, activations, hidden
states, or private scratchpad text.

## Control-Plane Architecture

```text
RLRunConfig
-> capability validation
-> DatasetAdapter
-> RolloutBackend
-> RewardPipeline
-> RLAlgorithm
-> TrainablePolicyBackend
-> CheckpointStore and LoggingSink
```

### Capabilities

`CapabilityState` has three serialized values: `supported`, `unsupported`, and `unknown`. Each
capability entry also contains evidence text. A required capability passes only when its state is
`supported`; `unknown` fails as safely as `unsupported`.

The initial capability names are:

- `supports_generation`
- `supports_token_log_probs`
- `supports_reference_log_probs`
- `supports_backward`
- `supports_optimizer_step`
- `supports_value_head`
- `supports_full_weight_training`
- `supports_lora_training`
- `supports_distributed_training`
- `supports_mixed_precision`
- `supports_vulkan`
- `supports_cuda`
- `supports_gguf`

Capability validation runs before model loading, dataset materialization, checkpoint mutation, or
rollout collection.

### Core protocols

`gepa_mindfulness/training/contracts.py` defines these public protocols and result types:

```python
class RolloutBackend(Protocol):
    def generate(
        self,
        requests: Sequence[RolloutRequest],
    ) -> Sequence[Trajectory]: ...

    def capabilities(self) -> BackendCapabilities: ...
    def close(self) -> None: ...


class TrainablePolicyBackend(RolloutBackend, Protocol):
    def evaluate(self, batch: TrajectoryBatch) -> PolicyEvaluation: ...
    def backward(self, loss: object) -> None: ...
    def optimizer_step(self) -> OptimizerStepResult: ...
    def zero_grad(self) -> None: ...
    def save_checkpoint(self, destination: Path) -> CheckpointManifest: ...
    def load_checkpoint(self, source: Path) -> CheckpointManifest: ...


class RewardProvider(Protocol):
    def score(self, request: RewardRequest) -> RewardResult: ...


class RLAlgorithm(Protocol):
    def required_capabilities(self) -> frozenset[Capability]: ...

    def compute_loss(
        self,
        batch: TrajectoryBatch,
        evaluation: PolicyEvaluation,
    ) -> AlgorithmLoss: ...
```

`CheckpointStore`, `DatasetAdapter`, `LoggingSink`, `ActorTransport`, and
`DifferentiableTensorOps` are separate protocols. `DifferentiableTensorOps` contains only the
primitives needed by PPO and GRPO formulas. Algorithm modules contain no device strings, CUDA API
calls, AMP contexts, distributed collectives, model loading, or filesystem operations.

### Trajectory schema

`Trajectory` is backend-neutral and JSON-serializable. It contains:

- `trajectory_id`
- `case_id`
- `prompt`
- `response`
- `prompt_token_ids`
- `response_token_ids`
- `old_log_probs`
- `reference_log_probs`
- `value_predictions`
- `reward_total`
- `reward_components`
- `advantage`
- `returns`, serialized as `return`
- `sampling_parameters`
- `backend_name`
- `backend_version`
- `model_identifier`
- `adapter_identifier`
- `policy_version`
- `seed`
- `trace_references`

Token-level arrays align to response tokens only. Prompt and padding masks live in
`TrajectoryBatch`. Any unavailable value is `null`; no backend may insert placeholder token IDs,
log probabilities, value predictions, or versions.

## PPO and GRPO

### PPO

PPO v1 requires generation, token log probabilities, reference log probabilities, backward,
optimizer step, and a value head. It implements token-masked clipped policy loss, value loss,
optional clipped value loss, entropy regularization, reference-policy KL diagnostics, generalized
advantage estimation, configurable advantage normalization, gradient accumulation, and target-KL
warning or early stopping.

The terminal response reward is applied to the final generated token. Generalized advantage
estimation propagates that reward across generated tokens using configured `gamma` and
`gae_lambda`. Prompt and padding tokens never contribute to policy, value, entropy, or KL means.

For each generated token, PPO computes `ratio = exp(new_log_prob - old_log_prob)`. The policy term
is the negative masked mean of the minimum of the unclipped and clipped ratio times advantage. The
value term is a masked mean squared error against returns, using the maximum of unclipped and
configured clipped value errors when value clipping is enabled. The diagnostic reference KL is the
masked mean of `new_log_prob - reference_log_prob`.

### GRPO

GRPO v1 requires grouped generation, token log probabilities, reference log probabilities,
backward, and optimizer step. It groups trajectories by prompt and group ID, normalizes rewards
within each group, applies response-token masks, and adds reference-policy KL control.

The configurable zero-variance policies are `zero`, `center_only`, and `skip`. The default is
`zero`, which assigns zero advantages and performs no policy-gradient contribution for the group
while retaining group metrics and KL diagnostics.

GRPO computes `ratio = exp(new_log_prob - old_log_prob)` and uses the same configured clipped
surrogate form as PPO without a value term. It adds the configured reference-policy KL coefficient
to the loss. Group normalization uses the population standard deviation plus the configured
epsilon unless a zero-variance policy applies.

Both algorithms expose pure loss functions tested against hand-computed tensors. The engine owns
gradient accumulation and asks the backend to clip gradients, step the optimizer, and clear
gradients.

## Runtime Backends

### Portable PyTorch

`TorchPolicyBackend` owns model loading, generation, policy/reference evaluation, the value head,
optimizer state, scheduler state, gradient clipping, and checkpoint serialization. It accepts a
validated `torch.device` rather than branching on a fixed list of device names.

`torch_portable.py` supplies the portable runtime factory and capability detector. CPU is mandatory
in CI. Other registered PyTorch devices are accepted only after device and dtype probes succeed.

Full-model training is built in. LoRA mode requires PEFT and fails with an installation hint when
PEFT is absent. The reference policy is frozen. Old-policy log probabilities are captured from the
policy that generated each trajectory.

The offline acceptance fixture constructs a tiny causal language model from local Transformers
configuration and uses a deterministic local tokenizer. It performs no network access.

### CUDA

`torch_cuda.py` configures the same `TorchPolicyBackend` for CUDA. It validates the selected device,
precision, BF16 support, AMP behavior, gradient checkpointing, and distributed strategy.

FP16 uses `torch.amp.GradScaler`; BF16 does not require loss scaling. CUDA out-of-memory exceptions
are converted to an error that reports the attempted batch, sequence, precision, accumulation, and
device configuration without claiming automatic recovery.

Single-GPU training is the first production CUDA target. DDP and FSDP are optional extensions with
rank-zero manifests, rank-safe checkpoint behavior, metric reduction, and no duplicate sample
records.

### llama.cpp with Vulkan

`llama_cpp_vulkan.py` connects to a configured local `llama-server` endpoint. It collects grouped
rollouts and server metadata, but it never exposes backward, optimizer, value-head, or model-weight
training capabilities.

The backend records token IDs and log probabilities only when the server response returns those
values. Vulkan support is `supported` only when local executable output, `vulkaninfo`, or trusted
endpoint metadata proves Vulkan use. A reachable endpoint without such evidence reports Vulkan as
`unknown`.

The first integration uses the server boundary because llama.cpp documents OpenAI-compatible
completion routes and native completion probability records. A Vulkan-enabled llama.cpp build is a
separate system artifact produced with `GGML_VULKAN`; it is not installed as a project wheel.

### Mojo coordinator and hybrid learner

The Mojo coordinator is an external executable using newline-delimited JSON requests and responses.
`MojoCoordinatorBackend` starts or connects to the coordinator through `ActorTransport`. The
coordinator calls the configured llama.cpp actor and returns backend-neutral trajectory envelopes.

Every actor request and trajectory contains a `policy_version`. The learner has a configured
maximum version lag and one of two explicit stale policies: `reject` or `down_weight`. The default is
`reject`. The down-weight factor and observed lag are logged when down-weighting is enabled.

Adapter publication writes a temporary artifact and manifest, verifies hashes and version ancestry,
then atomically replaces the published manifest. Conversion to a llama.cpp-compatible adapter is a
separate operator-invoked step. The Python learner never silently converts model formats.

`--learner mojo` fails during capability validation unless a separately approved pure-Mojo learner
implements and proves all required training capabilities.

## Configuration and CLI

`RLRunConfig` in `runtime_config.py` is the canonical configuration for new execution. Its top-level
sections are `runtime`, `policy`, `algorithm`, `reward`, `dataset`, `checkpoint`, and `logging`.

Supported backend identifiers are:

- `pytorch`
- `cuda`
- `llama-cpp-vulkan`
- `mojo-vulkan-llamacpp`

Supported execution modes are:

- `train`
- `collect`
- `evaluate`
- `resume`
- `doctor`

The installed CLI adds:

```text
gepa rl train
gepa rl collect
gepa rl evaluate
gepa rl resume
gepa rl doctor
```

`gepa rl doctor` is read-only. It reports optional imports, PyTorch version and devices, CUDA and
BF16 status, distributed availability, Mojo and llama.cpp executables, configured endpoint health,
Vulkan evidence, backend capability reports, and requested unsupported combinations. Endpoint
checks occur only when the operator supplies an endpoint.

## Compatibility and Migration

The existing public imports remain available during migration.

- `LightweightPPOTrainer`, `LightweightGRPOTrainer`, and
  `CompatibilityTrainingOrchestrator` become the explicit names for current test implementations.
- `PPOTrainer`, `GRPOTrainer`, and `TrainingOrchestrator` remain deprecated compatibility wrapper
  classes. Construction emits a `DeprecationWarning` with `stacklevel=2` and names the replacement.
- `config.py` and `configs.py` retain their current public types for compatibility. Calling a legacy
  loader translates the legacy shape into `RLRunConfig` for the production engine and emits a
  `DeprecationWarning` with `stacklevel=2`. New production modules import only
  `runtime_config.py`.
- Existing YAML files continue to load. Compatibility tests snapshot their translated canonical
  values.
- Existing Click and `training.train` commands remain available and delegate where their semantics
  have an exact new equivalent. Commands that only run compatibility behavior say so explicitly.
- Existing reward fields remain in logs. New reward-integrity and optimizer metrics are additive.

No compatibility adapter may silently strengthen an inference-only operation into a training claim.

## Dependencies

`pyproject.toml` gains these optional groups:

```toml
rl = [
    "torch>=2.9.0,<3",
    "transformers>=4.57,<6",
    "peft>=0.17,<1",
]

rl-dev = [
    "pytest>=8,<10",
]
```

The existing `train` extra remains synchronized with `rl` for compatibility and is marked
deprecated in documentation. The implementation does not add TRL because PPO and GRPO are local
algorithm implementations. It does not add `datasets` because adapters stream JSONL directly. It
does not add Accelerate because initial distributed support uses PyTorch DDP and FSDP directly.

There is no `rl-cuda` wheel extra. Documentation instructs operators to install the appropriate
PyTorch build for their platform and CUDA runtime, then install `.[rl]`. Mojo, the Vulkan SDK,
`vulkaninfo`, `llama-server`, and GGUF conversion tools remain external runtime dependencies.

## Logging and Checkpoints

Every run writes `run_manifest.json`, `metrics.jsonl`, and `trajectories.jsonl` using one schema
across backends. The run manifest contains `run_id`, `algorithm`, `backend`, `actor_backend`,
`learner_backend`, `model`, `reference_model`, `adapter`, `dataset_hash`, `config_hash`, `seed`,
`software_versions`, `device_capabilities`, `start_time`, `checkpoint_parent`, and an explicit
schema version. Rank-zero is the only writer of the manifest and aggregate metrics in distributed
runs.

Metrics retain `task_success`, `gepa_alignment`, `honesty`, `hallucination`,
`paraconsistent_truth`, `objective_fidelity`, `feedback_integrity`, `skill_transfer`,
`reality_contact`, `exploit_disclosure`, `long_horizon_agency`, `benign_creativity`,
`repair_quality`, `total_reward`, `policy_loss`, `value_loss`, `entropy`, `kl`, `gradient_norm`, and
`learning_rate` individually. GRPO records group-level and response-level metrics. Every record also
identifies the backend, actor, learner, and policy version.

A checkpoint directory contains policy or adapter state, value-head state when present, optimizer
state, scheduler state or an explicit `null` scheduler marker, algorithm state, global step, Python
and PyTorch CPU/CUDA random-number-generator states where available, canonical configuration,
dataset and configuration hashes, parent checkpoint, and schema version.

Checkpoint creation occurs in a sibling temporary directory. The store verifies the manifest and
artifact hashes before atomically renaming the directory. Resume accepts only an operator-selected
local checkpoint, validates all hashes and compatibility fields, restores the global step, and
records the parent checkpoint in the new run manifest.

## Test Matrix and Acceptance Evidence

### Curriculum and conversion

- Every rich case validates against the extended repository schema.
- Exactly eight required case families exist and each contains all eight response classes.
- Every flat record resolves to a source case, path, line, version, and hash.
- Rebuilding the same input produces byte-identical output.
- Preference orderings include every required relation and contain no contradictory reverse pair.
- Benign shortcuts outrank rigid method compliance when the real objective is achieved.
- Addiction language and tampering safety checks pass.

### Rewards

- Grounded success outranks proxy exploitation.
- Exploit disclosure outranks silent exploitation.
- Honest failure outranks deceptive apparent success.
- Benign creativity receives no penalty for method novelty alone.
- Repair raises feedback integrity and long-horizon agency.
- Equal aggregates retain distinguishable component records.
- Hidden-thought or hidden-state inputs are rejected by the reward-integrity provider.

### Algorithms

- PPO clipped policy loss, value loss, clipped value loss, entropy, KL, masking, and generalized
  advantage estimates match hand-computed tensors.
- GRPO group advantages, zero-variance policies, KL, and masking match hand-computed tensors.
- Required capabilities reject inference-only and value-head-free combinations before model load.

### Portable PyTorch

- The CPU test is offline and deterministic.
- The test constructs a tiny local causal model and generates trajectories.
- The reward pipeline emits every GEPA and reward-integrity component.
- Backward and optimizer step run.
- At least one selected model or LoRA parameter changes.
- Checkpoint save and reload reproduce the saved trainable state.
- Resume restores global step and continues to the next step.

### CUDA

- Tests skip cleanly when CUDA is unavailable.
- Hardware tests cover supported mixed precision, a real parameter update, checkpoint round trip,
  and distributed configuration validation.
- CUDA uses the same algorithm and `TorchPolicyBackend` implementation as CPU.

### llama.cpp, Vulkan, Mojo, and hybrid mode

- A mock server verifies request, response, grouping, metadata, and optional probability contracts.
- Missing log probabilities remain `null`.
- Inference-only backends reject backward and optimizer operations.
- Vulkan remains `unknown` or `unsupported` without positive evidence.
- Policy-version mismatch, stale rejection, down-weight logging, and atomic publication are tested.
- A mocked actor plus the tiny PyTorch learner completes a hybrid parameter-update and checkpoint
  round trip.
- Optional local integration tests are separately marked for llama.cpp, Vulkan, and Mojo.

### Regression and quality gates

- All existing tests pass.
- Ruff, Black with the 100-character Python limit, configured MyPy targets, wheel build, wheel smoke,
  and CLI smoke pass.
- Documentation uses one maturity label per path and maps every normative statement to a test,
  static check, or documented manual verification.

## Documentation Corrections

The implementation resolves these current documentation blocks:

1. `gepa_mindfulness/training/README.md` must identify dataclasses correctly and distinguish
   compatibility simulators from model-weight training.
2. `docs/execution_flow.md` must call the current model-backed GRPO operation rollout and scoring,
   not training, until the parameter-update acceptance test passes.
3. `docs/NEWCOMER_GUIDE.md` must make `gepa rl` canonical and list compatibility commands with their
   exact limitations and replacements.

The new RL guide includes exact CPU, CUDA, collection, hybrid, resume, and doctor commands. Each
command documents prerequisites, expected artifacts, success evidence, and failure behavior.

## Maturity Labels

- **Production-ready reference:** portable PyTorch CPU only after its offline acceptance test passes.
- **Production-ready NVIDIA path:** single-GPU CUDA only after its hardware acceptance test passes.
- **Optional/experimental:** CUDA DDP and FSDP until their marked integration tests pass on supported
  hardware.
- **Experimental inference:** llama.cpp/Vulkan rollout collection and evaluation.
- **Experimental hybrid RL:** Mojo/Vulkan/llama.cpp actors with a PyTorch learner.
- **Unsupported:** pure Mojo model-weight reinforcement learning unless the feasibility report and a
  separate implementation prove backward, optimizer, parameter-update, and checkpoint requirements.

## Phase Gates

Each phase must pass its focused tests and the relevant repository quality gate before the next
phase begins. A later phase may revise a shared interface only with a compatibility test and an
updated design record. No phase may claim full reinforcement learning without observed parameter
change and checkpoint reload evidence.
