# Training Modules

The training package contains the canonical portable PyTorch RL engine and compatibility training
surfaces retained for older workflows. Start with the [portable RL guide](../../docs/rl/README.md)
for offline CPU PPO, GRPO, checkpoint resume, collection, evaluation, and capability diagnosis.

## RL maturity matrix

| Path | Maturity | Verified boundary |
| --- | --- | --- |
| Portable PyTorch CPU PPO/GRPO | Supported | Local automated training, checkpoint, and resume evidence. |
| PyTorch CUDA and distributed | Implemented; hardware unqualified | Mock/CPU contracts; native CUDA/DDP acceptance must run on target hardware. |
| llama.cpp/Vulkan actor | Experimental external runtime | Inference/collection only; native Vulkan/llama.cpp was not run here. |
| Mojo coordinator actor | Experimental external runtime | Operator-supplied configured coordinator only; checked-in source never generates. |
| Mojo/Vulkan/llama.cpp actor + PyTorch learner | Experimental hybrid | Requires `--learner pytorch`; conversion, deployment, and reload remain external. |
| Pure Mojo learner | Unsupported / no-go (3/9 supported) | `--learner mojo` fails closed; see the evidence report. |

The [pure Mojo feasibility report](../../docs/rl/mojo_learner_feasibility.md) records why only three
of nine gates are supported. `--learner mojo` raises `pure Mojo learner is unsupported`. Use
`--learner pytorch` and follow the
[hybrid runbook](../../docs/rl/README.md#train-with-the-experimental-hybrid-mojovulkan-actor) for
the exact command, bootstrap/current-manifest checks, publication audit behavior, and recovery.

`mojo/rl_coordinator/main.mojo` is a non-generating protocol/compile reference. It returns
`actor_unconfigured` for generate requests and is never a training coordinator. Supply a configured
coordinator that implements the same protocol and exact provenance. The hybrid config and Mojo
source require a source checkout and are not included in the wheel.
Native Mojo was not installed or executed on the verification host. MAX was not probed. Native
Vulkan/llama.cpp lanes were skipped. Those skips are limitations. The hybrid command does not
convert PEFT to GGUF, deploy to llama.cpp, or prove an actor reload.

## Canonical portable RL

`gepa rl` is the sole canonical command path that updates Transformers policy weights. Install
`.[rl]`, prepare a strict authored chosen/rejected JSONL dataset, and set `policy.model_name` to a
local Transformers model directory or an existing cache entry. The canonical loader passes
`local_files_only=True`; it does not download model artifacts.

The two presets in `configs/rl/` use
`policy.model_name: /absolute/path/to/local-transformers-model` as an operator-replaced template.
Before you run a preset, replace that value with the absolute path to a local model directory. The
directory must contain `config.json`, tokenizer assets accepted by `AutoTokenizer`, and either
PyTorch `.bin` or Safetensors model weights. Follow the local validation command in the
[portable RL guide](../../docs/rl/README.md). `gepa rl doctor` checks runtime capabilities; it does
not validate the model directory. Run a shipped preset from the repository root so the bundled
dataset path resolves.

```bash
gepa rl doctor --config run.cpu.ppo.yaml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl train --config run.cpu.ppo.yaml --max-steps 1
```

`engine.py` composes the dataset adapter, reward pipeline, PPO or GRPO algorithm,
`TorchPolicyBackend`, local checkpoint store, and structured JSONL logger. The CLI supports
`train`, `resume`, `collect`, `evaluate`, and `doctor`.

`runtime_config.py` defines the frozen dataclasses that validate the canonical configuration:

- `RuntimeConfig` selects the PyTorch backend and CPU or CUDA device.
- `PolicyConfig` selects the local model and generation settings.
- `AlgorithmConfig` selects PPO or GRPO and its optimization settings.
- `RewardConfig` selects reward weights and the optional reward-integrity overlay.
- `DatasetConfig` selects training and validation inputs plus their record format.
- `CheckpointConfig` selects the checkpoint directory and save cadence.
- `LoggingConfig` selects the structured-log directory and logging level.
- `RLRunConfig` combines those sections with the run seed and enforces cross-section rules.

`gepa rl train` performs full-weight updates. The backend API supports optional PEFT LoRA,
but the canonical configuration and CLI do not yet expose a LoRA selection field. See the portable
RL guide for the exact support boundary and artifact layout.

## Compatibility modules

The older modules remain available as rollout and scoring simulators for existing callers. Their
outputs are not evidence of model-weight training:

- `configs.py` defines Pydantic models and YAML loaders for all configurable
  hyper-parameters including reward weights (α, β, γ, δ).
- `pipeline.py` produces placeholder rollouts, reward scores, optional deception fingerprints,
  and dual-path evaluations. It does not run a Transformers policy.
- The Click CLI in `cli.py` writes rollouts or runs `LightweightPPOTrainer` and
  `LightweightGRPOTrainer`. Those trainers update only simulator-owned Python scalar tables; they
  do not load, update, or save Transformers policy weights.
- `train.py --mode grpo` loads a Transformers policy and reference, generates responses under
  `torch.no_grad()`, and writes reward and advantage summaries. It never runs backward or an
  optimizer step, so the saved policy is the unchanged loaded policy. Its `--mode ppo` path is not
  operational: it passes an argument list to the no-argument Click `main()` function and raises
  `TypeError` before a simulator run.

Use `gepa rl train` or `gepa rl resume` when an operation must update model weights. Verify an
update from the command result's `policy_parameters_updated` field, distinct before/after policy
checksums, and the emitted canonical checkpoint.

Use Python 3.10 or newer. The `rl` extra installs bounded PyTorch, Transformers, and PEFT versions.
TRL, Datasets, and Accelerate are not runtime dependencies of the canonical engine.

## Repository workflows

See beads/README.md and AGENTS.md for repository-level workflows.

## Semantic intent robustness integration

The repository also ships `modules/semantic_intent_robustness` for semantic
invariance training and evaluation. Use its `SemanticBatch` and
`compute_loss_breakdown()` helpers to add invariance, topic-vs-intent
contrastive, policy consistency, and abstention calibration objectives to
existing trainers without rewriting the core GEPA loop.
