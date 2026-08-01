# Training Modules

The training package contains the canonical portable PyTorch RL engine and compatibility training
surfaces retained for older workflows. Start with the [portable RL guide](../../docs/rl/README.md)
for offline CPU PPO, GRPO, checkpoint resume, collection, evaluation, and capability diagnosis.

## Canonical portable RL

Install `.[rl]`, prepare a strict authored chosen/rejected JSONL dataset, and point
`policy.model_name` at a local Transformers model directory or an existing cache entry. The
default loader passes `local_files_only=True`; it does not download model artifacts.

```bash
gepa rl doctor --config run.cpu.ppo.yaml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  gepa rl train --config run.cpu.ppo.yaml --max-steps 1
```

`engine.py` composes the dataset adapter, reward pipeline, PPO or GRPO algorithm,
`TorchPolicyBackend`, local checkpoint store, and structured JSONL logger. The CLI supports
`train`, `resume`, `collect`, `evaluate`, and `doctor`.

The default CLI path performs full-weight updates. The backend API supports optional PEFT LoRA,
but the canonical configuration and CLI do not yet expose a LoRA selection field. See the portable
RL guide for the exact support boundary and artifact layout.

## Compatibility modules

The older modules remain available for existing callers:

- `configs.py` defines Pydantic models and YAML loaders for all configurable
  hyper-parameters including reward weights (α, β, γ, δ).
- `pipeline.py` orchestrates the PPO trainer, GEPA scoring, Circuit Tracer
  abstention, and dual-path evaluation.
- `cli.py` exposes a command line entry point for running training or
  dual-path-only sweeps.

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
