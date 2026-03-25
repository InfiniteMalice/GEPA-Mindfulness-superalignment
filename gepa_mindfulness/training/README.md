# Training Modules

The training package wires the GEPA core logic into an end-to-end PPO training
loop that records Circuit Tracer logs, enforces abstention, and performs dual-path
checks.

- `configs.py` defines Pydantic models and YAML loaders for all configurable
  hyper-parameters including reward weights (α, β, γ, δ).
- `pipeline.py` orchestrates the PPO trainer, GEPA scoring, Circuit Tracer
  abstention, and dual-path evaluation.
- `cli.py` exposes a command line entry point for running training or
  dual-path-only sweeps.

All modules assume Python 3.10+ with `torch`, `transformers`, and `trl`.

## Repository workflows

See beads/README.md and AGENTS.md for repository-level workflows.

## Semantic intent robustness integration

The repository also ships `modules/semantic_intent_robustness` for semantic
invariance training and evaluation. Use its `SemanticBatch` and
`compute_loss_breakdown()` helpers to add invariance, topic-vs-intent
contrastive, policy consistency, and abstention calibration objectives to
existing trainers without rewriting the core GEPA loop.
