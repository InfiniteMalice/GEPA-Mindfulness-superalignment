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
