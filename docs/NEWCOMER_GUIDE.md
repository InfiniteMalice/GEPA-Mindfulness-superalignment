# GEPA Mindfulness: Newcomer Guide

This guide summarizes the current codebase so you can orient yourself quickly.

## High-Level Layout

- `gepa_mindfulness/core/` – contemplative principles, imperatives, reward shaping,
  abstention, Circuit Tracer integration, and dual-path probe scaffolding.
- `gepa_mindfulness/training/` – configuration models, GRPO + PPO orchestration,
  CLI entry points, and reporting helpers for training loops.
- `gepa_mindfulness/examples/` – runnable CPU and vLLM demos.
- `gepa_datasets/` – JSONL datasets for ethical QA, OOD stress testing, abstention
  calibration, and thought-trace templates.
- `gepa_mindfulness/configs/` – YAML presets with reward weights and runtime settings.
- `gepa_mindfulness/metrics.py` – aggregation utilities for GEPA practice metrics.
- `scripts/` – shell helpers for demos and dual-path sweeps.
- `tests/` – Pytest coverage for metrics and training utilities.

## Core Alignment Logic

The `core` package implements the conceptual building blocks of GEPA alignment:

The dual-path architecture compares two candidate responses for the same prompt.
It replaces legacy adversarial probes with a unified path-by-path trace and
selection workflow to surface alignment trade-offs and deception signals.

- **Contemplative principles** – `contemplative_principles.py` models the Mindfulness,
  Empathy, Perspective, and Agency axes.
- **Imperatives & paraconsistency** – `imperatives.py` and `paraconsistent.py`
  combine Reduce Suffering, Increase Prosperity, and Increase Knowledge signals.
- **Abstention & honesty rewards** – `abstention.py` enforces confidence-aware
  abstention and computes honesty rewards.
- **Reward shaping** – `rewards.py` fuses task success, GEPA scores, honesty traces,
  hallucination penalties, and paraconsistent truth into a PPO scalar.
- **Circuit tracing & dual-path probes** – `tracing.py` wraps the optional Circuit
  Tracer dependency; `dual_path.py` offers dual-path probe scaffolding.

These components are re-exported via `gepa_mindfulness.core.__init__` for convenient
imports across the project.

## Training Pipeline

The `training` package turns alignment primitives into GRPO and PPO workflows:

- **Configuration** – `configs.py` defines dataclasses for PPO and GRPO
  hyperparameters, reward weights, model selection, and dual-path thresholds.
- **Orchestration** – `grpo_trainer.py` implements GRPO with GEPA rewards, while
  `pipeline.py` maintains the legacy PPO path for comparison.
- **CLI tooling** – `train.py` selects between GRPO and PPO modes; `cli.py` stays
  available for backwards-compatible PPO runs.

## Integration Adapters

- `policy_adapter.py` exposes a `TextGenerator` protocol with Hugging Face and
  vLLM implementations.
- `tracing_adapter.py` turns detailed `ThoughtTrace` events into compact checkpoints
  for downstream logging or reward shaping.

Exports live in `gepa_mindfulness.adapters.__init__`.

## Configurations & Examples

YAML presets live under `configs/ppo/`, `configs/grpo/`, and `configs/comparison/`.
There are two CLI entry points. The recommended path is the Click-based CLI
(`gepa_mindfulness.training.cli`). The legacy entry point
(`gepa_mindfulness.training.train`) uses `--mode` and is kept for compatibility.

```bash
python -m gepa_mindfulness.training.cli \
  --config gepa_mindfulness/configs/default.yaml \
  --dataset path/to/prompts.txt
```

Example scripts:

- `examples/cpu_demo/run_cpu_demo.py --trainer grpo` executes a short CPU-friendly
  GRPO loop; swap to `--trainer ppo` for the PPO baseline.
- `examples/vllm_demo/run_vllm_demo.py` targets a vLLM endpoint defined in
  `configs/vllm.yaml` for remote inference.

The `scripts/run_full_pipeline.sh` helper validates configs, runs the CPU demo,
and executes a GRPO pass using the Click-based training CLI.

## Metrics & Testing

Outside the alignment loop, `gepa_mindfulness/metrics.py` models mindfulness practice
sessions and aggregates GEPA metrics with numerical safety checks. Pytest coverage
in `tests/test_metrics.py` exercises weighting, validation, and edge cases.

## Suggested Next Steps

1. Extend datasets and reward shaping by tweaking reward weights in YAML.
2. Instrument new backends by implementing additional `TextGenerator` adapters.
3. Deepen dual-path evaluation by expanding `core/dual_path.py`.
4. Broaden tests for `core` and `training` modules to protect alignment logic.
