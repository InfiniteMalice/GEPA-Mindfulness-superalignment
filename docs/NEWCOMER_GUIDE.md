# GEPA Mindfulness: Newcomer Guide

This guide summarizes the current codebase so you can orient yourself quickly and decide where to contribute next.

## High-Level Layout

| Area | Purpose |
| ---- | ------- |
| [`gepa_mindfulness/core/`](../gepa_mindfulness/core) | GEPA contemplative principles, paraconsistent imperatives, reward shaping, abstention, Circuit Tracer integration, and adversarial probes that power the alignment logic. |
| [`gepa_mindfulness/training/`](../gepa_mindfulness/training) | Configuration models, GRPO and legacy PPO orchestration, CLI entry points, and
reporting helpers for alignment training loops. |
| [`gepa_mindfulness/examples/`](../gepa_mindfulness/examples) | Runnable CPU and vLLM demos that show the pipeline end-to-end. |
| [`gepa_datasets/`](../gepa_datasets) | JSONL datasets for ethical QA, OOD stress testing, anti-scheming probes, abstention calibration, and thought-trace templates. |
| [`gepa_mindfulness/configs/`](../gepa_mindfulness/configs) | YAML presets exposing reward weights (α, β, γ, δ), model choices, and runtime parameters. |
| [`gepa_mindfulness/metrics.py`](../gepa_mindfulness/metrics.py) | Data structures and aggregation utilities for GEPA practice session analytics. |
| [`scripts/`](../scripts) | Shell helpers that validate configs, run demos, and trigger adversarial sweeps. |
| [`tests/`](../tests) | Pytest coverage for the GEPA metrics module, ensuring edge cases are handled correctly. |

## Core Alignment Logic

The `core` package implements the conceptual building blocks of GEPA mindfulness alignment:

* **Contemplative principles** – `contemplative_principles.py` models the Mindfulness, Empathy, Perspective, and Agency axes, providing aggregation utilities and rationale summaries for each rollout.【F:gepa_mindfulness/core/contemplative_principles.py†L1-L41】
* **Imperatives & paraconsistency** – `imperatives.py` and `paraconsistent.py` combine signals for the Reduce Suffering, Increase Prosperity, and Increase Knowledge imperatives using dialetheic conjunction to tolerate conflicting evidence.【F:gepa_mindfulness/core/imperatives.py†L1-L55】【F:gepa_mindfulness/core/paraconsistent.py†L1-L44】
* **Abstention & honesty rewards** – `abstention.py` enforces confidence-aware abstention and computes honesty rewards driven by mindfulness and emptiness cues.【F:gepa_mindfulness/core/abstention.py†L1-L45】
* **Reward shaping** – `rewards.py` fuses task success, GEPA scores, honesty traces, hallucination penalties, and paraconsistent truth into a single PPO scalar.【F:gepa_mindfulness/core/rewards.py†L1-L36】
* **Circuit tracing & adversarial probes** – `tracing.py` wraps the optional Circuit Tracer dependency while gracefully degrading when unavailable; `adversarial.py` offers scheming-inspired OOD scenarios.【F:gepa_mindfulness/core/tracing.py†L1-L141】【F:gepa_mindfulness/core/adversarial.py†L1-L45】

These components are re-exported via `gepa_mindfulness.core.__init__` for convenient imports across the project.【F:gepa_mindfulness/core/__init__.py†L1-L22】

## Training Pipeline

The `training` package turns the alignment primitives into GRPO and PPO workflows:

* **Configuration** – `configs.py` defines dataclasses for PPO and GRPO hyperparameters, reward weights, model
  selection, and adversarial thresholds, plus YAML loaders for presets.【F:gepa_mindfulness/training/configs.py†L1-L187】
* **Orchestration** – `grpo_trainer.py` implements Group Relative Policy Optimisation with GEPA rewards, while
  `pipeline.py` maintains the legacy PPO path for comparison.【F:gepa_mindfulness/training/grpo_trainer.py†L1-L207】【F:gepa_mindfulness/training/pipeline.py†L1-L147】
* **CLI tooling** – `train.py` selects between GRPO and PPO modes; `cli.py` stays available for backwards-compatible
  PPO runs.【F:gepa_mindfulness/training/train.py†L1-L96】【F:gepa_mindfulness/training/cli.py†L1-L67】

## Integration Adapters

Adapters decouple the core logic from specific backends:

* `policy_adapter.py` exposes a `TextGenerator` protocol with concrete Hugging Face and vLLM implementations, letting you swap generation backends without touching training logic.【F:gepa_mindfulness/adapters/policy_adapter.py†L1-L41】
* `tracing_adapter.py` turns detailed `ThoughtTrace` events into compact checkpoints for logging or reward shaping downstream.【F:gepa_mindfulness/adapters/tracing_adapter.py†L1-L33】

Exports live in `gepa_mindfulness.adapters.__init__`.【F:gepa_mindfulness/adapters/__init__.py†L1-L11】

## Configurations & Examples

Two YAML presets provide CPU-friendly and vLLM-oriented defaults (`default.yaml`, `vllm.yaml`).【F:gepa_mindfulness/configs/default.yaml†L1-L18】【F:gepa_mindfulness/configs/vllm.yaml†L1-L18】 Use them with:

```bash
python -m gepa_mindfulness.training.train --mode grpo \"
  --config gepa_mindfulness/configs/default.yaml \
  --dataset path/to/prompts.txt
```

To see the system in action, run the example scripts:

* `examples/cpu_demo/run_cpu_demo.py` executes a short PPO loop on CPU, printing responses, rewards, and trace summaries.【F:gepa_mindfulness/examples/cpu_demo/run_cpu_demo.py†L1-L30】
* `examples/vllm_demo/run_vllm_demo.py` targets a vLLM endpoint defined in `configs/vllm.yaml` for remote inference.【F:gepa_mindfulness/examples/vllm_demo/run_vllm_demo.py†L1-L27】

The `scripts/run_full_pipeline.sh` helper validates configs, runs the CPU demo, then executes a GRPO pass
using the new training entry point.【F:scripts/run_full_pipeline.sh†L1-L16】
## Metrics & Testing

Outside the alignment loop, `gepa_mindfulness/metrics.py` models mindfulness practice sessions and aggregates GEPA metrics with strong validation and numerical safety checks.【F:gepa_mindfulness/metrics.py†L9-L190】 Pytest coverage in `tests/test_metrics.py` exercises weighting, validation, and edge cases to prevent regressions.【F:tests/test_metrics.py†L1-L199】

## Suggested Next Steps

1. **Extend datasets and reward shaping** – Customize prompts and tweak reward weights (α, β, γ, δ) in YAML to study how different objectives interact.
2. **Instrument new backends** – Implement additional `TextGenerator` adapters for other serving stacks (e.g., OpenAI API, local quantized models).
3. **Deepen adversarial evaluation** – Expand `core/adversarial.py` with richer scheming probes, then surface them via the CLI.
4. **Broaden testing** – Mirror the existing metrics-focused tests for `core` and `training` modules to protect the alignment logic as the project evolves.

Keep the README and this guide handy as you plan contributions; both outline how components fit together and where to plug in new work.【F:README.md†L1-L45】
