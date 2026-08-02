# Execution Flow Overview

## Stage 1 – Shell pipeline driver
`scripts/run_full_pipeline.sh` keeps the compatibility one-command workflow alive. It validates a
compatibility GRPO config, runs the CPU simulator, and invokes
`gepa_mindfulness.training.train`. The script exercises rollout and scoring paths; it does not
produce a canonical model-weight update.

## Stage 2 – CPU demo wrapper
`gepa_mindfulness/examples/cpu_demo/run_cpu_demo.py` maps the selected trainer to the
right YAML preset and delegates to `python -m gepa_mindfulness.training.cli train`. The
wrapper exists so smoke tests can call the demo from the examples directory without
worrying about PYTHONPATH tweaks. The wrapper runs a lightweight scalar-policy simulator, not a
Transformers policy.

## Stage 3 – Compatibility Click rollout and scoring CLI
`gepa_mindfulness/training/cli.py` exposes a Click group. Invoking the root command
mirrors the historical behaviour: it reads the prompt file, instantiates a configurable
`LightweightTrainingOrchestrator`, runs placeholder rollouts or dual-path probes, and writes
`rollouts.jsonl`. The `train` subcommand selects the lightweight PPO or GRPO trainers,
runs them against the chosen config, and records simulator summaries or metrics. These trainers
update only Python scalar tables; they do not load, update, or save Transformers policy weights.
The `compare` helper loads two simulator runs and prints a tabular reward summary.

## Stage 4 – Minimal training orchestrator
`gepa_mindfulness/training/pipeline.py` now provides the compatibility stub that the
legacy CLI expects. It focuses on reward shaping: computing honesty bonuses, writing
optional deception fingerprints, and returning placeholder rollouts so downstream loggers
continue to work even though full model inference is no longer embedded here.

## Stage 5 – Compatibility model-backed scoring simulator
`gepa_mindfulness/training/train.py --mode grpo` parses command-line flags, loads prompts, and
instantiates policy and reference models plus a tokenizer. `LightweightGRPOTrainer.train_epoch()`
generates responses under `torch.no_grad()` and computes rewards and advantages. The path never
runs backward or an optimizer step. It saves the unchanged loaded policy and tokenizer plus a JSON
summary. The `--mode ppo` path currently passes an argument list to the no-argument Click `main()`
function and raises `TypeError` before a simulator run.

> **Compatibility warning:** If `training.train` receives a non-local model identifier, its
> Transformers calls can download artifacts and still will not update model weights. For offline
> model-weight training, use `gepa rl` with a locally validated `policy.model_name` path.

## Stage 6 – Canonical portable RL engine

`gepa rl` is the sole canonical model-weight execution path for local PPO and GRPO. Before model
construction, `gepa_mindfulness/training/engine.py` detects the capabilities required by the
selected mode and algorithm. For training or evaluation, the engine snapshots and validates the
strict authored pair dataset, generates trajectories through `TorchPolicyBackend`, scores them
through the reward-integrity pipeline, and prepares response-token batches for the selected
algorithm.

During training, one optimizer step produces a local checkpoint and structured JSONL records.
Resume verifies artifact hashes plus dataset and configuration compatibility before it restores
model, reference, value-head, optimizer, algorithm, and random-number-generator state. A resume
budget is relative to the restored global step. Collection omits scoring and optimization;
evaluation omits optimization but retains scoring and policy evaluation.

The default model loader uses local Transformers artifacts only. See the
[portable PyTorch RL guide](rl/README.md) for exact CPU commands, prerequisites, checkpoint and log
layouts, and the full-weight/LoRA support boundary.
