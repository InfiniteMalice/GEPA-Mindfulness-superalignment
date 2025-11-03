# Execution Flow Overview

## Stage 1 – Shell pipeline driver
`scripts/run_full_pipeline.sh` keeps the legacy one-command workflow alive. It verifies
that the GRPO default config parses, runs the CPU demo, and then invokes the modern GRPO
entry point so tests can exercise both paths without manual setup.

## Stage 2 – CPU demo wrapper
`gepa_mindfulness/examples/cpu_demo/run_cpu_demo.py` maps the selected trainer to the
right YAML preset and delegates to `python -m gepa_mindfulness.training.cli train`. The
wrapper exists so smoke tests can call the demo from the examples directory without
worrying about PYTHONPATH tweaks.

## Stage 3 – Click-based training CLI
`gepa_mindfulness/training/cli.py` exposes a Click group. Invoking the root command
mirrors the historical behaviour: it reads the prompt file, instantiates a configurable
`TrainingOrchestrator`, runs either rollouts or adversarial probes, and writes
`rollouts.jsonl`. The `train` subcommand selects the lightweight PPO or GRPO trainers,
runs them against the chosen config, and records metrics in the trainer output directory.
The `compare` helper loads two runs and prints a tabular reward summary.

## Stage 4 – Minimal training orchestrator
`gepa_mindfulness/training/pipeline.py` now provides the compatibility stub that the
legacy CLI expects. It focuses on reward shaping: computing honesty bonuses, writing
optional deception fingerprints, and returning placeholder rollouts so downstream loggers
continue to work even though full model inference is no longer embedded here.

## Stage 5 – Model-backed GRPO entry point
`gepa_mindfulness/training/train.py` owns the modern training path. It parses command-line
flags, loads prompts (accepting plaintext or JSONL rows), ensures Transformers is
available, instantiates policy/reference models plus tokenizer, runs a single `GRPOTrainer`
epoch, saves weights and tokenizer artefacts, and writes a JSON summary with per-batch
statistics.
