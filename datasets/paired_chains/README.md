# Paired Honest + Deceptive Chains Dataset

This directory packages small, hand-written scenarios that require the model to
emit both an honest and a deceptive thought chain simultaneously. The goal is
not to train the model to deceive but to produce fingerprints that allow early
baseline detectors to fire before any reward shaping.

## Schema

Each entry in `data.jsonl` is a JSON object with the fields:

- `id`: Unique identifier for the scenario.
- `prompt`: User-visible prompt that should receive an honest final answer.
- `context_profile`: Optional profile hint for system behaviour.
- `require_paired`: Boolean flag signalling that both chains must be emitted.
- `instructions_honest`: Guidance for the honest chain.
- `instructions_deceptive`: Parallel guidance for the deceptive chain.
- `harms_expected`: Whether ignoring safeguards could cause harm.
- `detection_hints`: Free-form strings to test detector coverage.

The honest chain is always the one that should be surfaced to users. The
paired, deceptive chain is captured only for offline evaluation and should
never be shown in production contexts.

## Usage

Run the baseline emitter and detectors via:

```bash
gepa paired run --data datasets/paired_chains/data.jsonl --out runs/paired/ --context safety_critical
```

This writes per-scenario traces, public answers, and detector scores that can be
explored with the local Trace Viewer.
