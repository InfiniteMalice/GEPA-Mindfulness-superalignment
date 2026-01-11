# Shell Scripts

This directory gathers helper utilities and orchestration entry points. Notable
scripts include:

- `run_full_pipeline.sh` – Validates the default configuration, runs the CPU
  demo, and launches the GRPO example trainer for a quick end-to-end smoke test.
- `analyze_deception_fingerprints.py` – Summarises deception fingerprints and
  proposes circuit ablation targets from `fingerprints.jsonl` files produced by
  the training pipeline.
- `ablate_deception_circuits.py` – Applies a JSON recipe of ablation targets to
  cached fingerprints for offline what-if analysis.
- `deception_dashboard.py` – Streams deception statistics to the console for
  live monitoring during training or evaluation.
- `compare_models.py` – Aggregates fingerprint files across checkpoints to
  compare deception rates between models.
- `run_phi3_simple.py` / `run_llama3_simple.py` – Lightweight GPU sanity checks
  that download a small prompt set and run a few inference steps for Phi-3 Mini
  or Llama-3 8B.
- `train_model.py` – High-level entry point for dual-path deception fine-tuning
  against the bundled datasets.
- `labels_export.py` / `labels_import.py` – Move adjudicated labels to and from
  external tooling.

All scripts assume they are run from the repository root so relative paths to
configs, datasets, and outputs resolve correctly.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
