# Configuration Files

YAML configuration files define training hyper-parameters, reward weights, and
model selections. Two presets are provided:

- `default.yaml` – CPU-friendly settings for the minimal demo.
- `vllm.yaml` – Example configuration for integrating with an external vLLM
  engine.

Users can copy these files to craft new experiments and invoke the CLI with the
`--config` argument.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
