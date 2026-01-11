# Ethical QA (Tiny Subset)

This directory mirrors a tiny subset of the full `gepa_datasets/ethical_qa`
corpus so the fine-tuning notebooks can run CPU-only smoke tests without
external downloads. Each record contains a `question`, `answer`, and
`mindfulness_trace` scaffold that follows the GEPA framing \u2192 evidence \u2192
decision pattern.

For complete coverage, use the expanded dataset under `gepa_datasets/ethical_qa`.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
