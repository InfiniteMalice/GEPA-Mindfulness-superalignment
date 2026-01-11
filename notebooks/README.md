# Fine-Tuning Notebooks

This folder provides CPU-runnable smoke tests and GPU-ready pipelines for
integrating GEPA tracing with Unsloth/PEFT LoRA fine-tuning.

- `ft_phi3_mini_unsloth_gepa.ipynb` – End-to-end pipeline for Phi-3 Mini with
  GEPA-aware SFT, abstention-aware evaluation, optional PPO, and report export.
- `ft_llama3_8b_unsloth_gepa.ipynb` – Mirrors the Phi-3 workflow for
  Meta-Llama-3 8B with QLoRA defaults and gradient checkpointing.

Both notebooks depend only on local datasets and configs. The final cell in each
notebook renders an offline HTML report and viewer via `gepa score` and
`gepa view`.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
