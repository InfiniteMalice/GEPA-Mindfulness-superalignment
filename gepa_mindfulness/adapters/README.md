# Adapter Utilities

Adapters provide reusable bridges between the training pipeline and external
systems:

- `policy_adapter.py` abstracts text generation so the same code path can
  target Hugging Face Transformers or a remote vLLM endpoint.
- `tracing_adapter.py` converts detailed Circuit Tracer outputs into compact
  GEPA checkpoints for logging or reward shaping.

These utilities keep the core pipeline backend-agnostic and simplify
integration with new inference engines or logging backends.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
