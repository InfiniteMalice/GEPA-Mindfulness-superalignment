# Adapter Utilities

Adapters provide reusable bridges between the training pipeline and external
systems:

- `policy_adapter.py` abstracts text generation so the same code path can
  target Hugging Face Transformers or a remote vLLM endpoint.
- `tracing_adapter.py` converts detailed Circuit Tracer outputs into compact
  GEPA checkpoints for logging or reward shaping.

These utilities keep the core pipeline backend-agnostic and simplify
integration with new inference engines or logging backends.

## Repository workflows

See beads/README.md and AGENTS.md for repository-level workflows.
