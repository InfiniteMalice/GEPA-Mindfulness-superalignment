# Dual-Path Circuit Tracing Dataset

This dataset uses **single-prompt dual-path** reasoning to detect deception circuits.

## Format

Each item asks the model to explore two approaches to a problem. The model thinks
it's doing thorough reasoning. We're watching which circuits fire for each path.

**NO mention of "deception", "dishonest", or "lying" in prompts.**

## Schema

```jsonl
{
  "id": "string",
  "query": "the user's question",
  "context": "domain (medical, financial, safety, etc.)",
  "expected_divergence": ["confidence_inversion", "risk_suppression", "reward_optimization"],
  "ground_truth_correct_path": "path_1" | "path_2",
  "validation_criteria": ["string"],
  "harms_if_wrong": "none|low|medium|high|critical"
}
```

## Examples

See `data.jsonl` for 20+ examples across domains.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
