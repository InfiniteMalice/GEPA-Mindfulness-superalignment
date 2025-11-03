# OOD Stress Prompts (Tiny Subset)

This directory contains the minimal smoke-test subset used by notebooks and CI.
The full evaluation corpus with rich out-of-distribution scenarios lives in
`gepa_datasets/ood_stress` and now includes adversarial edge cases adapted from
Florian Dietz's [EdgeCasesInAiAlignment](https://github.com/FlorianDietz/EdgeCasesInAiAlignment).

## Record Schema
Each JSONL entry exposes only the fields that may be surfaced to the model. The
full dataset adds additional evaluator-only metadata; do **not** inject those
values into prompts.

- `id`: Stable scenario identifier
- `background`: Situation brief that can be shown to the model
- `prompt`: The operative request to respond to (often adversarial)
- `shift_type`: `distribution` | `adversarial` | `prompt_injection` | `goal_shift`
- `gold_behavior`: Reference policy outcome (`solve`, `refuse`, `abstain`, `explain_risks`)
- `unanswerable`: Boolean for quick notebook heuristics

For the full dataset, additional evaluator-only fields (`telemetry_signals`,
`pressure_factors`, `expected_behaviors`, `evaluation_notes`, etc.) help judges
score behavior. These must **never** be concatenated into the model prompt.

## Usage Notes
- Load this subset only for smoke tests; rely on `gepa_datasets/ood_stress` for
  real evaluation runs.
- When rendering prompts, combine `background`, `prompt`, and optionally
  high-level telemetry hints that do not leak scoring rules.
- Run evaluations with the dual-public scratchpad template in
  `prompts/judge/DUAL_PUBLIC_WITH_SCRATCHPAD.md` so the model gets a structured
  scratchpad while remaining unaware it is being scored.
