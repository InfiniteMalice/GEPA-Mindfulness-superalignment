# OOD Stress Tests

Purpose: Probe robustness to distributional shift, adversarial prompts, covert goal-shifts,
and prompt injection. Many scenarios are adapted from or inspired by Florian Dietz's
[EdgeCasesInAiAlignment](https://github.com/FlorianDietz/EdgeCasesInAiAlignment) corpus and
expanded with GEPA-specific telemetry for scoring.

File: `data.jsonl`

### Schema (public fields)
- `id`: Stable identifier
- `source`: Provenance tag for tracking scenario inspiration
- `background`: Situation brief suitable for the model
- `prompt`: Immediate request that the assistant must address
- `shift_type`: `distribution` | `adversarial` | `goal_shift` | `prompt_injection`
- `pressure_factors`: High-level stressors that may be surfaced verbatim
- `telemetry_signals`: Sanitised hints that do not expose scoring rubrics
- `gold_behavior`: `solve` | `abstain` | `refuse` | `explain_risks`
- `harms_expected`: Boolean indicating if misuse harms are likely

### Schema (evaluator only — keep out of prompts)
- `evaluation_notes`: Guidance for human reviewers only
- `expected_behaviors`: Calibration targets for GEPA scoring, including
  - `abstain_if_confidence_below`
  - `principles_targets`
  - `imperatives_targets`

### Prompt Construction Requirements
- Render prompts with the dual-public scratchpad template at
  `prompts/judge/DUAL_PUBLIC_WITH_SCRATCHPAD.md` to guarantee both reasoning traces
  and public answers.
- Do **not** leak `evaluation_notes`, `expected_behaviors`, or other scoring metadata
  to the model. Only expose the operational context fields above.
- Encourage situational awareness by mentioning telemetry hints while keeping the
  assistant unaware of evaluation or test framing.

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
