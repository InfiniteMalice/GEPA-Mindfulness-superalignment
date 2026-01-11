# Ethical QA

Purpose: Train/evaluate principled reasoning and calibrated abstention under GEPA.
File: `data.jsonl`

Schema (per line JSON):
- id: str
- question: str
- context: str | null
- choices: [str] | null
- correct: int | null  # index into choices; null -> no single correct (score by rubric)
- harms_expected: bool
- expected_behaviors:
    abstain_if_confidence_below: float (default 0.75)
    principles_targets: {mindfulness:int, emptiness:int, nonduality:int, boundless_care:int}
    imperatives_targets: {reduce_suffering:int, increase_prosperity:int, increase_knowledge:int}
- rubric_hints: [str]  # keywords the trace should mention

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
