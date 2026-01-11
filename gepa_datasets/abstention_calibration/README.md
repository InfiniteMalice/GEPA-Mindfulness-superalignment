# Abstention & Calibration

Purpose: Train and evaluate the confidence head and the abstention policy (c<0.75 -> 'I don't know.').
File: `data.jsonl`

Schema:
- id: str
- question: str
- answer: str | null
- answerable: bool
- expected_behavior: 'answer' | 'abstain'
- notes: str

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
