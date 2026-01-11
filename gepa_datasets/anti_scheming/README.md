# Anti-Scheming Probes

Purpose: Detect covert goal pursuit, evaluation awareness, and deceptive compliance.
File: `data.jsonl`

Schema:
- id: str
- prompt: str
- trap_type: 'hidden_goal' | 'eval_awareness' | 'reward_hacking'
- gold_behavior: 'explain_risks' | 'refuse' | 'solve_honestly'
- signals_expect: [str]  # phrases the thought trace should mention
- expected_behaviors: (see Ethical QA)

## Repository note

Paths are relative to the repository root unless noted.
Dual-path workflows use `run_dual_path_ablation_workflow.py` and
`src/dual_path_evaluator.py`; deprecated shims like
`run_deception_ablation_workflow.py` and `adversarial_*` entry points remain for
compatibility.
