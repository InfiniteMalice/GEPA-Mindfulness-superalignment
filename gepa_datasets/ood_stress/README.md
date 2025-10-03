# OOD Stress Tests

Purpose: Probe robustness to distributional shift, adversarial prompts, covert goal-shifts, and prompt injection.
File: `data.jsonl`

Schema:
- id: str
- prompt: str
- shift_type: 'distribution' | 'adversarial' | 'goal_shift' | 'prompt_injection'
- gold_behavior: 'solve' | 'abstain' | 'refuse' | 'explain_risks'
- harms_expected: bool
- evaluation_notes: str
- expected_behaviors: (see Ethical QA)
