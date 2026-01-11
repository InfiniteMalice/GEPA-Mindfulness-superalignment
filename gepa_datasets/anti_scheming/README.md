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

## Repository workflows

See beads/README.md and AGENTS.md for repository-level workflows.
