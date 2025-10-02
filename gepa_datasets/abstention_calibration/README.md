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
