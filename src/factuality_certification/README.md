# GeoCert-Inspired Factuality Certification and Over-Refusal Guard

Constraint-inspired answer certification layer for hallucination detection and over-refusal prevention.
It certifies relative to provided evidence/context (not a formal truth proof).

Modes: `off`, `shadow`, `advisory`, `gated`, `training`.

- Distinguishes refusal, abstention, scoped partial answers, and uncertainty-qualified answers.
- Preserves positive-only thought-trace reward features.
- Does not penalize hidden/internal traces.

```python
from factuality_certification import EvidenceItem, certify_answer

result = certify_answer(
    prompt="What does the paper claim?",
    answer="The paper reports 82% accuracy.",
    evidence=[EvidenceItem(id="e1", text="The paper reports 82% accuracy.", quality_score=0.9, retrieval_score=0.8)],
)
print(result.overall_label)
print(result.recommended_action)
print(result.hallucination_risk)
```
