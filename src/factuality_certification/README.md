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

## Structured Knowledge Graph Layer

Atomic factuality asks: is this individual claim supported by evidence? Structured factuality asks:
can an unsupported or contradicted claim still be reconstructed from correlated claims, evidence
dependencies, or inference paths?

The structured-knowledge layer nests atomic claims inside a lightweight relation graph. It is
disabled by default with `enable_structured_knowledge=False` and `structured_knowledge_mode="off"`.
When enabled in `shadow`, `advisory`, `gated`, or `training` modes, it logs graph summaries,
correlated claim IDs, inference path counts, maximum reconstructability, and recommended actions.

Example:

```text
Direct support score:          0.08
Indirect support score:        0.71
Reconstructability score:      0.68
Correlated knowledge risk:     high
Recommended action:            manual_review
```

This is not a formal truth proof and not a mechanistic claim about model internals. It is a
deterministic behavioral graph approximation for certification and future unlearning evaluation.
Atomic facts remain necessary; the graph layer catches indirect reconstructability that atomic
matching alone can miss.
