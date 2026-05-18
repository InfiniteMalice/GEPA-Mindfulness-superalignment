# Constitutional Training Data

This directory contains datasets derived from the existing
[GEPA Mindfulness Constitution](../../docs/GEPA_Mindfulness_Constitution.md).
The constitution remains the canonical source of truth; files here translate its
principles into training, fine-tuning, synthetic-data, and evaluation artifacts.

The goal is to train models to reason from virtues and principles, not merely
imitate approved outputs. Examples should make the "why" explicit: how a
response supports human prosperity, reduces suffering, protects scientific
knowledge, preserves autonomy, and remains corrigible under pressure.

The constitution is a living alignment document, not a replacement for system
prompts, legal requirements, safety policies, or deployment guardrails. Stricter
rules always override constitution-derived examples.

## Recommended Dataset Categories

1. **difficult_advice** - Advice where the helpful path is emotionally or
   practically difficult. Examples should preserve agency while naming risks,
   uncertainty, and trade-offs.
2. **semantic_laundering** - Requests that reframe harmful intent in sanitized
   language. Examples should detect the underlying risk rather than relying on
   surface wording.
3. **multi_turn_laundering** - Dialogues where harmful intent emerges across
   turns. Examples should track context, resist gradual escalation, and redirect
   safely.
4. **value_decomposition** - Cases that explicitly separate prosperity,
   suffering reduction, scientific knowledge, and autonomy. Examples should show
   how values interact instead of collapsing them into one metric.
5. **temporal_diffuse_harm** - Requests whose harms are delayed, cumulative, or
   distributed across many people. Examples should reason beyond immediate user
   benefit.
6. **corrigibility_and_oversight** - Cases involving correction, oversight, or
   institutional review. Examples should reward humility, auditability, and
   willingness to defer to appropriate human judgment.
7. **honest_uncertainty** - Situations where evidence is incomplete or contested.
   Examples should separate fact, inference, speculation, and needed verification.
8. **refusal_redirection** - Unsafe requests that require a refusal paired with a
   helpful alternative. Examples should avoid operational detail while preserving
   respect and useful next steps.
9. **intelligent_disobedience** - Requests where literal compliance would violate
   deeper values or safety constraints. Examples should explain the principled
   reason for noncompliance and offer safer help.
10. **autonomy_and_anti_coercion** - Cases involving manipulation, pressure, or
    dependency. Examples should strengthen informed choice and resist coercive
    optimization.
11. **scientific_integrity** - Research, measurement, or reporting tasks where
    truthfulness and reproducibility are central. Examples should protect data
    integrity and epistemic humility.
12. **grounded_compassion** - Supportive responses to distress or vulnerability.
    Examples should combine warmth with truthful limits, agency support, and
    appropriate escalation when needed.
13. **pluralism_and_moral_uncertainty** - Value conflicts where reasonable people
    may disagree. Examples should model humility, perspective taking, and
    transparent trade-off reasoning.

## Example Fields

Constitution-grounded examples should include:

- source constitution sections,
- user prompt,
- context if needed,
- ideal response,
- bad response or bad-response summary,
- principle explanation,
- values at stake,
- metacognitive checks,
- relevant risk labels.
