# 13-Case Schema V3: Control + Compositional Reasoning Overlay

V3 is an additive overlay on the existing 13+0 abstention, hallucination, and
thought-trace reward schema. It does **not** replace the cases, change their
identities, introduce negative hidden-thought penalties, add deception penalties
to the main training path, or collapse reward into one monolithic scalar.
The broader 17-case framework appends cases 14-17 for high-stakes ambiguity
handling while preserving the original 13 cases and this V3 reward identity.

## V1 / V2 / V3 relationship

- **V1**: behavioral case identity: answer versus IDK, correctness, confidence,
  and thought alignment.
- **V2**: factuality/observability overlay: O0-O5 verification tier,
  provenance, evidence, trace packages, routing, repair, and certification.
- **V3**: public control and compositional-reasoning overlay: reasoning units,
  control loops, causal/scientific checks, MDL control, and transformation
  stability diagnostics.

The default confidence threshold remains `tau = 0.75` unless callers override
it. `R_token`, `R_confidence`, `R_thought`, and `R_abstain` remain decomposed.
`R_thought` is positive-only: `H * optimizer_score()` for an eligible thought-aligned case with
verified components, otherwise `0`; it is never negative.

## Data model

The package exposes dataclasses for:

- `ObservabilityOverlay`
- `ReasoningOverlay`
- `ControlOverlay`
- `CausalScientificOverlay`
- `GroupTheoreticOverlay`
- `MDLControlOverlay`
- `RewardComponents`
- `Diagnostics`
- `CaseV3Result`

`classify_case_v3(...)` accepts base V1 inputs plus optional overlays and returns
an object with the unchanged `case_id`, decomposed rewards, diagnostics,
`compact_label`, and a JSON-serializable `to_dict()` method.

## Reward augmentation

V3 adds optional additive components: `r_grounding`, `r_control`,
`r_reasoning_unit`, `r_observability`, and `r_group_theoretic`. Each component is the exact
`[0.0, 1.0]` score of a matching verified process component, or `0` when that component is
absent. The overlays are retained in the serialized diagnostic record, but populated overlay
fields alone do not produce reward. Hidden/internal thought traces are never penalized directly.

The complete optimizer-facing contract, provenance routes, compatibility aliases, and component
table are in [`docs/epistemic_process_rewards.md`](../../docs/epistemic_process_rewards.md).

## Group-Theoretic Reasoning: Symmetry, Invariance, and Equivalence

Group theory is used as a practical reasoning lens, not as a requirement that
every problem be formal algebra. It helps identify when surface transformations
preserve or change relevant structure.

It strengthens semantic laundering detection by treating paraphrases,
translations, wrappers, and multi-turn fragments as transformations over an
underlying intent representation. It strengthens over-refusal prevention by
detecting symmetry breaks: same topic does not mean same intent, same action
does not mean same authorization, and same words do not mean same risk.

It also supports mechanistic interpretability by giving a vocabulary for
invariant circuits, equivalent behaviors, canonical forms, and
transformation-stable concepts.

Conceptual framing:

- **Category theory**: how reasoning transformations compose across typed
  structures.
- **Lambda calculus**: how operations bind variables and compute through
  substitution/application.
- **Causal reasoning**: how interventions, mechanisms, and consequences
  propagate.
- **Group theory**: which transformations preserve structure, which break
  symmetry, and which variants belong to the same equivalence class.

## Synthetic data and registries

- `registry.yaml` provides a machine-readable V3 registry.
- `data/synthetic/schema_v3/examples.jsonl` contains A-M training examples,
  including more than five group-theoretic examples.

V3 is compatible with GEPA, GRPO, PPO+GRN, DAPO-hybrid, DSPy pipelines, circuit
tracing, attribution graphs, semantic intent robustness, and factuality
certification because it preserves the base case identity and adds only public,
structured metadata.

For the appended ambiguity cases, see
[`docs/17_CASE_FRAMEWORK.md`](../../docs/17_CASE_FRAMEWORK.md). They distinguish
IDK abstention from high-stakes ambiguity abstention, use assumptive proceed for
low-stakes ambiguity, and score clarify-then-resume behavior across turns.
Case 17 should not assume the user clarifies perfectly: if clarification remains
incomplete, the model should continue conditionally when possible with explicit
assumptions, foreseeable consequences if wrong, and user or authorized
decision-maker responsibility, while avoiding irreversible external action under
unresolved high-stakes ambiguity.
