# 13-Case Schema V3 for RG-Tracer

13-Case Schema V3 is a **Control + Compositional Reasoning Overlay** for the
existing 13+0 abstention, hallucination, and thought-trace reward schema. It
uses `docs/epistemic_alignment.md` as the V1/V2 reward foundation.

V3 does not replace the 13 cases. It does not change case identities, does not
introduce negative hidden-thought penalties, and does not add deception penalties
to the main training path. It separates answer correctness, verification
quality, reasoning-unit use, control-loop quality, and transformation-stability
diagnostics.

## Relationship to V1 and V2

- **V1**: answer/IDK behavior, correctness, confidence, and thought alignment.
- **V2**: observability tiers O0-O5, provenance, evidence, trace packages,
  routing, repair, and certification.
- **V3**: reasoning units, control loops, causal/scientific method checks,
  MDL-control gates, and group-theoretic transformation diagnostics.

V3 remains compatible with GEPA, GRPO, PPO+GRN, DAPO-hybrid, DSPy pipelines,
circuit tracing, attribution graphs, semantic intent robustness, and factuality
certification.

## Group-Theoretic Reasoning: Symmetry, Invariance, and Equivalence

Group theory is used as a practical reasoning lens, not as a requirement that
every problem be formal algebra. It helps identify when surface transformations
preserve or change relevant structure.

It strengthens semantic laundering detection by treating paraphrases,
translations, wrappers, and multi-turn fragments as transformations over an
underlying intent representation. It strengthens over-refusal prevention by
detecting symmetry breaks: same topic does not mean same intent, same action
does not mean same authorization, and same words do not mean same risk.

It supports mechanistic interpretability by giving a vocabulary for invariant
circuits, equivalent behaviors, canonical forms, and transformation-stable
concepts.

Conceptual framing:

- **Category theory**: how reasoning transformations compose across typed
  structures.
- **Lambda calculus**: how operations bind variables and compute through
  substitution/application.
- **Causal reasoning**: how interventions, mechanisms, and consequences
  propagate.
- **Group theory**: which transformations preserve structure, which break
  symmetry, and which variants belong to the same equivalence class.

## V3 fields

A V3 result includes the unchanged `case_id`, base case name, output mode,
correctness, confidence band, `threshold_tau`, thought alignment,
observability, reasoning overlay, control overlay, causal/scientific overlay,
group-theoretic overlay, MDL-control overlay, decomposed rewards, diagnostics,
and a deterministic compact label.

The registry includes causal subtypes such as common-cause confounding and
interventionist reasoning, plus group-theoretic subtypes such as equivalence
class reasoning, normal form reasoning, inverse transformations, orbit
reasoning, isomorphism detection, and symmetry breaking.
