# Reasoning Generalization Tracer (RG-Tracer)

RG-Tracer provides lightweight utilities for reasoning-generalization diagnostics,
semantic verifier/repair workflows, attribution graphs, and epistemic alignment
reward cases.

## Abstention and epistemic alignment foundation

The compatibility reward foundation covers canonical Cases 1–13 and is documented in
`docs/epistemic_alignment.md`. Case 0 is reserved for null or fallback behavior; Cases 1–8 cover
concrete answers; Cases 9–13 cover IDK/abstention behavior. The repository-level V5 manifest also
defines canonical ambiguity Cases 14–17. The default confidence threshold remains `tau = 0.75`
unless a caller overrides it.

## Schema V3 overlay

RG-Tracer includes `rg_tracer.schema_v3`, a dependency-minimal dataclass overlay
that preserves the compatibility identity for Cases 1–13 while attaching public reasoning-unit,
control-loop, causal/scientific, MDL-control, observability, and
group-theoretic transformation diagnostics. V3 does not replace the 13 cases and
does not introduce negative hidden-thought penalties.

V3 is intended for synthetic dataset generation, DSPy-style routing, GEPA
scoring, semantic verifier/repair, factuality certification, attribution graph
and circuit-trace diagnostics, abstention calibration, semantic laundering
detection, over-refusal prevention, and transformation-stability testing.

See `docs/schema_v3.md` for the complete V1/V2/V3 relationship and practical
examples.
