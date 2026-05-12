# Reasoning Generalization Tracer (RG-Tracer)

RG-Tracer provides lightweight utilities for reasoning-generalization diagnostics,
semantic verifier/repair workflows, attribution graphs, and epistemic alignment
reward cases.

## Abstention and epistemic alignment foundation

The base reward foundation is the 13+0 abstention / hallucination /
thought-alignment schema documented in `docs/epistemic_alignment.md`. Case 0 is
reserved for null or fallback behavior; cases 1-8 cover concrete answers; cases
9-13 cover IDK/abstention behavior. The default confidence threshold remains
`tau = 0.75` unless a caller overrides it.

## 13-case Schema V3 overlay

RG-Tracer includes `rg_tracer.schema_v3`, a dependency-minimal dataclass overlay
that preserves the base 13+0 case identity while attaching public reasoning-unit,
control-loop, causal/scientific, MDL-control, observability, and
group-theoretic transformation diagnostics. V3 does not replace the 13 cases and
does not introduce negative hidden-thought penalties.

V3 is intended for synthetic dataset generation, DSPy-style routing, GEPA
scoring, semantic verifier/repair, factuality certification, attribution graph
and circuit-trace diagnostics, abstention calibration, semantic laundering
detection, over-refusal prevention, and transformation-stability testing.

See `docs/schema_v3.md` for the complete V1/V2/V3 relationship and practical
examples.
