# ADR 0001: Canonical 17-Case V5 Unified Architecture

## Status

Accepted

## Date

2026-09-10

## Context

The repository needs one stable identity boundary for the GEPA Mindfulness 17-Case Framework V5.
Before this decision, case identities, ambiguity behavior, and related evaluation terminology were
duplicated across code and documentation. The approved
[V5 unified architecture design](../../history/2026-09-10-gepa-v5-unified-architecture-design.md)
defines the intended boundary. This ADR records the repository policy that implements that design;
it does not replace the approved design specification.

The canonical case manifest, recommendation registry, and research registry provide machine-readable
records for this policy and its traceability links. The research registry records evidence and bounded
repository inferences. It does not establish empirical proof of the architecture.

## Decision

1. `evaluation/cases/17_case_manifest.yaml` is the canonical source for exactly 17 base cases with
   stable IDs `1` through `17`. Case `0` remains a noncanonical fallback and is not a base case.
2. Robustness stripes, repeat runs, and representation phenomena are evaluation dimensions or
   subtypes. They do not add base cases.
3. Optimizer-facing epistemic-process reward requires independent verification. Generated trace,
   reasoning, rationale, and uncertainty prose remain diagnostic and cannot independently change
   optimizer fitness.
4. Experimental overlays remain disabled and noncanonical unless a separately reviewed decision
   validates and accepts each overlay. An overlay does not create a case or optimizer reward by
   default.
5. `docs/recommendations/registry.yaml` and `docs/recommendations/references.yaml` are the
   recommendation and research traceability path for this architecture.

This decision states repository policy and design intent. It makes no empirical claim that the
architecture, its research references, or its implementation has been proven effective.

## Considered Alternatives

### Maintain handwritten case maps in each consumer

Rejected. Independent handwritten maps allow case names, IDs, and compatibility rules to drift.

### Treat stripes, repeats, or representation phenomena as additional cases

Rejected. Expanding the base-case set would violate the exactly-17-case invariant and conflate
identity with evaluation conditions.

### Reward generated reasoning or self-described uncertainty directly

Rejected. Self-generated prose lacks the independent verification required for optimizer-facing
epistemic-process reward.

### Enable experimental overlays as canonical architecture

Rejected. These overlays require separate validation and acceptance before they can change default
repository behavior.

## Consequences and Tradeoffs

Consumers must derive or check canonical case identity against the manifest. This reduces local
flexibility but makes divergence observable through the V5 registry and documentation consistency
tests.

Case `0` remains available for compatibility and fallback handling, but consumers must not count it
as a canonical case. Evaluation tooling may add stripes, repeats, and representation subtypes
without changing the base-case count.

The reward boundary can reject appealing but unverified process claims. The tradeoff is deliberate:
diagnostic signals can guide investigation, but only independently verified properties can affect
optimizer fitness.

Experimental overlays stay available for explicit, disabled-by-default research work. Their
noncanonical status prevents experimental results from silently changing the framework identity or
training objective.

## Verification and Follow-up

- `tests/test_v5_case_registry.py` verifies the ordered canonical case and stripe registries.
- `tests/test_v5_documentation_consistency.py` verifies the marked canonical case table against
  the manifest.
- `tests/test_recommendation_registry.py` and `tests/test_research_traceability.py` verify the
  recommendation and research traceability registries.
- `tests/test_documentation_links.py` includes this ADR in the consolidated Markdown link coverage.
