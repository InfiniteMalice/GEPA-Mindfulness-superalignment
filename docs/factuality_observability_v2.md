# 13-Case Schema v2: Architecture, Evaluation, and Instrumentation Upgrade

## Scope

This upgrade preserves the existing GEPA 13-case core taxonomy and extends each case with a
v2 overlay capturing observability, verification, routing, taxonomy, and trace diagnostics.

## Architecture summary

The new package is `gepa_mindfulness/factuality_observability/`:

- `schemas.py`: v2 overlay fields and enum contracts.
- `decomposition.py`: atomic-fact decomposition, per-fact verification, localized repair.
- `calibration.py`: confidence fusion with observability tiers.
- `routing.py`: budget-aware verification routing.
- `scoring.py`: decomposed multi-axis scores.
- `logging.py`: schema-complete sample logs + trace package format.
- `consistency.py`: paraphrase/related-query consistency diagnostics.
- `adapters/`: graph/circuit/feature/trace-viewer adapter interfaces.
- `pipeline.py`: evaluation pipeline hook.

## Case mapping: base to v2 overlay

Base cases stay intact (`Case1..Case13`, plus fallback `Case0`) and receive tier overlays:

- Case 1 (correct/high/aligned) -> `Case1-O0` to `Case1-O5` depending on available observability.
- Case 5 (incorrect/high/aligned) -> `Case5-O2` when uncertainty telemetry exists.
- Case 10 (miscalibrated grounded IDK) -> `Case10-O5` for deep trace candidates.
- Case 12 (honest grounded IDK) -> `Case12-O3` when verified abstention rationale is grounded.

## Scoring model

v2 scores remain decomposed:

- `answer_correctness_score`
- `atomic_fact_support_score`
- `attribution_precision_score`
- `provenance_binding_score`
- `calibration_score`
- `abstention_appropriateness_score`
- `routing_decision_score`
- `trace_capture_utility_score`
- `failure_mode_localization_score`
- `hallucination_taxonomy_coverage_score`
- `guessing_vs_abstention_diagnostic_score`

No monolithic score is required by design.

## Trace package specification

`TracePackage` exports reusable artifacts for attribution-graphs and circuit tracing:

- `trace_package_id`
- sample metadata
- prompt/answer text
- tokenization map
- atomic-fact map
- evidence map
- critical span annotations
- per-token uncertainty (optional)
- retrieval/provenance links
- optional model-internal summaries
- case overlay and failure mode tags
- graph candidate priority and downstream graph status

## Taxonomy and diagnostics

Added taxonomy fields:

- `hallucination_axis_intrinsic_extrinsic`
- `hallucination_axis_factuality_faithfulness`
- `hallucination_primary_type`
- `hallucination_secondary_types`
- `task_specific_hallucination_type`

Added diagnostics:

- guessing-vs-abstention pressure block,
- related-query consistency block,
- knowledge-boundary/staleness fields,
- reasoning-structure diagnostics,
- prompt/attack diagnostics,
- domain/routing diagnostics.

## Implementation plan

Implementation tasks are tracked in bd.

## Limitations

- decomposition is currently heuristic and should be swapped for stronger parsers,
- external verification integration is scaffolded, not fully provider-specific,
- mech-interp hooks are optional adapter interfaces pending backend availability.
