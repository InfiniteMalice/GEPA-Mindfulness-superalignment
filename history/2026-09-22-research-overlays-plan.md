# Experimental research overlays: design and implementation plan

The supplied task specification requests four additive research capabilities. Baseline commit:
`d33dc7c956c098e124b1e6b8a097f5e50227b167`. The canonical recommendations end at REC-015.
This plan applies the superpowers planning/testing workflow and repo-quality-gate.

## Architecture and invariants

Preserve the 17 canonical case identities, reward contracts, runtime governance, and raw evidence.
Add opt-in library APIs, without network calls or new mandatory dependencies. Prefer existing
frozen records, `VariantType`/`build_variant`, `SoTStateSnapshot`/`state_distance`, V5 provenance,
`TrainingEligibility`, and `FailureAtlas.observe`. No diagnostic creates a runtime grant,
deployment decision, training promotion, repair success, or private thought-text reward.

Use typed external audit references for composition without changing old V5 serialized records.
Keep exact and verified equivalence distinct from heuristic similarity. Attribute serialization
versus extraction faults only with independent stage evidence; otherwise preserve UNKNOWN.
Search admission uses verified semantic preservation and action-bound behavioral evidence.
Structured novelty and Pareto fitness are separate. Reject held-out inputs before adapters run.

## Implementation sequence and ownership

1. Communication audit: `evaluation/serialization_roundtrip.py` and
   `tests/test_serialization_roundtrip.py`. Add immutable expression/source/result records,
   serializer/extractor/verifier protocols, bounded deterministic expression equivalence,
   stage attribution, and laundering classification. Test exact/reordered/missing structure,
   heuristic and unknown results, and stage-specific failure evidence.
2. Formal audit: `gepa_mindfulness/verification/formal_reasoning.py` and
   `tests/test_formal_reasoning_audit.py`. Audit public claims in a bounded propositional
   fragment, separate premise grounding and validity, and implement explicitly bounded retries.
   Test invalid/unknown/unsupported claims, ungrounded valid inference, retries and authority.
3. Latent transition: `modules/semantic_intent_robustness/latent_language_transition.py` and
   `tests/test_latent_language_transition.py`. Reuse comparable SoT snapshots; report normalized
   latent, output, action deltas independently, including unavailable/proxy origins. Test both
   anomalies, incomparable spaces, unavailable states and absence of behavioral authority.
4. Evolutionary search: `modules/semantic_intent_robustness/evolutionary_atlas.py` and
   `tests/test_evolutionary_semantic_atlas.py`. Frozen strategies/phases, explicit budgets,
   deterministic mutation/crossover/genesis adapter, feature novelty, bounded local competition,
   provenance-linked generation memory, and external execution/evaluation adapters. Test hidden
   taint in all ingress paths, immutable lineage, near duplicates, semantic rejection, exhausted
   budgets, and verified FailureAtlas admission.
5. Integration: append REC-016–019 and four paper references to the authored YAML registries and
   readers. Extend disabled overlay declarations. Add COMMUNICATION_FIDELITY as a PARAPHRASE
   subtype to retain the existing stripe inventory and default evaluation matrix. Add diagnostic
   event kinds and an immutable reference-only attribution bundle linked to unchanged V5 records.
6. Validation: demonstrate missing-feature test failures, run focused new tests, semantic module
   tests, V5/failure-atlas/recommendation tests, full pytest where practical, repository ruff and
   Black checks, and project-configured mypy plus new modules. Record exact counts, skips and
   environmental limitations. Review changed code and documentation against the supplied spec.

## Review focus and limits

Reject non-finite numbers, booleans passed as budgets, mutable nested provenance, mismatched
record/strategy coordinates, and relabeled hidden source ancestry. Stage and solver adapters
are trusted host boundaries, not authentication systems. Search does not promote training data.
Formal grounding references require host authentication; logical validity is not factual truth.
Transfer ratios are descriptive and normalization-dependent. Synthetic software tests do not
establish alignment effectiveness; no model experiments are planned.
