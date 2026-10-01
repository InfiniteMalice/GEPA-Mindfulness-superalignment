# PR-13: dynamic uncertainty management

## Task spec

Train a caller-owned decision policy on causally validated PEO histories and independently
verified next-decision assessments. Compare frozen policies on disjoint non-TRAIN histories.
The eight requested behaviors are mismatch, independent support, correlated support,
contradiction, missing evidence, unresolved evidence, high stakes, and scoped success.
They are experiment strata, not new canonical cases.

## Reconciliation

| Status | Existing mechanism and decision |
| --- | --- |
| Existing | Action-bound event validator, epistemic reconciliation, immutable JSON snapshots, TRAIN admission, verified process assessment, RecommendedAction. Reuse them. |
| Partial | Scalar temporal estimator and fusion quantify declared numerical uncertainty; PR-7 guards routing proposals. Neither trains a next-decision policy. |
| Missing | Opt-in optimization of verified decisions from PEO histories and matched behavioral evaluation across the eight strata. Add this. |
| Redundant | A second epistemic state, reward ontology, runtime router, or synthetic-world implementation. Do not add them. |
| Experimental | Host-provided verifier and policy callbacks. Tests establish executable contracts and small-model learning, not LLM effectiveness. |

## Design

`training/dynamic_uncertainty.py` accepts retained provenance, an explicit split, a public
next-decision context and existing structured events ending in reconciliation. Before callbacks,
snapshot and validate the full catalog, numeric residuals, chronology, context identity and
observable evidence. Extract a narrow public history projection with no IDs, stratum labels,
source metadata, private reasoning, target actions or future events. The host must authenticate
and sanitize public context and observation content; structural validation cannot do that.

A separately supplied, versioned trusted evaluator returns an existing
`EpistemicProcessAssessment` for every canonical `RecommendedAction`. Only CALIBRATION,
CONSEQUENCE_PREDICTION, BELIEF_UPDATE, CONTRADICTION_HANDLING,
MISSING_EVIDENCE_DETECTION, JUSTIFIED_ABSTENTION and RECOVERY are allowed, with the
same component set for every action. Every component must name the supplied evaluator contract.
The evaluator must use externally verifiable decisions/outcomes, never uncertainty language,
private reasoning, internal state or residual magnitude. Numeric history remains input and
diagnostic data, not an automatically generated reward. No absent verification earns credit.

Training maximizes expected verified score under softmax action logits using a caller-owned
Torch optimizer. Validate all records and assessments before the first optimizer update.
Require all eight behavior strata and informative targets. Record source and assessment
digests, evaluator contract, seed, exposure counts and losses. No file/network effects.

`evaluation/dynamic_uncertainty.py` reuses the same preparation and verified tables, and
evaluates named/versioned policies in one seeded order. Return raw decisions, verified scores,
best-action agreement and per-stratum coverage; missing strata and missing training-catalog
split checks are explicit. Reject overlap in identity, source group and normalized public input.
Preserve the strictest non-training restriction in reports. No deployment integration.

## Acceptance and limits

Tests exercise all eight behavioral strata, chronological and residual corruption, failed
verification, private evidence, malformed contracts/components, nested restricted data,
callback isolation, duplicate/overlapping catalogs, disabled paths, finite gradients and an
actual small-model before/after improvement. Unknown uncertainty stays null in diagnostics.
Inference outputs remain proposals and cannot grant authority. Existing runtime gates,
default training, canonical 17 cases, and reward weights stay unchanged.

Research notes distinguish original source results, repository inference, hypotheses and
unmeasured experiments. Resolve the supplied search-training shorthand to the primary paper
with explicit provenance. Package the usage guide and validate installed-wheel behavior.
