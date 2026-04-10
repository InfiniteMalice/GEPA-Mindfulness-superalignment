# Factuality + Epistemic Observability Module

This package extends (not replaces) the existing GEPA 13-case schema with an explicit v2 overlay for:

- atomic-fact decomposition and selective repair,
- observability-aware confidence calibration,
- verification-routing under bounded budgets,
- hallucination-taxonomy diagnostics,
- mech-interp-friendly logging and trace export.

## Design rationale

Final-answer-only scoring is insufficient for alignment-sensitive evaluation. This module separates:

1. answer outcome,
2. atomic support and provenance,
3. calibration quality,
4. routing quality,
5. abstention quality,
6. trace and taxonomy usefulness.

This keeps the system inspectable and avoids collapsing behavior into one opaque metric.

## Data flow

1. `decomposition.py`: split answer into clauses and atomic facts.
2. verify facts independently and aggregate evidence.
3. localize unsupported/contradicted fragments and repair answer.
4. `calibration.py`: fuse confidence in priority order:
   external verification > provenance > mechanistic risk > uncertainty > declared confidence.
5. `routing.py`: choose action (accept, decompose, retrieve, external checker, abstain, escalate).
6. `scoring.py`: compute decomposed score vector.
7. `logging.py`: write per-sample bundle + optional trace package.
8. `adapters/`: convert trace bundles for graph/circuit/feature tooling.

## 13-case schema v2 overlay

Keep the base `Case1..Case13` logic, then attach an observability tier:

- `O0`: text only
- `O1`: text + behavioral cues
- `O2`: text + latent uncertainty telemetry
- `O3`: text + external verification/provenance binding
- `O4`: composed verification stack
- `O5`: composed stack + mech-interp trace package

Example labels: `Case1-O0`, `Case5-O2`, `Case10-O5`.

## Supported verification actions

- accept answer,
- decompose and verify atomic facts,
- retrieve more evidence,
- route to specialized external checker,
- abstain,
- escalate to human review.

## Trace package schema

`TracePackage` includes:

- sample metadata,
- prompt/answer text,
- tokenization map,
- atomic-fact and evidence maps,
- critical span annotations,
- uncertainty series (when available),
- optional internal summaries,
- case overlay and suspected failure mode.

## Adapters

Interfaces are provided for:

- attribution-graph builders,
- circuit tracing tools,
- feature inspection / SAE tools,
- trace viewers.

Default null adapters allow graceful operation when these backends are unavailable.

## Hallucination taxonomy + guessing diagnostics

The schema includes:

- intrinsic/extrinsic axis,
- factuality/faithfulness axis,
- primary/secondary manifestation labels,
- task-specific manifestation label,
- guessing-vs-abstention pressure diagnostics.

## Threshold calibration by domain

Use `FactualityObservabilityConfig` for:

- entropy/semantic/disagreement thresholds,
- trace capture thresholds,
- verification/trace budgets,
- domain-level threshold scaling and risk levels.

## Limitations

- Decomposition currently uses lightweight rule-based splitting.
- Verification hooks use external lookups passed by caller.
- Internal tensor hooks are adapter placeholders by default.
- Routing quality still depends on upstream evidence quality.

## Pipeline integration

Use `run_v2_pipeline` from `pipeline.py` in evaluation flows. It returns:

- `CaseOverlayV2`,
- atomic decomposition outputs,
- decomposed score vector,
- schema-complete sample log bundle.
