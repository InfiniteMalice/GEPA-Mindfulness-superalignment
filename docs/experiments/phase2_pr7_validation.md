# Phase 2 PR 7: alignment-specific mechanistic studies

Maturity: **Experimental** software; original captures/reports remain **Diagnostic**.
Trained behavioral change, ethical circuit understanding, deceptive intent and alignment
improvement remain **Unqualified**. Base: local PR 6 commit `5fe5453`. This extends existing
diagnostics and verification owners; no inference default, legacy schema or model parameter changes.

## Inventory and API

New source: `evaluation/mechanistic_study_records.py`, `evaluation/mechanistic_studies.py`,
`synthetic_data/alignment_studies.py`. New tests: `tests/test_alignment_studies.py`,
`tests/test_study_native_execution.py`. The [typed API/protocol](../alignment_mechanistic_studies.md),
`phase2_pr7_study_probe.py`, compressed raw JSON, test logs/XML and source/artifact manifest
accompany this report. Documentation index, package data and wheel-resource expectations are
extended. The refinement is `history/2026-11-05-phase2-alignment-studies-design.md`.

The 26 authored comparisons cover grounded honesty, abstention, clarification/resumption,
laundering transformations, dual-path specificity controls, constitutional conflicts and CVA.
Exactly 13 are single-variable and 13 compound after explicit confound accounting. Host-supplied
canonical V5 cells remain intact; these are orthogonal research overlays, not new cases.
Expected actions and private factors never enter the actor input API.

The runner captures complete public behavior before native instrumentation, retains censored
failures, and authenticates exact original token/model/tokenizer/prediction bindings. Frozen backend
checks prevent diagnostic callbacks from silently changing weights. Existing stage-6 interventions
perform real forwards; the study report keeps them raw and unresolved. Existing behavioral
adjudication, CVA verification, mechanistic replay/registration and protected promotion retain
their separate authority. No inferred circuit or hidden thought becomes a reward or penalty.

## Test-first and independent review disposition

Initial tests failed on the missing runner; the initial wheel test failed on the missing guide.
Boundary tests exercise original public output retention, private oracle isolation, exact source
coordinates, actor censoring, default-off paths, native model/input/tokenizer mismatches, timing,
strict callback results, detached mutation rejection and parameter-change detection. A reporting
test failed because the workflow protocol list did not distinguish executed stages: reports now
explicitly retain NOT_RUN replication/revision and NOT_PERFORMED registration.

Fresh read-only review found two Important defects, both reproduced before fixing:

1. A CVA result from another session could be joined using only a planned slot, including onto a
   censored behavior arm. CVA data now originate inside the immutable original `PublicBehavior`.
   Its JSON response and evidence references must agree exactly. The host's `StudyCVAObservation`
   packet includes original study input/output/capture and plan/item/batch bindings alongside the
   original benchmark packet. External CVA inputs are only a complete-source cross-check.
   Unrelated, censored and duplicate-slot joins fail, and missing binding authentication remains
   unresolved. Ruling: authenticating a benchmark capture cannot authenticate a different source
   observation. The cost is requiring original structured data and distinct observation slots;
   free-text answers cannot be retroactively turned into checked certificates.
2. Some evidence contrasts also changed task instructions or conversation structure while declaring
   a single factor. Evidence task instructions now match, and date contrasts retain identical prior
   turns/roles. Wrapper pressure, multi-turn operations and value-conflict alternatives declare
   additional factors and remain compound. Ruling: a semantic disclaimer does not justify a false
   single-variable label. The cost is narrower causal claims and explicit compound analysis.

All four review reproducers failed before the fix. Five targeted review/binding tests then passed,
including a positive original-JSON CVA join and mutating-host rejection. Three further tests reject
post-capture raw-output, reference and parsed-response replacement. No Critical or separate Minor
findings were reported; none are deferred. No second review substitutes for red/green fix checks.
Final focused suite: **26 passed**, 117.37 seconds.

## Actual software probes

Reproduce from the checkout without external targets or model downloads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m docs.experiments.phase2_pr7_study_probe --enabled > study-probe.json
```

The constant script returns uncertainty on all 52 public arms and reads no private expectations.
A synthetic host compares its normalized labels against authored expectations after capture.
It matches **2/26 before labels and 2/26 after labels**. These are exact scripted-label checks,
not semantic accuracy on a trained model. All 26 pairs and all original outputs remain available.
Case-1 coordinates in this software probe are structural join fixtures, not independently justified
semantic case classifications. Dependency between families/repeats/transforms is retained.

The same original structured uncertainty outputs supply 14 CVA observations under explicit
synthetic-host binding authentication. Verified completion is **0/14**, premature abandonment
**7/14**, appropriate abstention **7/14**, and useful alternatives **0/14**. False impossibility
and unwarranted certainty are **0/14**. No action proposal or testing attempt occurs, so unauthorized
proposal, fabricated-success and attempt-reporting rates remain **null**, not fabricated zeros.
This illustrates why constant refusal/uncertainty cannot be equated with legitimate noncompletion.
All nine original finite fixtures are independently re-qualified by the existing deterministic
checker; that qualification does not establish the script's competence.

The native software control reuses stage 6's deliberately constructed GPT-2 writer/reader (seed 71).
A fully declared two-prompt lookup maps the public prompts onto artificial token inputs. This is
not a natural-language tokenizer or an alignment-specific semantic study. Baseline and diagnostic
copies produce identical public outputs on both contexts, preserving model state. The actual
positive patch effect is `5.178619384765625`, with zero random-component effect. An opposite-sign
precommitted prediction is falsified with that same measured effect. Both raw runs and original
behavior batches remain in the artifact. Synthetic-host independent replay yields scoped
INTERVENTION_SUPPORTED and CONTRADICTED findings, without registry admission or real organizational
independence. No trained, behaviorally updated or intervention-informed candidate was run.

## Quality gate, limits and proposed PR description

The maintainer-approved fallback for the unavailable named quality-gate skill applies: spec,
design, test-first implementation, formatting/lint/types, fresh review, fix validation and this
recorded gate. Beads automation was not invoked. The final full regression and source/artifact
hashes are retained in `phase2_pr7_manifest.json`, adjacent stdout and compressed per-test JUnit.

Protected full regression: **7,265 passed, 11 skipped, 18 warnings, 162 subtests passed**,
559.48 seconds, exit 0. Repository-wide Ruff and Black (855 Python files), both existing CI
mypy commands, and explicit type checks of the three modules plus probe pass. The isolated
wheel-resource test passes. Independent scalar/JSON recomputation verifies all source hashes,
report digests, 14 original CVA joins, native contrast/effect arithmetic, 56 actual public captures,
14 native measured conditions, baseline/diagnostic output identity and explicit unrun states.

The software supports controlled studies; it does not qualify an honesty, refusal, clarification,
deception or moral-value circuit. Behavioral, predictive, causal and alignment evidence remain
distinct. All human normative interpretations, semantic equivalence, trained-model tokenization,
real actor independence, cross-model generalization, CUDA and candidate training effects remain
unmeasured here. Existing protected coevolution and evaluation govern any later candidate change.

Proposed description: Add behavior-first alignment research over existing paired diagnostics,
CVA benchmarks and native interventions. Supply controlled honesty, abstention, clarification,
laundering, dual-path specificity, value-conflict and CVA fixtures; preserve exact host-classified
V5 coordinates and private oracles. Authenticate original behavioral/native/CVA source bindings,
check frozen diagnostic models, retain null/falsified/censored outcomes, and emit existing
MechanisticAuditReference records without reward or promotion authority. Include adversarial
tests and reproducible scripted/constructed-model evidence. No migration or inference-default
change is needed; trained semantic or alignment improvement is not claimed.
