# V5 consolidation implementation plan and audit

Spec: the maintainer's approved 29-section consolidation request, September 15, 2026.
Baseline: e7efa11. Use Superpowers and repo-quality-gate; no Beads mutations authorized.

## Architecture and implementation decisions

The manifest already defines exactly 1–17. Keep it byte-for-byte unchanged. Extend
`V5EvaluationRecord`, not a competing CaseResult. Keep legacy record readers and seed
derivation compatible. Register representation/laundering/reward subtypes beneath existing
stripes, and explicitly select subtypes in the existing planner.

The existing provenance validator verifies prediction/action/observation ancestry, but permits
failed results to export rewards without attribution or repair. Separate provenance validation
(which must retain failures) from optimizer admission (which must reject unrepaired failures).
Attach decomposed success, evaluator attribution, verification rung, lineage, failure family,
repair and regression references to the existing result. Unknown evidence stays unknown.

Extend failure localization with a V5 atlas that retains each observation, groups explicit
equivalent intents/families, and uses independently verified reruns for repair/regression state.
Do not infer semantic equivalence from fuzzy similarity or invent an intent classifier.

`EvidenceState` and runtime grants already implement strong snapshots and scoped authority.
Extend them with verified claim commit, revisions and provenance-preserving equivalence.
The ontology workbench already has governed proposals and noncanonical exports; extend its
context/provenance validation. Semantic relations may cycle, provenance lineage may not.

`factuality_observability/calibration.py` owns confidence fusion. Add sources there; internal
sensors may increase scrutiny but cannot raise confidence on their own. Existing thought reward
fields already require verified process assessments; preserve them as compatibility names and
prove no trace-style-only reward. Add self-serving scrutiny to existing reward-integrity diagnostics.

Training adapters currently lack explicit eligibility. Add one policy used by synthetic
adapters, GEPA compilation and RL ingress, rejecting nested held-out labels. Reuse the existing
controlled-evolution split/receipt machinery. Synthetic examples gain explicit supplied V5 cell
lineage rather than guessed case identity.

## Existing component map

| Primary system | Existing attachment points |
| --- | --- |
| V5 | evaluation/cases, v5_records, v5_runner, run_v5_framework, experimental overlays |
| Ontology | verification/state, factuality_certification/structured_knowledge, ontology workbench |
| Evaluation | alignment battery, factuality_observability, semantic_intent_robustness, objective_validator_robustness |
| Epistemic control | verification/interfaces, runtime_governance, recovery, core/clarifying_abstention, memory_safety |
| GEPA/RL | learning_surfaces, coevolution, skill_lifecycle, training/engine, reward_pipeline, dspy_modules/compile |

Trace alignment, dual-path deception, circuit tracing and attribution remain diagnostic attachments.
Research recommendations/references already live in docs/recommendations/{registry,references}.yaml;
extend those families, do not create duplicate JSON ledgers. Only ontology workbench exists under
apps: adversarial-reward-ci and causal-abstention app directories are absent. Extend corresponding
Python libraries instead. These are concrete path deviations, not architectural deviations.

## Verification sequence

Run existing suite before changes; preserve clean baseline worktree for exact reruns. Write
regressions first for subtype seeds, immutable result extensions, unsafe trajectory scoring,
unrepaired reward export, atlas grouping/repair, holdout leakage, confidence sensors and claim
authority. Run scoped tests, then full suite, changed-file Ruff/Black, documentation contracts,
manifest hash comparison and diff checks. Record environmental failures separately from regressions.

## Risks and compatibility

New metadata is additive. Old confidence is LEGACY_UNSPECIFIED, never silently calibrated.
Failed legacy optimizer exports intentionally fail closed; audited records remain readable.
No automatic semantic-equivalence verdicts, remote judge execution, training or deployment is
claimed. Hosts remain responsible for authenticating external verifiers and supplying holdout
policy; strings and local dataclasses are not a remote trust boundary.
