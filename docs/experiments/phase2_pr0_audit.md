# Phase 2 PR 0 audit and baseline freeze

Date: 2026-11-04. Maturity: **Diagnostic** inventory; no behavioral changes in PR 0.

Baseline repository: `InfiniteMalice/GEPA-Mindfulness-superalignment`, `main` at
`fcc4142b2a5f0bdd9e9f4f7a8b90d5c72d13d923` (merge of PR #830).
The workspace initially had no checkout. A fresh clone at this SHA supplies an isolated baseline;
the new branch is `codex/phase2-pr0-audit`. Original files are preserved.

## Authoritative foundations inspected

The audit covers README, Constitution, 17-case framework/manifest, verified epistemic-process
rewards, verification/runtime authority, controlled evolution, causal diagnostics, research
ablation suite, DeLoop/PEO, attribution graph extraction/comparison/metrics, circuit-tracer
adapter, coevolution, learning surfaces, model–harness attribution, deception/dual-path paths,
testing/configuration and protected verification contracts.

There are exactly 21 numbered constitutional sections and 17 canonical V5 cases. Section 17
already teaches Constraint Respect and Least Action; graceful failure belongs before its positive
scoring signals. Section 21 already contains “When blocked, respect constraints.” The additions
must preserve all existing numbered headings and foundational values. Pre-existing text encoding
artifacts are outside the requested additive change.

## Capability matrix

“Implemented” describes code presence, not qualified efficacy. The project's maturity labels
are Supported, Implemented; hardware unqualified, Experimental, Diagnostic and Unsupported.

| Capability | Existing implementation / extension owner | Audit status and maturity | Phase 2 action |
| --- | --- | --- | --- |
| Canonical V5 | `evaluation/cases/17_case_manifest.yaml`, `V5EvaluationRecord`, `plan_v5_cells` | Implemented; Supported schema/CPU contracts | Keep `CASE × STRIPE × REPEAT`, exactly 17 cases; overlays only |
| Normative commitments | `docs/GEPA_Mindfulness_Constitution.md`, imperative/principle enums | Implemented; normative foundation | Add graceful failure within sections 17 and 21 |
| Verified process rewards | `EpistemicProcessAssessment`, `RewardProvenance`, `TrustedEvaluatorContract` | Implemented; Supported bounded software contract | No new reward or hidden-thought penalty |
| Action-bound evidence | Prediction/action/outcome/verification events; `EvidenceReference` | Implemented; Supported structural contracts | Bind new records to original IDs and references |
| Local/relational verification | `verification.interfaces`, `commit_verified_claim` | Implemented; host authentication required | Extend with candidate/feasibility distinction; no new authority owner |
| Formal reasoning | `verification.formal_reasoning` finite propositional entailment | Partially relevant; Experimental/Diagnostic | Reuse closed bounded-checking discipline; entailment is not task feasibility |
| Runtime permissions/recovery | Grant registry, `consume_authorization`, recovery store | Implemented; Supported scoped contracts | Keep missing authorization closed and one-use grants unchanged |
| Controlled promotion | Epoch/skill/Coevolution catalogs and protected receipts | Implemented; Experimental research integration | New studies use existing promotion; no diagnostic auto-admission |
| ActionOPD | `training.actionopd_runtime.build_actionopd_provider` | Implemented; Experimental CPU path, efficacy unqualified | Reuse semantic action/verified-credit boundaries |
| LoCD | `training.locd_runtime.run_local_correction` | Implemented; Experimental isolated CPU trial | Reuse protected candidate/regression lifecycle |
| SST | `training.subspace_tuning`, native RL SST config/checkpoints | Implemented; Experimental CPU, hardware/retention unqualified | Preserve source/checkpoint behavior |
| DeLoop | `training.deloop`, `deloop_runtime`, `live_loop_guard` | Implemented; Experimental audit/guard/DPO, efficacy unqualified | Preserve useful retries and original-state recovery |
| KnowGPO / PRM4RL | `training.knowledge_progress_runtime.run_knowledge_progress_trial` | Implemented; Experimental verified grouping/shaping | No substitution of diagnostic CVA/circuit credit |
| VPO | `training.vector_policy_runtime.run_vector_policy_trial` | Implemented; Experimental bounded set-level GRPO | Constitutional admissibility remains independent of weights |
| AutoSFT-style workflow | `synthetic_data.dataset_workflow.run_dataset_workflow` | Implemented; Experimental admitted data workflow | Preserve role/review/immutable data boundaries |
| RSIFlow / harness evolution | `dispatch_improvement_workflow`, `run_harness_improvement` | Implemented; Experimental bounded isolated JSON procedure trials | Reuse workflow selection and protected admission |
| Adaptive routing | `run_decision_workflow`, `decision_routing` | Implemented; Experimental authorized dispatch; native LEV Unsupported | CVA remains optional, not an alternate dispatcher |
| Causal behavioral diagnostics | `evaluation.causal_diagnostics.evaluate_causal_suite` | Implemented; Experimental/Diagnostic host-authenticated pair analysis | Reuse uncertainty/rosters; no internal causal claim |
| Model/harness attribution | `report_model_harness_attribution` | Implemented; Diagnostic matched 2×2 report | Preserve separate model, harness and interaction effects |
| Attribution graphs | `interpret.attribution_graphs`, graph comparison/metrics | Implemented discovery; Diagnostic, not causal proof | Add explicit edge evidence and actual supported interventions |
| Native neuron semantics | Existing MLP/attention block hooks average channel values; no specific neuron index | Partial discovery; Diagnostic | Add exact architecture hooks, signed regimes and tensor provenance |
| Dual-path/deception tracing | Public dual-path captures, fingerprint/probe records; circuit tracer emits empty maps and zero circuits | Partially implemented; Diagnostic stub | Reuse public comparisons; do not claim intent or actual circuits |
| V5 mechanistic overlay | `MechanisticAuditReference`, `mechanistic_circuit_audit` flag | Implemented; disabled Experimental/Diagnostic declaration | Attach genuine research artifact references using current owner |
| CVA outcome certificates | No dedicated task/candidate/feasibility/certificate contract found | Missing | Add optional bounded typed checkers |
| Proof-carrying infeasibility | No dedicated exact-domain task infeasibility certificate found | Missing | Deterministic finite examples; UNKNOWN separate from UNSAT |
| Activation simulation / downstream KL | No verified explanation-derived substitution runner found | Missing | Copied frozen models, measured logits, matched controls |
| Intervention-backed mechanism registry | No prediction-before-intervention internal causal evidence lifecycle found | Missing | Typed predictions/results, independent admission, retain nulls |
| Native versus decomposition comparison | No matched explanation/substitution comparison runner found | Missing | Optional measured comparison, reconstruction error separate |
| Alignment improvement from Phase 2 | No Phase 2 model experiment yet | Experimentally unqualified; Unsupported claim | Later reports must distinguish software and empirical evidence |

## Testing baseline

The complete unchanged-source test invocation is running. The final freeze will retain the
command, exit code, complete log, JUnit results, dependency versions and any failures before PR 0
is offered for merge. No partial progress is a passing baseline.

Environment: workspace-local CPython 3.12.16; CPU PyTorch 2.15.0+cpu; transformers 5.22.1.
Installed the declared `rl-dev,pdf,train,interpret,fusion` extras plus pytest-cov. No external
model weights, model endpoints or paid research runs are used. Baseline collection: 7,074 tests.

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -ra -q --disable-warnings --junitxml=baseline-pytest.xml
ruff check .
black --config pyproject.toml --check --diff .
```

Ruff passes; Black passes (822 files unchanged). Both CI mypy commands pass (9 source files and
the logging schema separately). The sandbox initially blocked Black's worker socket; the same
check ran successfully with approved execution outside the sandbox. This is an environment
limitation, not a formatting failure. The test suite's eventual status remains to be recorded.

## Known limits and non-duplication decisions

Existing `DECEPTION_CIRCUIT_ABLATION.md` describes an aspirational ablation workflow more strongly
than the current stub warrants. It is not evidence that deception circuits were found or removed.
Phase 2 follows current diagnostic/reward contracts and will document the actual new research
entry points. No circuit-based punishment, automatic ablation or hidden-thought reward is added.

Existing methods already have native opt-in implementation paths; the older audit's scaffold-only
status is historical, not the present inventory. Their empirical qualification remains separate.
No unavailable optional/hardware backend is silently converted to a CPU/synthetic success.

## Workflow and design

Superpowers workflows are available. The required `repo-quality-gate` skill was searched in the
active skill roots and repository but is unavailable. The maintainer explicitly approved the
documented fallback for this phase: task spec, design, implementation, tests, formatting/lint/type
checks, fresh review and recorded gate summary. No beads commands or `.beads` edits are requested.

The [architecture decision](../adr/0026-cva-causal-interpretability.md) records the selected design,
trust boundaries, alternatives and migration principles. The dependency-ordered implementation
plan is in `history/2026-11-04-phase2-implementation-plan.md`, following the repository's location
rule. This PR changes documentation and preserved audit evidence only.

Fresh design review found no critical trust/compatibility defects. It identified the need to
preserve exact specification text for downstream builders; this is addressed by
`history/2026-11-04-phase2-required-text.md`, including the verbatim constitutional addition and
all eight experimental definitions. The incorrect initial AutoSFT module path was corrected to
`synthetic_data.dataset_workflow`. Review is software/design scrutiny, not independent scientific
verification of the cited results.
