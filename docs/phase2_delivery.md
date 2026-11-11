# Phase 2 delivery, boundaries and migration

Maturity: **Experimental** new software; **Diagnostic** research records; empirical alignment,
semantic explanation and trained-model improvement are **Unqualified**. This map is a delivery
reference, not evidence that all proposed empirical research has run.

## Required outputs

| Output | Reviewable artifact |
|---|---|
| Architecture and trust boundaries | [ADR 0026](adr/0026-cva-causal-interpretability.md), [CVA](constraint_verified_alignment.md), [causal evidence](causal_mechanistic_evidence.md), [source custody](research_reproducibility.md) |
| Exact constitutional diff | [Insertion-only diff](experiments/phase2_constitution.diff), [PR 1 validation](experiments/phase2_pr1_validation.md) |
| File inventory | [Baseline-to-phase-2 inventory](experiments/phase2_file_inventory.txt); stage 9 inventory is finalized with its validation manifest |
| Typed APIs | CVA guide; [native discovery](mechanistic_research.md), [simulation](activation_simulation.md), causal guide, [alignment studies](alignment_mechanistic_studies.md), [comparisons](comparative_interpretation.md), source-custody guide |
| Reproducible finite benchmarks | `synthetic_data/cva_fixtures.py`, `evaluation/cva_benchmark.py`, [fixture results](experiments/phase2_pr3_fixtures.json) |
| Interpretability protocol | Observe → Hypothesize → Predict → Intervene → Measure → Replicate → Revise; native/causal guides, distinct generation/development/locked data, frozen predictions and host registration |
| Actual simulation/causal results | Stages 4–8 raw gzip probes, original logits and per-stage validation reports; random tiny models and constructed controls only |
| Test logs and regression | Baseline freeze plus `experiments/phase2_prN_pytest.log`, compressed JUnit where available, source manifests and per-stage validation |
| Limitations | [Integrated results](experiments/phase2_results.md) and each API guide; software checks, predictive validity, causal validity and alignment improvement remain separate |
| Research/source traceability | [Primary sources](phase2_research_sources.md), historical pinned audit, per-source digests/model/hook/token/seed records |
| Next experiments | Integrated results: preregistered trained-model paired CVA/capability studies, semantic predictors, causal replication, actual learned decompositions |
| PR descriptions/migrations | Stage table below and individual validation reports |

## Independently reviewable stages

The phase is published through independently reviewable PRs [#831–#840](https://github.com/InfiniteMalice/GEPA-Mindfulness-superalignment/pulls). PR 0 is merged; the implementation PRs are ready for review and are merged in dependency order after current-head checks and review feedback are clear. Consult their timelines for live merge status.

The [published source provenance](phase2_publication.md) distinguishes the original audited commits, initial connector mirrors and subsequent reviewed corrections. It also explains importing the original-history bundle for pinned historical audits. Original validation snapshots are retained; their draft/local status statements describe the time those snapshots were recorded. The latest corrected software gate is [7,326 passing tests](experiments/phase2_publication_validation.md), with trained-model efficacy explicitly unrun.

| Stage / local branch suffix | Proposed PR description | Migration |
|---|---|---|
| 0 `audit` | Freeze baseline, inventory previous APIs and source status; document CVA and causal trust boundaries without behavior changes. | Documentation only. |
| 1 `graceful-failure` | Teach legitimate noncompletion in Section 17; extend examples/questions and final principle while preserving all original prose and 21 numbered sections. | Constitution prose only. |
| 2 `cva-contracts` | Add default-off typed constraints, witnessed candidate success, separately scoped finite infeasibility and evidence-limited noncompletion under host authority. | New APIs, no legacy format changes. |
| 3 `graceful-benchmark` | Add nine offline finite white-hat fixtures and original-source CVA checks as a V5 overlay. | Optional overlay; 17 canonical cases unchanged. |
| 4 `native-neurons` | Capture architecture-specific signed native measurements and test frozen explanations on held-out examples. | Optional exact GPT-2/Llama CPU adapter; unsupported backends fail. |
| 5 `activation-simulation` | Substitute causal-prefix predictions at verified hooks and compute actual KL/NLL with matched controls. | Copied frozen models only; no inference or reward default changes. |
| 6 `causal-circuits` | Add actual patch/ablation/path/restoration experiments, predicted effects, host replay verification and uncertain edge evidence. | Diagnostic overlays; existing graphs and authority owners preserved. |
| 7 `alignment-studies` | Capture behavior before instrumentation and bind original structured CVA responses across alignment-specific matched studies. | Optional research fixtures; trained conditions explicitly unrun. |
| 8 `comparative-interpretation` | Compare matched native/contextual and fixed dictionaries, measuring prediction, reconstruction, downstream effects, controls and costs separately. | Optional dictionaries; no learned-SAE or superiority claim. |
| 9 `reproducibility` | Preserve raw versioned sources with one-use scoped authority; report source-bound cluster statistics and protected failures; audit historical artifacts and finite reconstruction. | Optional POSIX single-writer archive; host persists current catalog pin. |

## Acceptance and nonclaims

Constitution tests preserve the three coequal imperatives, four wisdom values, original numbering,
placement and exact additions. CVA tests distinguish invalid candidate from scoped infeasibility,
unknown from UNSAT, unavailable proof from truth, authorization changes from bypass, and failed
search from impossibility. Synthetic fixtures have no route to external targets. All new evaluation
conditions are orthogonal V5 overlays; case 0 remains noncanonical.

Mechanistic tests isolate generation and locked data, enforce hook/shape/dtype/device/position,
compute real KL, retain signed/weak regimes and null/random controls, distinguish gradients from
actual interventions, reject unsupported families, and keep unknown evidence explicit. Native
neurons and decompositions are alternative research units. Constructed decompositions and tiny
random networks exercise software rather than semantic discovery. Host replay verification and
original event references remain necessary; language-model review alone cannot upgrade evidence.

The diagnostic condition uses a frozen model copy and measures baseline equality; it does not
modify model parameters. Behavioral training and separately authorized intervention-informed
candidates remain unrun. Existing ActionOPD, VPO, KnowGPO, AutoSFT, PRM4RL, RSIFlow, LoCD, SST,
DeLoop, routing and harness evolution remain under their prior governance and maturity categories.
Positive-only verified epistemic-process rewards are unchanged. No hidden-thought penalties,
activation fingerprint targets, direct inferred-deception punishment or mechanistic optimizer
credit is introduced. Existing protected regressions and promotion gates remain authoritative.

A software regression pass does not establish retained model capability. No trained comparison,
qualified statistical improvement, moral decision procedure, universal clean-neuron conclusion,
causally explained deception intent or general mathematical alignment theorem is claimed. External
SAT/SMT proof formats, unlisted architectures and trained feature dictionaries are optional future
integrations, not silently successful backends.


Publication qualification migration: native example references now use v2 visible-prefix metadata.
Legacy records retain exact readable wire formats; rebuild references and freeze a new protocol
for qualified held-out execution. The original bundled experiments remain historical diagnostics,
unqualified for causal-prefix isolation. New synthetic probe records and their limits are recorded
separately. Stages 0–3 are merged; later PR status is tracked by the linked PR timelines.

Latest qualification gate: [7,335-test report and corrected probes](experiments/phase2_prefix_validation.md).

Final reviewed software gate: [7,338-test validation and preserved evidence](experiments/phase2_prefix_reviewed_validation.md).

Final publication evidence: [7,338-test gate and final review correction](experiments/phase2_final_validation.md).
PR timelines establish merge status after this report was generated.

The [completion validation](experiments/phase2_completion_validation.md) records the 7,339-test
gate, predictive-evidence correction and final provenance annotations. PR timelines record merges.
