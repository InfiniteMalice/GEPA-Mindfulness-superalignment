# Unified research studies and reproducibility integration

Goal: prepare the master's Studies A–F and analyze independently captured outcomes through the
existing improvement-audit and clustered-statistics contracts, without launching models or
claiming empirical gains. PR9 is still under final-head CI; this isolated preparation will not
be published before PR9 merges. Current prospective base is PR811 head748d63d1e4e246a937eac416a005e467d622df88,
tree03489a83662e6efcbbe4af347a770167f7e2c31a. Reconcile the actual merge commit before publication.

Design: reuse SystemConfig, DatasetManifest, CandidateSpec, MetricSpec, EvaluationSlot,
ImprovementProtocol, AttemptJournal, ExposureRecord, FinalTestAuthorization and authentication
from evaluation/improvement_records.py and improvement_audit.py. Reuse estimate_paired and its
dependency-cluster bootstrap; random seeds/repeats do not become independent task families.
Study conditions are research overlays, never a second canonical case registry. Preserve all17
cases, case0fallback, CASE×STRIPE×REPEAT, protected evaluation, credit and host authority.

Declare six fixed study rosters (6/7/6/4/6/4 conditions) separately from runtime availability.
Explicit unsupported/not-requested conditions stay visible as unrun; no baseline substitution.
Configured means an operator declaration of an external runtime, not independent qualification,
permission to train or a claim of faithful reproduction. No worker/service/runtime import or
arbitrary executor is added. Default-off software fixtures establish contracts only.

Typed configuration pins the existing manifest, metric specs, evaluator/policy/evidence,
condition runtime declarations, common source checkpoint/training data/protected manifest,
training/inference budget digests, all seeds/repeats and source revision. A pure compiler emits
ordinary existing baseline/candidate slots. Seed is part of the condition coordinate, while
repeat retains its own index: existing statistics average within cases and clusters. Preserve
all attempts, failed/cancelled runs, costs, exposure history and final authorization externally.

A bounded host capture adapter emits a new explicit diagnostic source schema with complete
protocol/slot/case/budget/system/evaluator binding. Extend only the existing audit's closed
source reader for this schema; do not add reward eligibility or an authentication authority.
Numerical status remains observed until existing independent host authentication verifies the
capture and completed journal events. Unknown/unrun remains missing, never zero. All metrics
remain decomposed under existing MetricSpec; unit/rubric/version are retained. New source
captures and unified private reports remain HIDDEN_EVAL and grant no promotion authority.
Existing readers/serializations keep their behavior and unrelated source schemas unchanged.

Alternatives: a monolithic training runner would exceed operator authority/runtime availability;
a second statistics/provenance engine would duplicate existing tested contracts; merely listing
conditions would not validate runnable job declarations or real captured comparisons. The chosen
compiler/capture/audit integration supplies reproducibility without inventing runtime support.

Interfaces and task sequence:

1. Add evaluation/research_study_records.py with STUDIES, StudyRuntime, StudyConfiguration and
   StudyMetricCapture. Shared ImprovementRecord enforces strict versioned JSON, exact types,
   public evidence and no authority. Runtime declarations bind condition, availability/reason,
   SystemConfig, runtime version/config digest, common parent checkpoint and source references.
   Configuration embeds the existing manifest/metrics/evaluator/resampling policy and fixed
   reproducibility digests. Tests/test_research_studies.py first fails for absent modules,
   then covers complete rosters, default-off, unsupported arms, immutable serialization,
   duplicate conditions/seeds, incompatible source checkpoints and capacity before expansion.
2. Add evaluation/research_studies.py: prepare_research_study(config, enabled=False) compiles a
   detached JSON plan containing the existing ImprovementProtocol or explicit not_run status.
   Bound sizes before Cartesian expansion. Include baseline slots for each declared contrast;
   paired baseline cost is explicit, not assumed free. Canonical case definitions are only
   read through existing records, never rewritten. Test shared seed/budget/system bindings,
   fixed baseline, whole-input mutation, held-out lineage rejection, multi-method compositions
   and no executable/runtime/authority effect.
3. Add evaluation/research_study_capture.py: prepare_research_capture(plan, capture, enabled=False)
   creates DiagnosticEvidence for exact declared slots; audit_research_study wraps the existing
   audit, journal/exposure/final authorization callbacks and returns a private report with full
   condition availability and raw/verified decomposition. Plans are recompiled and compared to
   reject forged/mutated bindings before callbacks. Add one explicit new source-schema branch
   to improvement_audit._source_row, preserving old contracts. Test real emitted captures in
   the existing paired audit, withheld versus authenticated values, wrong protocol/slot/case/
   system/evaluator/budget, missing/failed runs, private evidence, final authorization, protected
   failures and independent-cluster interval behavior. Reserve complete output capacity before
   authentication callbacks and reject oversize input without side effects.
4. Add docs/research_ablation_suite.md and a disabled external-runtime example covering Studies
   A–F, nested controls/curricula, source/checkpoint/data/environment/seed/cost records, metric
   families, split independence, failure distributions, matched model/harness factorization
   and unsupported arms. Update existing integration/recommendation/source compatibility links
   and installed-wheel resources. Include the recommendation-documentation consistency tests
   in acceptance so every ordered evidence/source/test link is complete.
5. Run new/related audit/statistics/provenance/source suites, Python3.10/Black/line/import checks,
   actual wheel/resource/import smoke. Obtain ONE fresh whole-branch review required by inline
   execution. Publish a ready PR through GitHubplugin only after PR9 merge; monitor final-head
   CI/reviews, minimally fix verified findings, and merge only when clear. Do not repeat approval
   questions already answered by the master/standing workflow.

Review focus: incomplete study rosters must not hide unsupported arms; data-derived dependencies
cannot leak held-out tests into selection; untrusted status/probabilities cannot create reward or
verified evidence without the existing host contract; source identity must bind every numeric
slot; missing attempts cannot improve qualification or disappear from denominators. No circuit
placeholder zeros, hidden chain-of-thought or self-certified honesty metrics are admissible.

Maturity: Diagnostic/Experimental software only. Actual native backbones, training, API spending,
live execution/deployment and all unrun empirical comparisons remain Unsupported. Source methods
use the existing verified bibliography; no optional paper algorithm or code is imported here.
Quality-gate fallback is approved; no bd/.beads. Record baseline, RED/GREEN, rulings and final
evidence below. Repository instructions override skill defaults for prose plans/no task lists.

Baseline70passed. Study/compiler tests observed missing-module RED, then21GREEN. Capture
tests observed missing-module RED; source-type/status/reference cases then deliberately
exposed authentication calls before new-schema structural validation (3RED), and now reject
those inputs before authentication. Related core110passed in31.80s; added final authorization,
missing/failed/applicability/capacity/unrun tests23passed in35.71s separately.
Ruling: return a HIDDEN_EVAL StudyCaptureEnvelope instead of a bare legacy DiagnosticEvidence,
because its source JSON is text and stripping the envelope loses the shared training guard.
Only the host adapter unwraps after complete plan/source validation. No eligibility policy changes.
Ruling: reuse the existing audit's optional source-schema branch with strict typed binding;
legacy schemas retain behavior. Unknown/inapplicable measurements are missing, not zeros.

PR9 merged by maintainer and verified: maina71ad5d247a338754d2354c0725f1491fa26b6ff, tree03489a83662e6efcbbe4af347a770167f7e2c31a exactlymatches prepared source base. Both final-head Python/CodeRabbit checks passed, no actionablereviews/threads. Prior optionalcomment/merge/pluginandnetworkcalls stalled and wereinterrupted; do notretrythose obsoleteactions. ContinuePR10.

Final acceptance: 22-file related suite633passed,2skipped in79.58s. EightchangedPythonfiles
pass BlackAPI/Python3.10syntax/100cols and AST unused-import checks. Twoadditional capacity
regressions were observedRED for derived cost overflow and repeated exposure reasons, then
GREEN after existing journal-summary prevalidation and full candidate×metric exposure reservation.
One fresh independent whole-branch reviewer /root/review_research_studies found no actionable
findings, independently80passed in41.95s and confirmed all manifest/wheel hashes. No secondreview
needed. Actual final no-isolation wheel32resources+4runtimefiles byteequal; isolated import/defaultoff/
studyroster/registry/resource smoke passed. WheelSHA256d4f3768233e429182357cbba5d63d7f5633c3b0f345ceab154f40b183fd4402c.
Full local Torch/ruff execution unavailable; remoteCI remains the integration gate. No API/model
experiment, paid training or deployment occurred. Tests show software correctness, not research
effectiveness or supported hardware qualification. Existing protected/canonical/authority files
are outside the diff. Use ready PR and existing standing review/merge workflow.
