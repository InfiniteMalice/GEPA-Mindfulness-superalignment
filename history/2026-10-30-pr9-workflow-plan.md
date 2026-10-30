# Governed improvement workflow integration

Goal: extend existing controlled-improvement intake with evidence-reviewed advisory workflow
selection and host-side matched model/harness attribution. This implements the PR 9 integration
boundary from the maintainer's master specification, using the approved quality-gate fallback.
The maintainer has directed continuous implementation, ready PRs, review, merge and continuation.

Design: retain `assess_improvement`, `classify_learning_surface`, `CoevolutionStore`, existing
epoch/receipt/acceptance records, and private promotion as the only existing authorities.
Add a default-off selector that binds typed diagnosis to the exact registered correction,
obtains an independent host-configured diagnosis review, then recommends model/data/memory/
harness/skill/verification/human/no-change work. Human and unresolved intake routes dominate.
Procedural proposals use a closed allowlist, never code paths, commands or acceptance edits.
No proposal executes, admits, deploys, grants training eligibility, or modifies a catalog.

Add a separate host-private offline report over all four model/harness combinations. Require
matched tasks/seeds, separate model/harness versions, evaluator identity, evidence and equal
per-seed compute caps using the existing ComputeBudget. Retain failures and protected regressions;
report paired effects and interaction, never choose/promote a candidate. Private reports stay
HIDDEN_EVAL and are not workflow-selector inputs. No arbitrary runner, Python sandbox, model
trainer, learned selector qualification or empirical gain is claimed.

Alternatives considered: a second recursive-evolution store would duplicate admission authority;
an unrestricted self-editing executor would cross protected boundaries; a source-only update
would not provide working experiments. The chosen extension reuses existing authority and leaves
actual isolation, metering, source authentication and deployment with the operator.

Primary sources checked: RSIFlow 2610.21791v1 sections3.1–3.3/B.1–B.5, 2026-10-15,
arXiv non-exclusive; author-linked 1HUAN1/RSI-Flow revision496d3cfb9ff39997d107d0ff29e746b07e638180,
MIT. Its released variant evolves skill memory while keeping the executable Meta program fixed.
Harness Self-Evolution2609.38372v1 sections3.1–3.5, 2026-09-29, CC BY4.0. Its engineering-check
acceptance is weaker than GEPA's protected evaluation/admission and is not imported. No authoritative
code repository established for that exact paper; a third-party link cited a different paper.
Local raw publisher HTML digests: RSIFlowc5299977786aa04f7a42ce82d618ea9b160c7d9198b41fede203f3d10debd106;
Harness3a9a95d8055024dc53707a0730b7fcc4ebb693784e42ddae4fbbdbf244d0a2a9.
No upstream code copied into the package; authored method notes distinguish adaptations.

Implementation plan (normal prose follows repository rules against Markdown tracking lists):

1. Create `tests/test_improvement_workflows.py` using existing registered-source fixtures.
   Observe missing module failure; implement `gepa_mindfulness/improvement_workflows.py` with
   `WorkflowDiagnosis`, `WorkflowReview`, and `select_improvement_workflow`. Exercise exact
   proposal binding before catalog access, independent reviewer binding, human dominance,
   uncertainty/failure, every learning surface, data/no-change, procedure allowlist, serialization,
   capacity reservation, private evidence rejection, default-off and no catalog/authority changes.
2. Create `tests/test_model_harness_attribution.py`; observe missing module failure; implement
   `evaluation/model_harness_attribution.py` with typed protocol/observations and pure
   `report_model_harness_attribution`. Exercise all four cells, task/seed/evaluator bindings,
   missing/duplicate cells, mismatched budgets, protected failures, no promotion and exact effects.
3. Add `docs/improvement_workflows.md`, disabled example config, method source notes, canonical
   bibliography/traceability/applicability links, docs packaging/resources and source checks.
   Document experimental compatibility, host trial/rollback handoff and unsupported learned/live
   extensions. Preserve all canonical cases, constitutional text, verifiers and reward eligibility.
4. Run focused and related CPU suites, formatting/Python3.10 syntax, wheel/resource/import smoke.
   Obtain one fresh whole-branch review required by the execution skill. Validate any real fixes,
   publish a ready PR via GitHub plugin, monitor final-head CI/reviews, merge only when clean.

Review focus: unsupported causal diagnoses never trigger model work; review agreement cannot
weaken intake/human routes; arbitrary protected edit targets cannot enter procedural proposals;
private paired outcomes cannot silently enter optimizer data; missing/failed factorial cells
cannot disappear from denominators. All output text is data, never executable instructions.

Ruling: implement inline with one final independent review, consistent with the standing workflow.
No local Git metadata is exposed, so use an isolated source snapshot and preserve the verified
upstream Git tree through the GitHub plugin. Do not invoke bd or edit .beads. Baseline and final
evidence will be recorded below; no supported maturity or empirical improvement is inferred.

Acceptance evidence: unchanged source baseline194passed. New adapter/attribution tests were
observed RED with missing modules before implementation; metadata test RED before registry additions.
Final related suite467passed,2skipped in35.21s. One fresh independent whole-branch reviewer found
no findings and independently102checks passed. Reviewer isolated-build provisioning failed in its
sandbox; the operator-enabled packaging test subsequently PASSED (1test in3.85s). Actual
no-isolation wheel checked31resources plus2runtime files byte-for-byte and isolated import/default-off/
reference/resource smoke; wheelSHA2568d74c717318fd223867db29ffc26ed67a215d547dec9b81bdd5bffce5aa2a7c8.
Eight changed Python files pass Black API, Python3.10 syntax and100-column checks; four new files
pass AST unused-import checks. Full local Black CLI was blocked by sandbox Unix-socket permissions,
not formatting; CI lint/full-suite remain required. Local Torch is unavailable, so no complete
local suite or model effectiveness is claimed. No upstream model training, API experiment or
deployment was run. Source-primary network fetch succeeded after scoped network access; the
earlier source-information question is resolved. Existing source/admission/rollback contracts stay intact.
