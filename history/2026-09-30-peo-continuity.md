# PR 6: PEO evidence influence and continuity

## Reconciliation and task spec

Main at 8795c3d includes PRs 1–5 and the EvoFlint overlays (REC-016–019).
Existing: public commitments, verified supersession/scope updates, historical recall,
pressure-correlated omission diagnostics, PEO causal and numeric reconciliation validation,
and explicit evidence kind/influence policy. Partial: retention checks do not measure use.
Missing: one audit linking prospective recognition/retrieval/influence to executed behavior
and subsequent retention. Redundant: another memory store, event stream, or motive scorer.
Experimental: host-observed influence stages; public observations do not establish causality.

## Design and sequence

1. Add matched stage tests before implementation. Reuse existing PEO and continuity fixtures.
2. Add an explicitly enabled adapter in `semantic_intent_robustness.peo_continuity`.
   Capture prospective stages inside the immutable prediction's structured outcome. Capture
   retrospective stages in an existing verification-linked epistemic assessment after numeric
   reconciliation. Bind both to the original commitment and evidence-use snapshot digests.
3. Reuse sequence validation and continuity's verified update/scope rules at the later proposal.
   Require original evidence before prediction, exact unit/conversation, causal ancestry, and
   monotonic timestamps. Preserve unknown telemetry. Earlier measured failures survive updates.
4. Document the contract and research transfers; preserve the EvoFlint registries and 17 cases.
5. Run focused, full, static, packaging and independent review checks; open a draft PR.

## Decisions and cost if wrong

- R1: Host observations, captured in existing events, describe evidence influence. A host that
  mislabels behavior can produce misleading diagnostics; no causal or motive claim is generated.
- R2: Bind exact snapshots and chronology. Incomplete legacy traces fail closed or remain
  unresolved, so hosts must instrument stages before they can use the new audit.
- R3: Earlier measured failures take precedence over later legitimate updates. This preserves
  history but callers must inspect the included continuity assessment for the later valid state.
- R4: Keep this adapter explicitly disabled by default and separate from reward/routing. Hosts
  must opt in; PR 7 routing and runtime integration remain deferred.

## Validation and risks

Baseline: 156 existing continuity, forgetting, evidence-use and latent-transition tests passed.
Before: available retained evidence can appear consistent despite no influence on action.
After: matched traces distinguish retention, retrieval and influence failures, legitimate update,
legitimate scope change and unresolved omission. Test wrong identity, chronology, digests,
unknown stages, ignored evidence, norms/procedures and preservation after nonzero residuals.
No reward, authorization, private reasoning, canonical cases, memory writes or model calls change.
Public schema is additive and experimental; no new dependency or automatic evaluator hook.

## Execution evidence

The new adapter tests failed on the missing module before implementation. Matched Ignore and
future-update controls subsequently failed before adding target-aware retrieval/reflection and
later-update timestamp checks. Initial implementation passed 44 stage tests; further malformed
input, supersession, exports and source-binding controls brought this to 53 tests.
Research inventory tests failed on the two missing sources before adding CDR/TTSE. The focused
adapter/registry suite then passed 99 tests. Ruff, Black (590 files) and CI-selected plus new-module
mypy checks passed. Wheel and sdist built; outside-checkout wheel smoke exercised consistent and
failed action influence, JSON output, 45 references, 19 recommendations and the packaged guide.

The first full suite stopped at 1835 passed because REC-015's reader lacked the newly registered
guide/test/research links. Corrected the reader; its 7 consistency tests pass.

Independent read-only review by review_pr6 covered 8795c3d..d8dedb0 plus the added research
hypothesis/experiment table. It independently passed 99 tests and alternate parent-order,
multi-verifier, Control and Ignore controls. One Important finding: accepted supersession could
use a replacement source dated after the later decision because only the update's own sources
were checked. A new regression reproduced the failure. The fix includes accepted replacement
sources in chronology/context validation, with future-date, wrong-conversation and valid controls.
This also resolves the review's documentation BLOCK on the chronology guarantee. No Minor issues.

Final: Ruling: host authenticity and semantic observation accuracy remain host responsibilities,
reaffirming R1. Incorrect declarations can produce misleading diagnostics despite valid structure.
Final: Ruling: model effectiveness and causal identification require the documented deferred
experiments. Treating synthetic contract results as empirical validation would overstate safety.
Final: Ruling: full-suite and installed-distribution acceptance remain the implementer's duty.
Skipping either can hide integration failures or source/distribution discrepancies.

Final validation results follow below; no deferred minor findings.
