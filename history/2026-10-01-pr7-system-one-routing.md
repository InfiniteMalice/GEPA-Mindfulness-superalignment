# PR-7: bounded System-One routing

## Task spec

Consume the existing action-bound epistemic reconciliation at a later proposal. Add an
explicitly enabled adapter beside the legacy factuality router. Its output is a routing
recommendation, with no execution, persistence, reward, or authority capability. Preserve
the 17-case manifest and existing default pipeline. Compare candidate backends on identical
public inputs and labels, reporting raw proposals separately from guarded decisions.

Affected surfaces: factuality routing adapter, offline routing benchmark, focused tests,
research registry/readers, package documentation, and ADR. No dependencies or event schema
changes. Host-supplied backend callbacks are trusted code; their returned proposals are data.

## Design and implementation plan

1. Add negative and positive routing tests, observe failure, then implement immutable feature
   snapshots from validated causal prefixes. Reject malformed inputs; keep absent, stale,
   changed-context, or unverified state from enabling continuation. Preserve mandatory
   verification, exhausted-budget, representation and high-impact boundaries. Keep backend
   output restricted to existing RecommendedAction values and inspect it after the callback.
2. Test and implement a common offline benchmark for named/versioned callbacks. Compare the
   existing routing baseline and uncertainty rules, retain per-case raw and guarded actions,
   failures and measured latency. No automated promotion or aggregate alignment score.
3. Document the executable interface, caller duties, research transfers and limits. Register
   six primary sources with reciprocal recommendation links and exact reader mirrors.
4. Run focused tests, lint/format/type checks, wheel smoke and the full suite. Run one fresh
   whole-branch review, verify and address important findings, then commit and open a draft PR.

## Acceptance and failure cases

High world uncertainty requests evidence; high monitor/model uncertainty requests verification;
mismatch requests hypothesis reconsideration. Low complete diagnostics plus typed bound evidence
may propose continuation only for low-risk reversible work and when the legacy router allows it.
Unknown values are not zero. Test wrong units/versions, future or superseded reconciliation,
clock age, invalid numeric/boolean context, unavailable diagnostics, callback exceptions,
malformed proposals, attempts to bypass gates, and detached audit output. Benchmark tests must
show unsafe raw proposals even when guards correct them. Evidence authenticating and relevance
remain host responsibilities; these fixtures cannot establish model quality or causal influence.

## Execution record

Base: origin/main 88b2a18 (merged PR-6 and EvoFlint). Skills: Superpowers planning, inline
execution, TDD, verification, and repo-quality-gate. Existing isolated task checkout reused.

Pre-flight: benchmark consumes the exact immutable routing features and guarded assessment
from task 1; docs and registry name the same API and tests. No incompatible interfaces found.

Ruling: benchmark installed deterministic baselines and expose the same callback contract to
JEV, CLM and classifiers; do not download model weights or fabricate learned-model results —
no configured model service exists — cost if wrong: an additional host model benchmark is needed.

Ruling: use the September 2026 controlled cybersecurity study for “Reasoning Topology Matters”
because its linear/branching/graph comparison matches the supplied program — cost if wrong:
the alternate Network-of-Thought source needs a separate traceability entry.

Ruling: restrict this stage to routing/evidence/stopping proposals through existing actions;
skill-bank persistence and tool execution remain later-stage/host duties — cost if wrong:
additional routing features will need separate evaluations.

Implementation: opt-in adapter and matched benchmark complete. The baseline 103 factuality/
reconciliation tests passed. New routing/benchmark tests were observed failing before
implementation, then passed. The synthetic comparison reports legacy raw 5/15 versus rules
15/15, with both guarded 15/15; this measures the authored policy contract only.

Independent review: one fresh gpt-6-astra reviewer found P1 stale evidence (legacy verifier
without envelope action ID; newer observation) and P2 omitted negative verifier before the
reconciliation. All three reproductions were added as failing tests and fixed in one pass.
The guard now uses validated observation ancestry, requires the latest same-action outcome,
and requires every verifier of the selected outcome. All 45 routing/benchmark tests pass;
304 related routing, provenance, authority and registry tests pass after the fixes.

Ruling: accept both review findings and conservatively request verification for uncited
current-outcome verifiers — a selected positive result must not hide another recorded finding —
cost: even an additional positive verifier requires a refreshed reconciliation.

Ruling: correct the review's minor topology venue metadata now — the primary source explicitly
states AIAIS 2027 — cost if wrong: both metadata mirrors need correction; no runtime change.

The initial full suite passed 3961 tests with 18 skips before the three review regressions
and CLM URL test were added. Final validation results are recorded in the draft PR.

Final validation: 3965 passed, 18 skipped, 16 warnings on Python 3.12.14, offline CPU settings
(227.27 seconds). Focused new-module coverage: 94% across 45 tests. Ruff passed, Black checked
594 files, CI-selected mypy plus both new modules and the registry passed, and all changed
Python lines are at most 100 characters. Source/wheel build passed; the installed wheel ran
the routing and matched benchmark smoke outside the checkout and loaded all 51 references
and the packaged guide. Documentation precision review has no unresolved BLOCK finding.
