# PR 8: Skill bank and failure localization

## Task spec

Add an opt-in diagnostic adapter that annotates the existing FailureGraph with nine
failure layers using validated prediction/action/outcome/reconciliation events. Retain
multiple hypotheses and unknowns. Separate reported diagnostic claims from typed
verifier findings; neither establishes a new causal edge or authorizes a repair.

Add immutable skill cards with separate routing_description and operational_guidance,
bound to existing lifecycle artifacts. Proposals may target routing or guidance; other
layers suggest their own review paths. Foundational cards cannot be autonomously
changed or retired, including after performance improves. No new persistence authority.

Keep exactly 17 cases, default routing, rewards, event schemas, durable lifecycle,
and runtime recovery unchanged. No private reasoning as failure evidence. Public
interfaces are additive; no new dependencies. Risks are over-attribution, stale/cross-run
evidence, and treating a proposal or metadata binding as an approved skill deployment.

## Design and execution plan

1. Write failing tests for nine-layer classification, unknown/ambiguous signals,
   action ancestry, evidence provenance, default opt-out, and norm protection.
2. Implement a pure failure diagnostic sidecar and immutable skill bank/proposals.
   Lifecycle binding retains artifact identity/digest; metadata is not a validation receipt.
3. Document interfaces and an ADR. Register ARISE and connect relevant existing
   primary sources to REC-007/009. Preserve reciprocal traceability mirrors.
4. Run focused tests, typing/lint/format checks, build and installed-wheel smoke,
   then the full offline suite. Obtain one fresh whole-branch review, address valid
   findings, commit/push the feature branch, and create a draft PR.

## Review focus

Check evidence laundering through graph references, forged or mutated immutable records,
cross-action/run ancestry, unsupported promotion from residuals to causal diagnoses,
foundational-card erasure through revision, stale lifecycle artifact bindings, and
accidental changes to default authority/reward behavior.

## Decisions and validation ledger

Ruling: Use the existing isolated checkout from origin/main 2945b34 — the previous
PR is merged and the tree is clean — avoids an unnecessary checkout.

Ruling: Layer labels are diagnostic hypotheses, even with observed verifier failures —
an observation alone does not prove a root cause — callers must perform independent
review before remediation. Existing graph causal roles remain unchanged.

Ruling: Skill changes are proposals only, with foundational changes blocked — existing
lifecycle receipts retain persistence authority — deployment still needs the existing
held-out validation and commit path, and text metadata is not itself certified.

Ruling: Keep the plan and ledger in history under repository instructions; execute
the test-first steps directly in PowerShell — shell-specific bookkeeping scripts add
no validation to this repository — record each check here instead.

Pre-flight: failure diagnostics produce layer annotations consumed by skill proposals;
both use the same enum. Skill cards refer to lifecycle artifacts without changing their
serialized schema. Research metadata mirrors require exact reciprocal updates.

Task 1/2 validation: initial tests failed on the absent new modules. An adversarial
private-evidence relabeling test then failed against the first implementation; typed
source-kind preservation fixed it. The new tests and existing graph, lifecycle and
reconciliation tests pass: 193 tests. The 57 new tests cover 92% of the two modules'
statements. New-module mypy and whole-repository Ruff/Black checks pass.

Ruling: Preserve the existing lifecycle digest format sha256:<64 hex> — the metadata
must match canonical receipts exactly — a raw hex-only check was corrected after the
integration fixture exposed the mismatch.

Task 3 validation: 89 research registry and documentation consistency tests pass.
ARISE is source 52. Its performance-based rubric retirement is explicitly excluded
from foundational governance; source behavior and repository inference are separate.

Whole-branch review: one fresh gpt-6-astra reviewer found an Important provenance gap:
checking only a node's immediate event allowed reconciliation to relabel a private
reference from a typed verifier input. Independently reproduced with a failing test;
the adapter now rejects source-kind conflicts across the selected ancestry and compares
node declarations to that shared provenance. No other actionable findings were raised.

Ruling: Host producer authentication, replacement of trusted configuration, deployment
and empirical source-benchmark reproduction remain outside this pure adapter's scope —
the docs make these limits explicit — accepting those review exclusions adds no authority.

Final validation after the review fix:

- Full offline suite: 4048 passed, 18 skipped, 16 warnings (213.19 seconds).
- Focused graph/lifecycle/reconciliation compatibility: 194 passed.
- New tests: 58 passed; combined statement coverage 93%.
- Research/documentation consistency: 89 passed, also covered by the final full suite.
- Whole-repository Ruff and Black pass (598 Python files); CI-selected typing and
  new-module typing pass. Changed Python lines are at most 100 characters.
- Source distribution and wheel build pass. Installed-wheel smoke outside the checkout
  verifies source hashes, updated guide, 52 references and exactly 17 cases; CLI help passes.
- Documentation precision gate: PASS. Host-only obligations have explicit manual
  integration review checks; source results and repository hypotheses are separate.
- git diff --check passes. Remote main remains 2945b34.

A lint run during the second package build observed its temporary source copy; rerunning
after packaging completed passed. No temporary package files were edited or committed.

The completed deliverable is a draft PR on codex/skill-failure-localization. No merge,
automatic skill deployment, adaptive reward or PR-9 implementation is included.
