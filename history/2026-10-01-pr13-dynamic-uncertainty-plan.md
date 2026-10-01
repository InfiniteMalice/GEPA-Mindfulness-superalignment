# PR-13 implementation plan

Spec: [task spec](2026-10-01-pr13-dynamic-uncertainty-spec.md)

Use Superpowers executing-plans inline and repo-quality-gate. Existing isolated checkout,
branch `codex/dynamic-uncertainty-training`, base `6b28c61dd85ca61a270ce4d0a71e1fb2501d0b3b`.
The user's next-PR workflow authorizes implementation, feature-branch push and a draft PR.

## Global constraints

No Beads automation, merges, default-policy changes, new canonical cases or dependencies.
Keep host-owned evidence authentication, generated-data review and model persistence explicit.
Use optional existing Torch, finite gradients and fail-closed admission before callbacks.

### Task 1: Implement and document dynamic uncertainty experiments

**Interfaces:** Produce immutable prepared trajectories and verified action tables, consumed by
the trainer and evaluator. Reuse existing event, eligibility, routing-action and reward types.

1. Add failing tests for preparation, eight-stratum verified optimization and matched evaluation.
   Run focused pytest. Expected: import failures for the absent modules.
2. Implement shared preparation, versioned evaluator checks, real optimization, and evaluation.
   Run focused pytest. Expected: all behavioral and negative contracts pass.
3. Add source-linked guide, ADR, reciprocal research metadata and packaged guide entry.
   Run traceability tests and docs examples. Expected: consistent inventory and executable docs.
4. Run Ruff, Black, scoped mypy, Python 3.10 syntax/100-column checks, feature coverage,
   full suite, wheel build and installed-wheel smoke. Expected: green; new-code coverage >=80%.
5. Commit feature and evidence with truthful limitations. Expected: clean feature branch.

## Review focus

Check whether malicious/malformed histories or evaluator responses can earn credit; whether
public projections leak labels, private fields or later outcomes; whether rejected catalog
records can still reach callbacks or optimizer updates; whether invalid uncertainty is treated
as known; whether evaluation ordering/split checks support the claimed comparison; whether
training really changes model parameters; and whether source claims exceed implementation.
Inspect parameter/gradient edge cases and mutable callback inputs beyond the explicit tests.

## Completion

One fresh whole-branch reviewer using gpt-6-astra/high with no inherited context, followed by
one verified fix pass for important findings. Preserve rulings and deferred minors here. Push
only the feature branch, create and attach a draft PR, and report evidence and limitations.

## Evidence ledger

Pre-flight: shared preparation produces the immutable inputs and validated score tables used
by both consumers; evaluator computes split checks before any callback. No interface conflict.

Task 1 implementation evidence:

- RED: absent-module imports. GREEN: 42 feature tests including actual CPU updates.
- Additional RED-to-GREEN cases: nested holdout propagation, uninformative agreement,
  measurement binding order, and preserving the estimator normalization version.
- Full suite: 4,365 passed, 18 skipped, 16 warnings (242.38 seconds). After the final
  estimator-version projection addition, focused feature/research checks: 89 passed.
- Feature coverage: trainer 90%, evaluator 94%, combined 91% (304 statements).
- Full Ruff and Black: passed, 617 Python files. Scoped mypy: 12 modules plus separate
  logging-schema check passed. Changed Python parses as 3.10 and stays within 100 columns.
- Wheel and source distribution built. Installed wheel outside checkout matches source
  module/guide hashes; docs example, CLI, 17 cases and 63 references pass, without Torch import.
- Documentation precision gate: no unresolved BLOCK or WARN. Host evaluator authentication,
  scoring semantics, retained provenance, split checks and checkpoint recovery are explicit.
- No real-model effectiveness measured; the tiny tabular policy verifies optimization only.

## Fresh review and rulings

One gpt-6-astra/high reviewer inspected `6b28c61..933b77e`. Four Important findings:
nested validation overrides, shared callback input/report identity mutation, BF16 target
rounding, and unsupported sparse/required-closure optimizer operations. No Critical or Minor
findings. Nine new regression cases reproduced these issues before the single fix pass.
The complete suite at the reviewed commit passed again: 4,365 passed, 18 skipped (241.91s).

Final: Ruling: External verifier truthfulness/authentication remains the host's responsibility
because structural checks cannot establish it — a dishonest verifier could reward bad decisions.

Final: Ruling: Hosts sanitize permitted text/numeric inputs and names — hidden labels or secrets
could otherwise leak despite the narrow projection.

Final: Ruling: Real-model improvement and generalization remain unmeasured — contract tests
provide no evidence of production benefit.

Final: Ruling: Split checks cover retained declared provenance — stripped or renamed provenance
and undisclosed checkpoint exposure could leave contamination undetected.

Final: Ruling: Require causal prior links, not numeric equality with the preceding posterior,
because estimators may propagate state — hosts must assess the validity of numerical transitions.

Final: Ruling: Zero gradients from stationary policies or zero learning rates remain valid
caller configurations — update counts alone do not prove parameter movement or improvement.

Final: Ruling: Callback/optimizer failures retain earlier updates as documented — callers must
restore a checkpoint when partial training is unacceptable.

Final: Ruling: Host Python callbacks are not sandboxed — arbitrary external side effects remain
possible; isolated callback snapshots protect the experiment's own retained inputs and identity.


Final: fixed all four Important review findings in one pass. Nine new regression cases
failed before the fixes and passed afterward: exact nested component/provenance/contract
validation; dictionary-free callback records; disposable inputs for evaluator, comparison
arms and repeated training visits; stable evaluator identity; BF16 preference gradients;
sparse optimizer updates; and pre-callback rejection of required-closure optimizers.

The full fix-pass run found one registry/reader link mismatch (4,373 tests passed). The
existing documentation consistency test reproduced it; adding the new regression-test link
restored the reader/registry contract. No unrelated code changed.

Final focused validation: 98 passed; trainer 90%, evaluator 94%, combined 90% coverage
(334 statements). Ruff, Black (618 files), scoped mypy and Python 3.10/100-column checks
passed. The final wheel matches current source/guide and retains 17 cases and 63 references.

Final suite after the fix pass: 4,374 passed, 18 skipped, 16 warnings (220.46 seconds).
No unresolved review findings or deferred minors remain. All eight scope rulings above stand.
