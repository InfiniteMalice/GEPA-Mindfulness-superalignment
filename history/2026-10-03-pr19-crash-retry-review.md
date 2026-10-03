# PR19 Codex feedback: bind crash retries

Finding: discussion_r4173151066 on PR774, verified against 6f41b3d. Coevolution decisions are
idempotent by canonical input digest. The promotion transaction can roll back after the decision
commits, allowing another valid bundle identity to produce an additional decision.

## Spec and minimal design

Extract the existing read-only canonical decision preparation into a shared helper. Expose
validate_bundle() to return that input digest without writes; decide() uses the same preparation.
Before decide(), durably append an evaluation_attempt with the validated digest, exact bundle
and measured usage. Reacquire the promotion transaction and revalidate pending state and the
attempt after the commit boundary. Retries must match the attempt. A pinned attempt cannot be
replaced with a terminal failure; the host retries its original evidence. This preserves the
existing acceptance calculation and blocks input selection after a crash.

Invalid inputs rejected before attempt persistence still leave the original request pending.
After persistence, dependency failures leave the pinned attempt pending for recovery with the
same evidence. Audit inspection supplies the original bundle/usage to a restarted host. No public
response fields or defaults change. Existing unfinished requests without an attempt pin their
first completion under the new code; hosts upgrading an old crashed completion must reconcile
any pre-existing orphan decision before resuming it, because earlier attempts were not recorded.

## Validation plan

1. Reproduce different valid metric/held/protected receipt identities and changed usage after the
   actual post-decision crash; require rejection and recovery of the original decision.
2. Cover pre-decision crashes, forbidden failure replacement, read-only preflight, invalid
   canonical receipts and concurrent completion around the durable attempt boundary.
3. Run targeted tests RED/GREEN, private-promotion and coevolution suites, full offline CPU suite,
   Ruff/Black, scoped mypy, Python 3.10/line checks, and rebuilt installed-wheel smoke.
4. Update the guide and spec, commit and push to PR774. No review replies or merge are requested.

## Review ruling and evidence

The Codex finding is valid and fixed. Four changed-attempt variants (metric, held-out and protected
receipt identities, plus measured usage) cannot create a replacement decision after a crash.
Recovery after reopening preserves the original decision ID. Pre-decision crashes retain the
attempt, failure replacement is rejected, and concurrent completion appends only one result.

The initial regression run failed all six cases as expected. After implementation, the targeted
crash/preflight cases passed (8 tests), and the combined promotion/coevolution suites passed
(92 tests). Repository Ruff and Black, scoped adapter/coevolution mypy, Python 3.10 syntax,
100-column Python and whitespace checks passed. Wheel and sdist builds passed; installed-wheel
smoke verified exact modules/resources, disabled default, CLI help, no Torch import, 17 canonical
cases and 77 research references.

The full offline CPU suite passed: 4,772 passed, 18 skipped, 16 warnings in 358.52 seconds.
Command: pytest -q --disable-warnings with offline Hugging Face/Transformers and single-threaded
BLAS/OpenMP settings. No assertions or timeouts were weakened.

Documentation precision review: BLOCK in the guide's host-recovery paragraph and original spec:
the previous text implied any valid retry recovered the existing decision. Both now require the
persisted bundle and usage and describe the attempt/failure boundary. The crash and concurrency
regressions verify those requirements. No unresolved documentation findings remain.

The acceptance algorithm, public response allowlist and runtime defaults are unchanged. The two
catalogs remain separate transactions; recovery now depends on durable private attempt evidence.
