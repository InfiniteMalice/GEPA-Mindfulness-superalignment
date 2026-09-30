# PR-2: causal epistemic reconciliation

## Task spec and reconciliation

User authority: the two attached PEO program specifications and the instruction to proceed
after PR #756 merged. Base: main at 1fa304d. Implement only the earliest missing stage, PR-2.
Superpowers and repo-quality-gate govern this implementation; repository instructions prohibit
automated Beads operations. This document records design and verification, not issue tracking.

PR-1 diagnostic snapshots are present. Existing action-bound envelopes and sequence validation
already enforce immutable prediction IDs, proposal/execution equality, observation ancestry,
verifier ancestry, and evaluation identity. Missing: a typed update linked to those events,
numeric residual binding, timestamp chronology, and external measurement evidence checks.

## Design

Add one optional `epistemic_reconciliation` event to the existing stream. Its strict versioned
payload contains an existing UncertaintyUpdateRecord, prediction/observation/verifier event IDs,
and one OutcomeMeasurementBinding per measurement. Each binding carries an existing InnovationRecord
and explicit object-key/array-index paths into immutable predicted and observed JSON. Compare
the selected finite numbers to the innovation and measurement values; do not infer semantics.

The sequence validator resolves all inputs from earlier validated events. Require exact parents,
same run/repeat/model/harness, matching action and prediction, retained evidence, unique update IDs,
and RFC3339 chronological ancestry with prediction strictly before execution. External verifier
measurements require observable observation evidence and provenance from a successful verifier of
that observation. Relational verifiers additionally need matching typed field evidence for outcome
support and intact provenance. Local execution alone cannot certify an outcome. Legacy Boolean
verification keeps its existing outcome-level meaning. Host authentication remains an assumption.

World residuals are supported for explicit numeric outcome paths. Proposed/executed action payloads
must already match exactly; no independent action residual is available. There is no typed later
public-report event to bind a report residual. Defer those categories until the required telemetry
exists. Mismatch remains diagnostic and says nothing about intent or deception.

Legacy assessment and reward paths stay unchanged. Include reconciliation in V5 cell identity
checks. No estimator, fusion, runtime producer, persistence, routing, reward or training change;
the canonical 17 cases stay fixed. Constructing a payload alone does not validate its ancestry.

## Implementation and verification plan

1. Write failing causal, evidence, timestamp, residual, immutability and serialization tests.
   Run pytest: expect missing reconciliation module before implementation.
2. Implement the payload/adapter and extend existing sequence, envelope and V5 identity handling.
   Run focused tests: expect valid sequences accepted and adversarial mutations rejected.
3. Update existing docs and REC-002 traceability using canonical WMLLM/DWM/Kalman summaries.
   Run focused compatibility, research registry and reward tests; lint, format and type checks;
   complete full suite and wheel smoke. Expect no legacy behavior change.
4. Obtain a fresh whole-branch review, address important findings with regression tests, commit,
   push the dedicated branch and open a draft PR. PR-3 remains the next estimator stage.

## Review focus

Check evidence laundering across verifier fields, mismatched causal IDs, mutable payload aliases,
forward references, timestamp ties/reversal, cross-run/repeat/version links, forged external source
labels, numerical path confusion, duplicate updates, and V5 identity bypass. Distinguish declared
semantics and host authentication from what this code proves.

## Execution evidence

The initial test run failed at the missing module, then the implementation passed 447 focused
contract, event, reward, V5 and research/documentation checks. Frozen envelope arrays needed the
existing JSON thaw helper at the strict PR-1 deserialization boundary. Extending the research
registry exposed a test that selected Kalman by last position; it now selects the stable reference ID.

Fresh independent review found one Important issue and no Critical or Minor findings: supplied
prior causal IDs could be unknown or contradict recorded action ancestry. Eight regression cases
failed before the fix; seven positive initial/current/historical cases passed. The fix resolves
supplied prior IDs, their context and their pairing against validated history.

Ruling: optional prior IDs may identify the current action/prediction or completed earlier history;
historical inputs precede the current prediction in stream order and are no later by timestamp.
This permits action-conditioned priors while rejecting dangling, contradictory and future history.
Cost if wrong: a future estimator may need a separately versioned convention for concurrent priors.

Review boundary rulings: producer authentication and evidence/source-kind truth remain host duties;
measurement-path semantics and units remain producer contracts; numerical priors, posterior math,
covariance assumptions and mismatch labels remain diagnostics; independent action/report residuals
await telemetry; duplicate measurements across updates await fusion/accumulation rules; model
experiment reproduction remains deferred. These boundaries are explicit in the guide. Cost if
misused: downstream callers could treat structural validity as truth, calibrated estimation or
empirical behavioral evidence; PR-3 and later consumers must add their own acceptance checks.

Documentation precision review: corrected the stale PR-2-deferred statements in the guide and
REC-002 reader, and scoped the logging guide's identifier uniqueness rule to event-owned IDs.
Normative chronology/evidence/identity rules map to reconciliation tests; host truth and semantic
checks are explicit manual responsibilities. No documentation BLOCK or WARN remains.

Final verification: 3,431 passed, 18 skipped, 16 warnings in the full offline suite after the
review correction. The new reconciliation module has 76 passing tests and 95% statement coverage.
Repository-wide Ruff and Black checks pass. Changed-module and CI-target mypy checks pass.
The rebuilt wheel installed into an isolated environment and passed causal round-trip, invalid
chronology, 27-reference registry and bundled-guide checks outside the checkout. `git diff --check`
passes. Main was refreshed and remains at 1fa304d. No canonical case, reward, routing or training
implementation changed. No Critical/Important findings remain; no Minor findings were deferred.

Final: fixed unchecked prior causal IDs — unknown, contradictory, future and cross-context
regressions RED (8 failures) to GREEN; final full suite 3431 passed. Host and semantic boundaries
above remain explicit limits, not claims supplied by the validator. PR-3 is the next missing stage.
