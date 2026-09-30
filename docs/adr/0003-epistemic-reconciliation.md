# ADR 0003: Diagnostic reconciliation in the existing action-bound stream

## Status

Experimental, explicitly invoked. 2026-09-30. Implements temporal PEO program PR-2.

## Context and decision

PR-1 snapshots validate structure but leave causal IDs unresolved. Extend the existing envelope
and sequence validator with an optional `epistemic_reconciliation` event, using those snapshots.
Validate all inputs before accepting the update. Preserve existing assessment parents and reward
gates. No parallel stream, runtime producer, estimator, fusion method or authority is introduced.

Bind each measured world residual to explicit numeric paths in immutable prediction and observed
outcome JSON. Require prediction before execution in both event order and timestamp chronology.
Bind external measurements to verifier evidence; require field-specific typed evidence for relational
verification. Legacy Boolean verification retains its observation-level contract.

## Alternatives and consequences

Opaque innovation numbers would allow fabricated comparisons despite valid IDs. Explicit selectors
make numerical equality checkable, while units and semantic correspondence remain host contracts.
Automatically deriving selectors would require domain interpretation outside this stage.

Changing existing assessment ancestry would also change V5 reward validation. Keeping reconciliation
as an optional diagnostic branch preserves compatibility and makes its authority boundary explicit.
V5 Boolean scored outcomes remain unchanged; numeric telemetry can use an additional action.

Action and report residuals need independently observed action/report schemas. They are documented
as unsupported, never synthesized as zero. Producer authentication, evidence truth, and declared
prior-state numbers are not established by reference checks. Timestamp accuracy is a host assumption.

## Research and verification

[The guide](../epistemic_state.md) specifies constructors, chronology, evidence rules and commands.
REC-002 links this module and its tests to Kalman, WMLLM/DWM and the four new PR-2 research summaries.
FTA distinguishes reporting from execution; PINNForge motivates retaining execution feedback;
C3-JEPA motivates explicit variable correspondence; AI Neuroscientist motivates ordered artifact
dependencies. These are transfer hypotheses, not evidence of semantic alignment in this repository.

`tests/test_epistemic_reconciliation.py` checks causal and numeric forgery, temporal inversions,
verifier provenance, typed field evidence, immutable snapshots, and unchanged V5 optimizer scores.
Existing event/reward suites cover legacy compatibility. No model experiment is claimed.
