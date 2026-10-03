# PR18: PEO intake for controlled improvement

## Intent and reconciliation

Connect diagnosed prediction/execution/outcome (PEO) failures and epistemic estimates to
the existing controlled evolution stack. Proposal content never authorizes persistence.

Existing: FailureGraph derives supported/hypothesized root-cause status; CorrectionProposal
binds one localized failure; CoevolutionStore registers candidates and verifies protected and
held-out receipts, metrics and single-use acceptance. Epochs freeze model/harness versions.
Partial: these components lack one intake operation that binds epistemic diagnostics to the
recorded source action and triages investigation. Missing: that read-only intake and its tests.
Redundant: a second candidate lifecycle or promotion store. Experimental: triage thresholds.
PR19 retains private promotion/Experiment OS scope. Global replay and active/retired runtime
deployment remain host responsibilities, not new implemented lifecycle states.

## Design

Add CoevolutionStore.correction_source_events(correction) as a read-only validation boundary.
It snapshots an exact CorrectionProposal, reopens its recorded trajectory and closed source
epoch, validates existing localization/digest/evidence rules, and checks source versions.
It returns detached canonical events; it does not claim an epoch or register a candidate.

Add controlled_improvement.assess_improvement(store, correction, estimate, triage, *,
enabled=False). Reject disabled/malformed inputs before catalog reads. Use exact typed inputs.
Match estimate run/repeat/versions to source events and action/prediction to executed action.
Observable estimate and triage evidence must intersect localized failure evidence. Unavailable
estimates may have no evidence and cause investigation. Input declarations do not prove truth.

TriageDiagnostics contains protocol_id, observable evidence_refs and seven optional unit
numbers: severity, irreversibility, recurrence, ood_novelty, systemic_effect, autonomy_impact,
reward_hacking_signal. Uncertainty remains the estimate's three separate dimensions. None
means unknown. Reject bool, numeric subclasses, nonfinite and out-of-range values. Evidence
requires 1..32 references, each identifier at most 128 UTF-8 bytes; protocol_id has the same bound.

Return detached JSON with schema_version controlled-improvement-v1, training_eligibility
DEVELOPMENT, authority_granted false, source proposal/trajectory/epoch/action IDs, estimate,
triage, root_cause_status, route, priority, reasons, and correction. Preserve all diagnostics.
Evaluate all reasons, in field order. Severity/systemic_effect >= .8 or irreversibility,
autonomy_impact/reward_hacking_signal >= .5 require human_review (urgent priority), even if
other values are missing. Otherwise any missing triage/uncertainty, uncertainty >= .5,
unsupported/unknown root cause, or multi-component correction requires investigate.
Otherwise sandbox_review includes a detached CorrectionProposal JSON for explicit later
registration. Other routes set correction null. Priority is elevated if any available triage
or uncertainty >= .5 or anything is missing, otherwise routine; human review overrides urgent.
Supported root cause is graph status, not an inferred intent or textual classification.

## Boundaries and alternatives

The host supplies one localized correction and diagnostics; this PR never invents an edit,
executes tools, modifies weights/harnesses, creates reward, or calls registration automatically.
One changed model OR harness component is eligible for sandbox review; multi-component proposals
remain available through the existing separately reviewed API. Actual edit size is host-verified.
The complete result is rejected by optimizer admission. Extracted legacy correction/estimate
schemas lack eligibility; hosts retain the envelope and never feed extracted diagnostics to
optimization. Catalog ACLs, measurement truth/calibration, causal sufficiency and runtime replay
are external responsibilities. Validation is a snapshot; registration revalidates source and
epoch state. No authority token is minted.

A new lifecycle would duplicate durable gates. A pure detached helper would not bind source
identity to catalog evidence. The small read-only store method plus adapter is chosen.

## Acceptance

Tests cover disabled and invalid inputs, all routing thresholds and missingness, root cause
qualification, private evidence, source identity/evidence/digest tampering, unchanged catalogs,
candidate registration from the prepared correction, and existing acceptance gates. No new
canonical cases (17), rewards, dependencies, or default runtime/training behavior. Python 3.10,
100-column Python, Black/Ruff and scoped mypy. Full suite, build and installed-wheel smoke.
Primary research metadata and source/inference/maturity distinctions accompany REC-010.
