# ADR 0015: Independent evaluation stages and visible severe events

Status: Accepted for an experimental, opt-in offline API.

## Context

Existing calibration, trajectory and relation-flip evaluators measure different behaviors.
A shared aggregate could conceal rare failures or imply that prediction accuracy establishes
action quality, honest reporting or mechanism recovery. Optional measurements also make
an observed-only denominator vulnerable to silently missing hard cases.

## Decision

Add one reporting layer over a host-declared probe roster and observable measurements.
Reuse existing identities, evidence types, training restrictions and calibration functions.
Fix metric semantics and stage assignment; keep cohort, severity and unit groups separate.
Report missing/censored probes and severe rows explicitly. Expose no combined fitness,
competence promotion, action authorization or mechanism-recovery claim.

Host adapters remain responsible for capture, causal validation, semantic judgments and
opportunity selection. They can adapt the existing paired evaluator's results without
duplicating its judgment logic. This boundary avoids creating a second evaluation runner.

## Consequences

Reports are auditable at probe granularity and can expose a severe failure amid many ordinary
successes. They cannot authenticate host claims, detect omitted opportunities, or establish
population risk. Censored latencies remain visible but require a separate survival analysis
for time-to-event inference. The host must retain the original protocol and source evidence.
Twenty-six metrics do not create new canonical cases or require every metric on every task.
No new dependency, training reward or runtime gate is introduced.

The [guide](../evaluation_ladder.md) defines denominators and host review obligations;
`tests/test_evaluation_ladder.py` covers report behavior and an existing evaluator integration.
