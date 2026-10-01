# ADR 0012: observable relation-flip diagnostics over existing worlds

Status: accepted for experimental opt-in use.

## Context

Visible correctness alone does not show that a policy responds to decisive relations.
Existing worlds provide permission counterfactuals but no shared paired evaluation.

## Decision

Reuse SyntheticWorld with exact single-boolean interventions. Add a bounded fixture
policy, six requested relation flips, consequence acceptability, and two pressure
controls. Preserve actor visibility, provenance, source restrictions and defaults.
Reject hidden, unresolved or masked decisive pairs before scoring.

Bind captured public decisions to pair, arm, prompt and SystemIdentity. Require
complete captures with observable evidence. Report visible correctness, correctly
directed changes and control invariance separately, with explicit missing coverage.
No private reasoning or internal signal serves as behavioral evidence.

## Consequences

The fixture tests behavior under interventions and does not recover internal mechanisms.
The reversible-only policy is explicit and local; the world simulator is unchanged.
Hosts authenticate captures, predeclare coverage and keep evaluator exports private.
Existing training gates reject these non-TRAIN sources. No model execution, reward
update, attribution engine, dependency or canonical case is added.
See [the guide](../relation_flips.md) for executable examples and interpretation.
