# ADR 0014: verified next-decision training from epistemic histories

Status: accepted for experimental use.

## Context

Existing records reconcile predictions, executed actions, observations, verification,
residuals and posterior uncertainty. Scalar estimation and guarded routing do not train
the next-decision policy. PR-13 requires evidence-sensitive behavior without rewarding
uncertainty language, private reasoning or internal estimates.

## Decision

Reuse existing action-bound sequence validation, epistemic records, TRAIN admission,
RecommendedAction and verified process components. Add immutable public projections,
a host-trusted evaluator contract, a full-information expected-score Torch optimizer,
and matched evaluation. Complete admission and assessment validation before updates.

The evaluator assesses every candidate proposal using external behavior/outcome evidence.
The host authenticates this evaluator and its sources. The new API checks structure and
contract identity; it cannot prove evaluator semantics. Require the same allowed component
set for every action so omitted components cannot inflate one action's score.

Keep numeric state diagnostics separate from reward. Eight behavior strata organize
experiments without adding canonical cases. Require all strata for training and report
missing evaluation coverage. Preserve source restrictions and check declared split overlap.

## Consequences

The integration can train a caller-owned policy with actual gradients, using optional
existing Torch. It leaves default training, runtime routing and authority gates unchanged.
Models, independent evidence, verifier authentication, deployment histories and checkpoints
belong to the host. A supplied action proposal is evaluated offline and is never executed
by this API. Real-model effects and calibration remain unmeasured.

See [usage, trust boundary and research interpretation](../dynamic_uncertainty.md).
