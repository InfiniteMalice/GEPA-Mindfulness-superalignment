# PR-15: World-model value ablation

## Problem and reconciliation

Test whether explicit state improves behavior enough to justify its added costs. Do not infer
usefulness from interpretability. No model or checkpoint was supplied; this change delivers an
experimental host-adapter harness and deterministic controls, not measured model effectiveness.

| Status | Evidence and decision |
| --- | --- |
| Existing | `synthetic_data/worlds.py` provides public observations, evidence state and simulation. Reuse. |
| Existing | `synthetic_data/world_peo.py` validates prospective predictions and causal reconciliations. Reuse. |
| Partial | PR-13 compares decision backends; PR-14 reports independent evaluation stages. Neither runs matched world rollouts. |
| Missing | Three-arm rollout, common resource caps, public-only inputs and paired cost/outcome comparisons. Implement. |
| Redundant | New world schema, uncertainty reward, canonical cases, trained world model. Do not introduce. |
| Experimental | Model identity, resource metering and session isolation remain host responsibilities. |

## Contract

An opt-in `compare_world_models` runs DIRECT, STRUCTURED and PEO for every input case. A case
contains immutable JSON for a validated SyntheticWorld, actor, target action, cohort and severity.
The supported task is information acquisition followed by a decision to execute the target or
abstain. Auxiliary actions belong to the same actor and have no effects; only reveals are allowed.
This prevents an auxiliary action from changing the target's truth conditions to manufacture success.

Use one model/checkpoint/training contract, one host factory and identical resource caps for all
arms. Create a fresh policy session per case/arm with the same episode seed. Seeded execution order
reduces fixed-order effects. Snapshot and validate the entire input catalog before any callback.
Reject duplicate case IDs and identical initial-world digests; variants remain evaluator cases.

DIRECT receives rendered public observations and public action/outcome history. STRUCTURED also
receives explicit EvidenceState, and supplies prospective success probability and confidence.
PEO additionally receives its own prediction history and numeric reconciliation from the existing
validated PEO episode. Full episode exports, labels, world IDs, world seeds and digests stay evaluator
only. Actor evidence references are replaced by public claim-local identifiers so arbitrary source
metadata cannot reveal latent values. No private reasoning or hidden model state is requested.

The factory does no unmetered inference. Every decision reports a positive integer compute receipt
in the host's named unit. The harness counts model calls and committed offline actions, rejects
over-budget receipts before simulation and makes no further model calls after compute/call
exhaustion. An abstention remains possible when only the action budget is exhausted; an attempted
action then ends as tool-budget exhaustion without simulation. The host enforces provider quotas
and authenticates all receipts; this Python API is not a sandbox. Representation bytes and local
harness duration are reported separately. Equal caps do not mean equal resource use.

Terminal target execution succeeds only when the simulator proceeds. Abstention succeeds only
when the current target judgment is abstain; abstaining before gathering required evidence fails.
Exhaustion is a failure. Preserve unsuccessful auxiliary attempts separately; terminal success does
not erase them. Include per-case rows, cohorts/severity, all failures, severe rows, per-arm success
rates, prediction Brier score and paired wins/losses/ties versus DIRECT. Report success and cost
deltas without a combined reward or automatic promotion. No statistical significance claim.

## Boundaries and evidence

Exactly 17 canonical cases; no default training, reward, authority, admission or runtime changes.
All cases remain non-TRAIN with the strictest input eligibility on the report. Python >=3.10,
no new dependency, 100-column Python lines. Public API docstrings describe inputs and errors.

Sources already registered: VGCompiler, C3-JEPA, MechBench, Generalized TAMP and CAT-Search.
Source mechanisms remain distinct from this local representation/budget experiment. Correct
official venue metadata for VGCompiler and C3-JEPA and add reciprocal PR-15 traceability.

Tests cover beneficial and no-benefit controls, visibility/relabel invariance, full PEO chronology,
fresh sessions and snapshot mutation, preflight rejection, exact numeric types, resource exhaustion,
permission denial, premature abstention, paired arithmetic and installed-wheel example. Require
>=80% new-code coverage, full suite, Ruff/Black, scoped mypy, wheel/sdist and documentation checks.

Risks: dishonest host metering, shared provider state, prompt contamination and checkpoint/training
claims cannot be verified here. Public text may itself encode labels; hosts audit case content and
held-out evaluation splits. Toy controls do not measure learned state quality or mechanism recovery.
