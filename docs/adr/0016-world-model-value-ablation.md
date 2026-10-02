# ADR 0016: Match world-model treatments and expose their costs

Status: Accepted for an experimental, opt-in offline API.

## Context

The repository can simulate worlds, project evidence state, validate PEO episodes and report
evaluation stages. These capabilities do not establish that state improves behavior. A comparison
needs shared worlds, model/training conditions and budgets, plus costs that remain visible when
task success ties or declines.

## Decision

Add one matched rollout API over the existing simulator. Use one shared host contract and factory,
fresh policy sessions, common caps and seeded triplets for DIRECT, STRUCTURED and PEO. Restrict
auxiliary actions to evidence acquisition without effects, then score a target decision against the
existing visible-evidence oracle. This avoids inventing a general planner objective in this PR.

Reuse EvidenceState and PEO replay validation. Expose only public projections to policies; normalize
source metadata so representation does not add privileged information. Report actual model/tool/
compute usage, representation bytes and local duration beside paired behavioral deltas. Preserve
failed and severe rows. Add no weighted reward, canonical case or automatic runtime promotion.

## Consequences

The API measures a bounded information-gathering task and can run a real host model adapter.
Deterministic positive/negative/tie controls validate report sensitivity but cannot establish model
value. Matching is partly declarative: hosts authenticate provider identity, training records, quotas,
metering and session isolation. The API is not a sandbox. PEO prefix replay adds bounded quadratic
local work, reported separately from host-metered inference. No new dependency is required.

The [guide](../world_model_ablation.md) defines adapter obligations, outcome semantics and costs;
`tests/test_world_model_ablation.py` covers matching, visibility, causal continuity and budget failures.
