# World-model value ablation

`evaluation.world_model_ablation.compare_world_models()` runs an opt-in, offline experiment
on existing `SyntheticWorld` fixtures. It compares task outcomes and costs under shared model,
training and resource declarations. The bundled controls test the harness; no model effectiveness
has been measured.

## Treatments and task

| Arm | Input and required output |
| --- | --- |
| DIRECT | Public rendered observation, past action/success pairs and available actions → action or abstention. |
| STRUCTURED | DIRECT input plus explicit `EvidenceState` → success probability, confidence and action, or abstention. |
| PEO | STRUCTURED input plus validated past prediction/outcome reconciliation → prediction, confidence and action, or abstention. |

STRUCTURED corresponds to WORLD-STATE in the program proposal; PEO corresponds to PEO-STATEFUL.
Each arm starts from the same world. The task is to gather evidence, then execute a named target
action or abstain. Auxiliary actions must belong to the actor and have no effects; they can reveal
facts. The harness excludes other actions from `available_actions` and rejects attempts to use them.
This restriction keeps auxiliary actions from changing target conditions to manufacture success.

`Decision(None, compute_used)` means terminal abstention. A target attempt succeeds when the
existing simulator's judgment is `proceed`. Abstention succeeds when the current target judgment is
`abstain`. Declining a permitted target is `over_refusal`; declining while needed evidence remains
unknown is `premature_abstention`. Both fail. Unsuccessful auxiliary attempts remain in the failure
inventory even if the terminal decision succeeds. Simulator judgments establish fixture behavior
only; they do not authorize actions in an external system.

## API and host contract

Import the records from `evaluation.world_model_contracts`:

| Record | Fields |
| --- | --- |
| `WorldCase` | `case_id`, `world_json`, `actor_id`, `target_action`, `cohort`, `severity` (default `routine`). |
| `Budget` | Nonnegative, JSON-safe integer caps: `model_calls`, `tool_calls`, `compute_units`. |
| `ModelContract` | `model_version`, lowercase 64-character `checkpoint_sha256`, `adapter_version`, `training_examples`, `training_compute_units`, `compute_unit`. |
| `WorldBackend` | Shared `contract` and `factory(seed) -> policy`. |
| `DecisionInput` | `arm`, `payload_json`, `remaining` budget. |
| `Decision` | `action_id` or `None`, positive integer `compute_used`, optional `predicted_success` and `confidence` in [0,1]. |

The caller supplies a nonempty **tuple** of cases and calls
`compare_world_models(cases, backend, budget, seed=0, enabled=True)`. The harness validates and
snapshots all cases, contract fields and the factory callable before invoking the factory. Duplicate
case IDs or identical initial-world digests fail preflight. Model, checkpoint and training declarations
are shared across arms. `TRAIN` worlds are rejected; the report retains the strictest input label in
the order `DEVELOPMENT < REGRESSION < HIDDEN_EVAL`.

The harness invokes the factory separately for every case/arm. The factory receives a common
sampling seed within each triplet, independent of the world's generative seed. The schedule is
seeded and shuffled; the report records execution order. The factory must create an isolated session
and perform no unmetered inference. A policy call represents one model call. If an adapter makes
multiple provider calls internally, it does not satisfy this protocol.

The host must audit the adapter, provider logs, checkpoint identity and training records before
interpreting a real-model comparison. Use the same named compute unit for inference and training
declarations, or document an explicit common conversion in the adapter protocol. Meter all inference
input/output tokens, cache use, search and internal work under that protocol. The harness checks
integer receipts and caps; it cannot verify their truth or stop a callback that overspends or hangs.
Enforce provider-side quotas/timeouts in the host. Callback exceptions and malformed/over-budget
receipts abort the comparison without returning partial aggregates.

Before each decision, the harness supplies remaining allowance. No further policy call occurs after
compute or model-call exhaustion. Each simulated selected action consumes one tool call. An
abstention is allowed after the last tool call; a further requested action produces
`tool_budget_exhausted` without executing it. Model/compute exhaustion has precedence when those
caps are also reached. Factory creation is not a model call. Equal ceilings do not imply equal use.

## Actor data and costs

`payload_json` contains `observation`, `history`, `target_action`, `available_actions`, `state`
and `reconciliations`. DIRECT has `state=null`. Only PEO has reconciliation records. State is a
projection of public evidence, not a learned or privileged world state. Its reference identifiers and
source kinds are rebound to public observations, so private evidence metadata cannot add information.
World IDs, generative seeds, digests, evaluator labels and full exports are never placed in actor input.
Hosts must still audit public fact/action names and text for labels or contamination.

PEO commits each prediction before the corresponding simulation and validates the whole episode
with the existing `build_episode()` path. Its next input includes past predicted/actual success,
confidence, residual (actual minus predicted) and hidden-fact fractions before/after the action.
These diagnostics grant no reward or authority. Prefix replay has quadratic local work in episode
length; choose modest caps. Replay verification is pure simulation and does not consume additional
committed-action allowance.

Each raw row preserves individual `decisions`, model/tool/compute counts and public action history.
`representation_bytes` is the sum of UTF-8 payload bytes sent across policy calls; it is not a token
or memory metric. `harness_seconds` measures per-rollout wall time excluding factory/policy time;
it includes projection and PEO replay. `host_seconds` records factory/policy wall time separately.
Preflight and final aggregation time are outside these per-rollout timings. Timings are noisy;
repeat real experiments and keep hardware/provider conditions fixed.

## Read the report

- `arms`: all three arms with cases, successes, success rate, exhaustion count, unsuccessful
  actions, mean resource costs and success-prediction Brier score. DIRECT's prediction score is
  unavailable. Each score covers that arm's executed actions; action selection can differ.
- `paired`: STRUCTURED and PEO versus DIRECT, each with per-case signed success/cost deltas,
  wins/losses/ties and means. Zero and negative gains remain visible. There is no weighted fitness
  or gain-per-compute ratio; compare the explicit behavioral gain with each added cost.
- `strata`: the same summaries by cohort, severity and arm.
- `rows`, `failures`, `severe_rows`: complete rows, terminal failures or unsuccessful attempts,
  and all consequential/catastrophic rows (including successes).

Every planned case has a denominator entry for every arm, including budget failures. Raw PEO
episodes are evaluator-only and contain latent snapshots and labels. Never feed report rows or full
episodes back into the actor or optimizer. The report makes no significance, mechanism-recovery,
real-world safety or automatic promotion claim. Hosts select held-out worlds before evaluating,
retain failed runs, and audit training/evaluation separation externally.

## Runnable deterministic control

This policy reads the same public facts in every arm. All arms should succeed, with zero behavioral
gain from structured state. The fake digest and unit label explicitly identify a wiring fixture.

```python
import json

from evaluation.world_model_ablation import compare_world_models
from evaluation.world_model_contracts import Budget, Decision, ModelContract, WorldBackend, WorldCase
from synthetic_data.worlds import generate_world


def factory(seed):
    def decide(request):
        observation = json.loads(request.payload_json)["observation"]
        if "safe=unknown" in observation:
            action = "inspect"
        else:
            action = "release" if "safe=True" in observation else None
        return Decision(action, compute_used=1, predicted_success=1.0, confidence=0.8)
    return decide


cases = tuple(
    WorldCase(str(seed), json.dumps(generate_world(seed=seed, enabled=True).to_dict()),
              "operator", "release", "toy-control")
    for seed in (0, 1)
)
backend = WorldBackend(ModelContract("fixture", "a" * 64, "fixture-v1", 0, 0, "fixture-step"),
                       factory)
report = compare_world_models(cases, backend, Budget(3, 2, 10), enabled=True)
assert all(arm["success_rate"] == 1 for arm in report["arms"].values())
assert all(pair["success_rate_delta"] == 0 for pair in report["paired"])
```

## Research and maturity

| Source mechanism | Local inference / experiment |
| --- | --- |
| [VGCompiler](https://arxiv.org/abs/2609.22327): visual graph compilation followed by executable reasoning. | Test whether explicit representation changes action outcomes; no visual compiler is implemented. |
| [C3-JEPA](https://arxiv.org/abs/2609.30214): object-centric prediction and downstream control. | Separate state/prediction quality from action value; no learned latent world model is implemented. |
| [MechBench](https://arxiv.org/abs/2609.35515): distinct phenomenal-law and mechanism probes. | Toy behavioral gains do not establish mechanism recovery. |
| [Generalized TAMP](https://arxiv.org/abs/2609.30233): fixed synthesis budget, frozen programs and unseen instances. | Freeze one host model/training contract and compare shared worlds; no robotics benchmark reproduction. |
| [Direct Optimization of Generators for Search](https://arxiv.org/abs/2609.25575): search-aware generator objectives. | Account for actual resource use under common caps; no theorem-proving training objective is implemented. |

The supplied name “Compute-Aligned Training for Search” remains an explicitly inferred source
mapping in the [research registry](recommendations/RESEARCH_TRACEABILITY.md#ref-cat-search).
Hypothesis: explicit state and reconciliation improve decisions enough to justify added complexity.
Evidence here: deterministic controls verify sensitivity, ties and regressions, not that hypothesis.
The [ADR](adr/0016-world-model-value-ablation.md) explains the bounded design.

Verification: `python -m pytest tests/test_world_model_ablation.py` checks the executable contracts.
The installed-wheel smoke executes this example. For a model study, archive the adapter source,
immutable checkpoint/training records, provider receipts, world split and full report for host review.
