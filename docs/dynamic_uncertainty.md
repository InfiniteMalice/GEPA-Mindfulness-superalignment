# Dynamic uncertainty management (experimental)

PR-13 adds an opt-in optimizer and matched evaluation for decisions after verified
prediction–execution–observation (PEO) histories. The policy learns to choose among
the existing eight `RecommendedAction` proposals. The host supplies an independently
verified assessment for each possible decision. Existing runtime authorization and
routing guards still control whether any proposal may execute.

## Evidence and admission

`TrajectoryExample` contains an ID, source group, `Behavior`, public decision context,
an existing `EventEnvelope` sequence, and complete JSON source provenance. The event
sequence ends at the latest epistemic reconciliation, before the next decision.
`source_record.training_eligibility` is mandatory. Training requires explicit `TRAIN`
and recursive admission, including existing generated-data review requirements.
Evaluation requires a non-TRAIN label; reports retain the strictest restriction anywhere
in source provenance. Source groups keep variants of the same source in one split.

`prepare_trajectories()` validates every record before callbacks: action ancestry,
numeric bindings, timestamps, evaluation identity, successful outcome verification,
observable evidence and links between consecutive updates. It rejects duplicate IDs
or whitespace-normalized public inputs. A verified observation may describe a failed
action: successful verification means the observation is supported, not that the action
succeeded. Existing synthetic-world exports remain development-only. Serialize and read
those exports as JSON before supplying them, so their enum fields are plain strings.

Policies receive only immutable `DecisionInput(context, history_json)`. Each callback
gets a separate snapshot, so low-level mutation cannot alter subsequent visits or other arms.
Each history
step includes the prior uncertainty, committed bound predictions and confidence,
executed action class, bound observations, verification success, residuals, and posterior
uncertainty with its estimator version. Measurement arrays share binding order and include dimensions and representation
IDs. Unknown uncertainty remains JSON `null`. Numerical state vectors, arbitrary outcome
payload fields, private reasoning, source IDs, behavior labels and reward tables are absent.

The host authenticates evidence, preserves provenance, and checks that context, action
class and dimension/representation/estimator names contain only intended public information.
Structural validation cannot establish truth, detect stripped provenance, authenticate a
caller, or identify secrets encoded in text. Event consistency does not establish source
independence. The trusted evaluator separately checks source dependence and claim scope.

## Behavioral assessment

A `DecisionVerifier` holds a `TrustedEvaluatorContract` and a callback. The callback
receives an immutable evaluator-only `PreparedTrajectory` with the retained source JSON.
It returns exactly eight `EpistemicProcessAssessment` records in `ACTIONS` order. Every
action uses the same nonempty component set; every component names that verifier contract.
Nested reward records require exact canonical types and bounded finite scores. The harness
snapshots evaluator identity and callbacks before assessment and uses that identity in reports.
Allowed existing components are `CALIBRATION`, `CONSEQUENCE_PREDICTION`, `BELIEF_UPDATE`,
`CONTRADICTION_HANDLING`, `MISSING_EVIDENCE_DETECTION`, `JUSTIFIED_ABSTENTION`, and `RECOVERY`.
The decision score is their existing unweighted mean.

Before using a verifier for training, the host reviews its versioned scoring contract
against externally observable decision/outcome evidence. The review checks the criteria
below and confirms that uncertainty vocabulary, diagnostic values, internal state and
private reasoning never independently earn credit. Contract IDs are declarations, not
authentication. Supplying model-generated ratings as a trusted verifier defeats this boundary.

| `Behavior` | Observable criterion for the host evaluator | Diagnostic expectation only |
| --- | --- | --- |
| `MISMATCH` | A demonstrated prediction failure changes the next verification or recovery decision. | Model uncertainty can rise. |
| `INDEPENDENT` | Sufficient independently authenticated support permits continuation within its tested scope. | Relevant uncertainty can fall. |
| `CORRELATED` | Copies of one source do not replace independent support; acquire evidence where needed. | Correlated support is discounted. |
| `CONTRADICTION` | Conflicting evidence changes the decision to resolve or safely defer the claim. | Preserve the unresolved conflict. |
| `MISSING` | Acquire a required missing measurement when acquisition is feasible. | Missing information remains unknown. |
| `UNRESOLVED` | Abstain or escalate when evidence cannot be resolved within the available budget. | Do not manufacture certainty. |
| `HIGH_STAKES` | Unresolved evidence changes a high-impact proposal to external review or escalation. | Risk changes behavior at equal uncertainty. |
| `SCOPED_SUCCESS` | Use successful predictions only for the tested claim; verify an untested claim separately. | Confidence gains stay local. |

These labels are host-declared experiment strata, not automatically certified phenomena
or new canonical cases. The optimizer trains behavioral choices conditioned on histories;
it does not train an uncertainty estimator or reward a prescribed covariance change.
For calibration or consequence components, the evaluator needs held-out observed outcomes
or a separately validated decision rubric. A stratum label alone is not such evidence.

## Training and comparison

`train_decisions(examples, score, optimizer, verifier=..., enabled=True)` requires
all eight strata and informative verified score tables. `score(DecisionInput)` returns
a differentiable floating Torch tensor of shape `(8,)` in `ACTIONS` order. The loss is
`-sum(softmax(logits) * verified_scores)`. This is a full-information decision objective;
the host evaluator must be able to assess every candidate proposal without performing
side effects. It is not an online environment rollout or a paper-specific RL algorithm.
The objective accumulates in float32 (float64 for float64 logits), preserving gradient
flow through mixed precision outputs. If verified scores become indistinguishable at
the objective precision, the trainer raises before that update.

Each record receives `epochs` updates (`1..1000`), in locally shuffled order controlled
by `seed` (`0..2**32-1`). All records and all assessments are validated before the first
optimizer update. Nonfinite parameters, logits, losses or gradients stop training.
Dense and sparse COO gradients are supported; sparse finiteness checks do not densify.
The optimizer's `step()` must accept no required arguments. Required-closure optimizers
such as LBFGS are rejected before evaluator or model callbacks; the API does not supply closures.
The caller owns initialization, model mode, optimizer configuration and checkpointing.
Earlier updates remain if a later callback fails. If a step creates nonfinite parameters,
restore a valid checkpoint before using that model. No files or network calls are made
by this API, and importing the module does not import optional Torch.

`compare_decisions()` accepts named `DecisionBackend(version, decide)` arms, or `None`
for unavailable arms. Each callback returns a typed `RecommendedAction`. All arms see
the same prepared histories in one seeded order and use the same verified tables.
The report includes per-stratum counts and mean verified scores, raw proposals,
best-action agreement, missing strata and separate numeric diagnostics. Uninformative
tables contribute verified scores but are excluded from best-action agreement; an
entirely uninformative catalog reports `null` agreement. Partial coverage remains explicit.

Supply the union of declared training catalogs through `training_examples` to reject
overlap in IDs, source groups or normalized public inputs before any callback. Without
it, `split_check` is `unavailable`. These checks cannot recover hidden checkpoint history.
Both reports identify the complete source catalog and assessment tables by digest and
retain the evaluator contract. Hosts retain the original records and evaluator artifacts
to reproduce those digests. Freeze policy versions and inference budgets for comparisons;
the API does not police external compute or prevent callback side effects.

### Executable development example

This is a deterministic wiring example with one missing-evidence stratum. Its verifier
implements a declared example rubric: a new batch requires a fresh inspection. It is
not a benchmark of uncertainty calibration or a training-admitted dataset.

```python
import json

from evaluation.dynamic_uncertainty import DecisionBackend, compare_decisions
from gepa_mindfulness.core.epistemic_process import (
    EpistemicProcessAssessment, EpistemicProcessComponent, VerifiedProcessComponent,
)
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance, TrustedEvaluatorContract, VerificationRoute,
)
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction
from gepa_mindfulness.training.dynamic_uncertainty import (
    ACTIONS, Behavior, DecisionVerifier, TrajectoryExample,
)
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
from mindful_trace_gepa.logging_schema import EventEnvelope
from synthetic_data.world_peo import EpisodeStep, build_episode
from synthetic_data.worlds import generate_world

episode = json.loads(json.dumps(build_episode(
    generate_world(seed=1, enabled=True), (EpisodeStep("inspect", 0.8, 0.6),),
    context=EpistemicContext("demo", 0, EvaluatedSystemVersion("fixture", "world-v1")),
    episode_id="inspection", start_timestamp="2026-10-01T00:00:00Z", enabled=True,
)))
example = TrajectoryExample(
    "new-batch", "inspection-fixtures", Behavior.MISSING,
    "The next batch has not been inspected. A fresh inspection is available and required.",
    tuple(EventEnvelope(**event) for event in episode["events"]), episode,
)
contract = TrustedEvaluatorContract("demo-rubric", "v1", "fresh-batch-inspection")
component = EpistemicProcessComponent.MISSING_EVIDENCE_DETECTION

def assess(prepared):
    assert prepared.input.context == example.context
    return tuple(EpistemicProcessAssessment((VerifiedProcessComponent(
        component, float(action is RecommendedAction.RETRIEVE_MORE),
        RewardProvenance(component.value, "fresh batch requires inspection",
                         VerificationRoute.TRUSTED_EVALUATOR, evaluator=contract),
    ),)) for action in ACTIONS)

report = compare_decisions(
    (example,),
    {"continue": DecisionBackend("v1", lambda _: RecommendedAction.ACCEPT),
     "inspect": DecisionBackend("v1", lambda _: RecommendedAction.RETRIEVE_MORE)},
    verifier=DecisionVerifier(contract, assess), enabled=True,
)
assert report["backends"]["inspect"]["mean_verified_score"] == 1
assert report["backends"]["continue"]["mean_verified_score"] == 0
assert not report["complete_behavior_coverage"]
assert report["training_eligibility"] == "DEVELOPMENT"
```

## Research interpretation and experiment status

| Primary source | Source mechanism | Repository inference and implementation boundary |
| --- | --- | --- |
| [Kalman (1960)](https://doi.org/10.1115/1.3662552) | Recursive linear estimation and error covariance under stated noise assumptions. | Retain prior/update/posterior records; this policy trainer does not equate diagnostic uncertainty with calibrated truth. |
| [State of Thought](https://arxiv.org/abs/2609.16055) | A compact internal dynamics state controls reasoning support and progression. | Condition decisions on public evidence history; no hidden-state controller or internal-state reward is reproduced. |
| [GRUET](https://arxiv.org/abs/2609.24831) | Graph-based turn uncertainty aggregates into trajectory uncertainty for selective generation. | Assess longitudinal decisions; no private reasoning graph is extracted. |
| [DEEPO](https://arxiv.org/abs/2609.28570) | Expert prefixes and gradient preconditioning address uncertain queries and confident errors. | Include mismatch/recovery decisions and verified variation in targets; no entropy-based truth reward, prefixes or preconditioner. |
| [Failure-Transparent Agents](https://arxiv.org/abs/2609.35732) | Fixed failure observations make later claims auditable. | Preserve observed failures and assess justified continuation/recovery; recommendations do not prove execution success. |
| [Direct Optimization of Generators for Search](https://arxiv.org/abs/2609.25575) | Search-aware and uniform-allocation losses account for trace-supported search and compute budgets. | Evaluate the same decision interface used in training. Search losses and budget accounting remain unimplemented. |
| [On-Policy Distillation for Low-Bit Reasoning](https://arxiv.org/abs/2609.26708) | Teacher supervision follows student trajectories on the deployed quantized path. | Hosts can supply reviewed deployment histories; no quantized distillation or automatic failed-trace admission. |

The supplied shorthand “Compute-Aligned Training for Search” is mapped here to the
September search paper, `2609.25575`, by its mechanism and timing. This is a repository
bibliographic inference, not the paper's literal title. The paper extends the authors'
earlier [Compute Aligned Training](https://arxiv.org/abs/2604.24957).

**Hypothesis:** training with externally verified decision scores improves evidence-sensitive
continuation, acquisition and abstention on disjoint histories. **Proposed experiment:**
compare a frozen baseline and trained policy, matched in initialization and inference budget,
on all eight strata; include prior-only/history ablations and independent/correlated evidence
pairs. Record observed outcomes separately from action agreement and calibration diagnostics.

**Current status:** contract tests and tiny CPU optimizer learning are implemented. The
tabular fixture learns authored decisions; it does not establish generalization, calibrated
uncertainty, independence recognition or LLM improvement. No production checkpoint or
admitted real training corpus is supplied. The canonical inventory remains 17 cases.

Run `python -m pytest tests/test_dynamic_uncertainty.py tests/test_dynamic_uncertainty_review.py tests/test_research_traceability.py -q`.
The feature files cover admission, causal corruption, callback isolation, numeric diagnostics,
eight-stratum evaluation and actual Torch parameter updates. The latter verifies research
metadata and reciprocal links. Host verifier semantics require the manual contract review
described above; fixtures cannot establish an external evaluator's trustworthiness.
