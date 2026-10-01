# Experimental PEO curriculum

The opt-in `PEOCurriculumDataset` selects data for the existing
`RLTrainingEngine.dataset_factory` interface. The seven data stages complement the
five participatory-agency head/loss phases. Selecting a head phase records its identity;
this provider does not apply its loss weights to the engine.

## Stages and annotations

| Stage | Curated content |
| --- | --- |
| `CAUSAL` | Causal and epistemic primitives |
| `INVARIANCE` | Equivalent representations with stable judgments |
| `COUNTERFACTUAL` | Minimal decisive relation changes |
| `EVIDENCE` | Hidden information, uncertainty, inquiry, abstention |
| `NORMS` | Conditional normative conflicts |
| `ADVERSARIAL` | Authority pressure, semantic laundering, evaluator pressure |
| `TEMPORAL` | Longitudinal prediction–execution–observation (PEO) |

The temporal catalog names supersession, stale evidence, motivated forgetting, state
drift, correction after error, reward pressure, evidence change, and source reliability
change. Unit labels describe host-curated content; the provider does not inspect prompts
to certify those properties or generate training examples.

Optional dimension annotations are finite numbers in [0, 1]:
U uncertainty; A adversarial pressure; H hidden information; P principle conflict;
D deceptive/reward-hacking incentives; T temporal depth; M semantic mutation;
C causal complexity; R reversibility/consequence; I information acquisition;
S routing complexity; J cross-variable dependence; E decisive-event sparsity.
They describe data and never enter a reward formula.

## Sampling and admission

Every complete materialized round includes positive integer quotas of **anchor,
weakness, frontier, and OOD** units. Defaults are 4/2/1/1. Anchors belong to the
foundational stage and persist across every stage. Their quota must be at least
25% of selected units. This floor does not cover tokens, trajectories, gradients,
partial rounds, or individual optimizer batches.

A unit groups one or more existing `RolloutRequest` objects. Requests in each
selected unit remain contiguous. The provider preserves prompts, case IDs, sampling
parameters and source metadata, and adds a reserved `peo_curriculum` metadata
record. Input metadata is deep-snapshotted; returned mappings are detached copies.
Use plain JSON values for source records.

Selection uses sorted unit IDs, seeded pool permutations and a round offset. Small
pools wrap and repeat. Missing pools fail closed. GRPO still requires unique nonempty
case IDs in each engine generation call: size pools to avoid repeated requests for
GRPO collection/evaluation, and use distinct case IDs across units. The provider
does not rename cases or change reward-provider lookup keys.

For `train` and `resume`, **every request in the full catalog** must declare
top-level `training_eligibility: TRAIN` and pass the existing recursive eligibility
gate before any selection. This includes inactive and untargeted units because they
contribute to the plan digest. Nested holdouts, failed V5 outcomes and unreviewed
generated data remain excluded. `collect` and `evaluate` retain source restrictions.
PR-9 worlds remain non-trainable; keep them in a separate evaluation plan. OOD means
a reviewed novel distribution, never permission to admit hidden evaluation data.

## Minimal collection plan

This example demonstrates scheduling with development fixtures, not training admission.

```python
from gepa_mindfulness.participatory_agency.training.curriculum import (
    DEFAULT_CURRICULUM, PEOStage,
)
from gepa_mindfulness.training.peo_curriculum import (
    CurriculumBucket, CurriculumMixture, CurriculumUnit, PEOCurriculumDataset,
)
from gepa_mindfulness.training.trajectory import RolloutRequest

units = tuple(
    CurriculumUnit(
        unit_id=bucket.value,
        stage=PEOStage.CAUSAL,
        bucket=bucket,
        requests=(RolloutRequest(
            prompt="Identify what evidence is still missing.",
            case_id=f"development:{bucket.value}",
            metadata={"training_eligibility": "DEVELOPMENT"},
        ),),
        dimensions={"U": 0.5, "I": 0.5},
    )
    for bucket in CurriculumBucket
)
plan = PEOCurriculumDataset(
    units=units,
    phase=DEFAULT_CURRICULUM[0],
    mixture=CurriculumMixture(1, 1, 1, 1),
    seed=42,
    enabled=True,
)
requests = plan.materialize("collect")
assert len(requests) == 4
assert {r.metadata["peo_curriculum"]["bucket"] for r in requests} == {
    "anchor", "weakness", "frontier", "ood",
}
assert all(r.metadata["training_eligibility"] == "DEVELOPMENT" for r in requests)
```

Pass `dataset_factory=lambda config: plan` when constructing the existing
`RLTrainingEngine`. Actual training also needs an approved reward provider compatible
with these requests; the default authored-pair provider does not score arbitrary
curriculum prompts. Save `plan.to_dict()` and `plan.digest` with the host run record:
the engine's dataset-file checksum and trajectory schema do not automatically store
the complete injected provider manifest. Request metadata is available to the backend,
but is not automatically a trajectory field or actor prompt. Hosts must keep evaluator
metadata and hidden source records out of actor-visible text.

## Retention-gated adaptation

After evaluating **every anchor**, the host constructs `AnchorEvaluation` with the
current plan digest, the actual `EvaluatedSystemVersion`, and one `AnchorResult`
per anchor. Each result needs a boolean pass and an existing external-record
`EvidenceReference`. The host must verify that those records belong to the declared
model/harness and represent the stated outcomes; references do not authenticate
themselves.

`plan.adapt(evaluation, stage=..., mixture=..., weakness_ids=...)` returns a new
immutable round. It rejects stale plan identities, incomplete or duplicate coverage,
unknown IDs, failed anchors, private-reasoning evidence, and stage skips. It can stay
at the current stage or advance one stage; all anchors and source units remain.
Weakness targets must name current-stage weakness units. Targets reset on advancement
unless explicitly supplied. Quotas must retain all four buckets and the anchor floor.
The returned request reports retain the prior assessment for audit.

Hosts can explicitly construct a starting stage. These Python records are scheduling
contracts for trusted hosts, not an authorization boundary or a model evaluator.
No automatic loop changes policy, promotes data, changes rewards, grants authority,
or makes a learned-retention claim.

## Research scope

- [TabPFN-3.5](https://arxiv.org/abs/2609.17895), §3.3: diverse synthetic distributions
  motivate multiple data dimensions; no TabPFN prior or benchmark is reproduced.
- [CHART](https://arxiv.org/abs/2609.22247), §3.2: rotation of active harnesses motivates
  controlled distribution changes. Persistent anchors are a repository design choice;
  this provider does not implement CHART's training algorithm.
- [Qwen Planner](https://arxiv.org/abs/2609.29892), §§2.3.4 and 2.4.3: frozen reviewed
  data configuration and competence-aware reward/advantage engineering (CARE) inform
  bounded host-directed adaptation. CARE reward and advantage formulas are not added.
- [Spectral grokking theory](https://arxiv.org/abs/2609.26679): delayed generalization
  motivates repeated retention checks after apparent mastery. No NTK analysis or
  grokking claim follows from these scheduling tests.
- [Low-bit on-policy distillation](https://arxiv.org/abs/2609.26708): student-induced
  distributions motivate reviewed weakness/recovery data. No quantization, teacher
  loss, or raw failed-trace admission is implemented.

Metadata and exact source/repository distinctions are mirrored in
[research traceability](recommendations/RESEARCH_TRACEABILITY.md).
The 17 canonical cases, default curriculum, reward formulas and runtime gates are unchanged.

## Validation and limits

Before this change there were five head phases without a shared seven-stage data
provider or anchor-retention adaptation contract. Tests now verify all seven stages,
all four pools, deterministic rotation, grouped requests, immutable metadata, full
catalog admission, PR-9 exclusions, stale/regressed anchor rejection and real CPU
engine collection. Run:

```text
python -m pytest tests/test_peo_curriculum.py tests/participatory_agency/test_curriculum.py tests/test_rl_engine_cpu.py
```

These are contract and integration checks. They do not measure learning quality,
semantic coverage, fairness of curator labels, or behavioral retention in a trained model.
