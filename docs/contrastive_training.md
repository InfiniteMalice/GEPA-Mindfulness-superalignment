# Contrastive causal hard negatives (experimental)

PR-12 adds an opt-in optimizer loop and matched comparison over existing
`PairwiseReasoningExample` records. No default trainer, runtime router, reward,
authority gate or canonical case changes. The canonical inventory remains 17 cases.

## Admission and model inputs

Each CPT pair requires `metadata.negative_family`, `metadata.source_group` and
`metadata.training_eligibility`. Both candidates must have the pair's problem ID,
identical prompts, distinct final answers and a strict A/B preference. IDs and
whitespace-normalized prompt/answer pairs must be unique within a catalog.

`prepare_pairs` snapshots the complete JSON-compatible record before returning an
immutable public projection. Training requires explicit `TRAIN` and the existing
recursive admission checks, including nested split restrictions and generated-data
review. Evaluation requires an explicit non-TRAIN label. Relabeling an outer field
cannot override retained source restrictions. Hosts must preserve provenance;
metadata cannot detect deliberately stripped source records.

Scorers receive only `(prompt, (answer_0, answer_1))`. They never receive pair IDs,
family, source group, labels, correctness, teacher rationale, reasoning summaries,
structured reasoning units, confidence or provenance. Hosts must also ensure that
authored prompt/answer text contains only intended public content; this API does not
classify arbitrary text as public or private. The objective uses public answer
preference, not private reasoning, uncertainty or residual magnitudes.

## Training

`gepa_mindfulness.training.contrastive.train_contrastive` requires `enabled=True`,
a caller-supplied differentiable scorer and a `torch.optim.Optimizer`. Torch is the
existing optional training dependency; importing this module does not import torch.
The scorer returns a floating tensor of shape `(2,)`, in supplied answer order.
The loss is `softplus(rejected_logit - chosen_logit)`, the two-candidate softmax
cross-entropy objective. Invalid logits, losses, optimizer gradients or parameters stop training.
The caller owns model mode, initialization, optimizer settings and checkpoints.
Caller-owned model/optimizer mutations remain if training fails. If an optimizer
step produces nonfinite parameters, the caller must restore a valid checkpoint
before using that model; the trainer raises without returning a success report.

All five families must be present:

| Order | Metadata value | Intended distinction |
| --- | --- | --- |
| 1 | `broad_semantics` | Broad public state/answer compatibility |
| 2 | `semantic_hard` | Similar content with a materially wrong answer |
| 3 | `causal_hard` | Wrong decision under a decisive world relation |
| 4 | `laundering_hard` | Untrusted claims overriding the world or policy |
| 5 | `peo_trajectory` | A public action/outcome summary that falsifies an observation |

`schedule="curriculum"` completes `epochs` passes over each family in this order.
`schedule="pooled"` shuffles all records together for the same number of passes.
Each record receives exactly `epochs` updates in either schedule. The seed controls
record ordering and deterministic answer-order randomization; matching seeds do not
make external model kernels deterministic. `epochs` must be an integer in `[1, 1000]`
and `seed` an integer in `[0, 2**32)`. Booleans are rejected for both.

The returned report contains a digest binding complete source records, source groups, per-family update
counts, actual family order and per-update losses. Use `compare_backends` on held-out
records for family margins. Compare schedules from the same initial model, optimizer,
data and update budget. Source groups must keep all variants of the same source
together. Family labels are host declarations; the API does not certify difficulty.

The CPU wiring test runs a tiny authored scorer from zero weight, checks that its
margin improves and its loss falls, and verifies matched update counts. It is not
evidence that the curriculum outperforms pooled training:

```console
python -m pytest tests/test_contrastive_training.py -q
```

## Deterministic negative construction

`synthetic_data.contrastive_negatives.relation_negatives` accepts an existing
decisive `RelationPair` and emits four CPT pairs: before/after contexts, each with a
causal and laundering variant. The rejected answer is the correct decision in the
other world. Nuisance controls cannot produce a decisive negative. The laundering
variant adds an explicitly untrusted retrieved note claiming the wrong decision is
official. These authored attack templates are bounded fixtures, not a diverse corpus.

`peo_negative` calls the existing `build_episode` simulator with at least two
caller-supplied steps. Its public prompt includes step index, action, prediction and
observed outcome. The rejected trajectory flips only the last observed success bit.
The complete validated episode stays in provenance, including hidden world snapshots;
those snapshots, oracle judgments and residual records never enter the prompt.
This tests observation fidelity in a trajectory, not every possible PEO failure.

World-derived examples retain their source's non-TRAIN eligibility. The current
world contracts forbid TRAIN; this PR does not change that restriction. Broad and
semantic training records, and admitted training records for later families, must
come from a separately reviewed corpus. There is no automatic corpus admission,
generic LLM negative generation, model download or external checkpoint integration.

## Four-arm comparison

`evaluation.contrastive.compare_backends` takes all four keys: `classifier`, `jev`,
`clm`, `clm_curriculum`. Each value is a `RankingBackend(name, version, score)` or
`None`. The callback returns a list or tuple of two finite numbers; larger means
more preferred. Hosts adapt their classifier or JEV/CLM implementation to this
contract. The repository's existing tier classifier is not silently substituted
for a state/action ranker. No external backend or trained checkpoint ships here.

The evaluator presents every pair in both answer orders to every available arm.
It shuffles the full presentation catalog across pairs using `seed` (default `0`),
then reuses that identical schedule for every arm. This removes the fixed
chosen-first/chosen-second call pattern. The report records the seed and actual
presentation order for audit; neither is passed to the scorer. The seed must be an
integer in `[0, 2**32)` and cannot be a boolean.
Accuracy requires the chosen answer to win strictly in both presentations. It
reports mean chosen-minus-rejected margin, tie rate, order disagreement and accuracy
per family, plus raw scores bound to pair IDs and backend version. Score scales are
backend-specific: compare accuracy across arms; interpret margins within an arm.
Missing families and unavailable arms are explicit, without fabricated metrics.

Pass the deduplicated union of all arms' training catalogs as `training_examples`.
The evaluator rejects overlap in pair ID, problem ID, source group or normalized
public content, including reused prompts with different negatives, before calling
any backend. An omitted/empty catalog is labeled
`training_catalog_not_supplied`; the report cannot certify held-out evaluation in
that case. The check covers declared records, not undisclosed pretraining data.
The caller must freeze backend weights/configuration for the entire comparison.

This runnable example builds development negatives and records unconfigured arms:

```python
from evaluation.contrastive import COMPARISON_ARMS, compare_backends
from synthetic_data.contrastive_negatives import relation_negatives
from synthetic_data.relation_flips import Relation, make_relation_pair

world_pair = make_relation_pair(Relation.CONSENT, enabled=True)
examples = relation_negatives(world_pair, enabled=True)
report = compare_backends(examples, dict.fromkeys(COMPARISON_ARMS), enabled=True)
assert report["arms"]["clm"]["status"] == "unavailable"
assert report["missing_families"] == ["broad_semantics", "semantic_hard", "peo_trajectory"]
```

## Research basis and experiment status

| Source and demonstrated mechanism | Repository inference | Status / discriminating experiment |
| --- | --- | --- |
| [CLM official trainer](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py): separate state/action heads, group-masked bidirectional in-batch contrastive loss | Explicit public state/answer ranking with a two-candidate objective | API and optimizer tested; no CLM encoder/head or source loss replication. Compare pooled and staged training from the same checkpoint. |
| [P-TTT Appendix D](https://arxiv.org/html/2609.35109v1): unambiguous contextual preference reversals | Use the opposite-world decision as the negative | Deterministic fixture contract tested; no fast-weight adaptation. Measure sensitivity on held-out relation reversals. |
| [VGCompiler](https://arxiv.org/html/2609.22327v1): graph representations and executable operations separate state from rendering | Derive labels from world structure rather than generic wrong-answer generation | Existing world simulator reused; test held-out rendering variants before claiming transfer. |
| [MechBench](https://arxiv.org/html/2609.35515v1): mechanism mutations and screening observationally indistinguishable alternatives | Reject uninformative causal pairs; separate observable ranking from mechanism recovery | Behavior only; no internal-mechanism claim. Compare decisive and nuisance interventions. |
| [CHART](https://arxiv.org/html/2609.22247v1): active harness windows graduate learned tasks and introduce learnable tasks | Expose increasingly difficult negative families | Fixed schedule implemented; no adaptive harness/GRPO reproduction. Compare equal-budget shuffled exposure. |
| [Physical Representation Languages](https://arxiv.org/html/2609.23381v1): controlled experiments and limits from observational equivalence | Preserve typed world provenance and avoid claims beyond identifiable public outcomes | No physical-system identification. Test transfer across world families before claiming generalization. |

Hypothesis: staged causal negatives improve held-out policy-sensitive ranking without
losing broad semantics. The four-arm experiment can reject that hypothesis. Real
classifier/JEV/CLM measurements, retained broad-semantic performance, multiple seeds,
and representative held-out corpora remain unmeasured. No winner is selected here.

Validation: `tests/test_contrastive_training.py`, `tests/test_contrastive_negatives.py`
and `tests/test_contrastive_evaluation.py` cover admission, visibility, gradients,
exposure, counterfactual labels, candidate-order bias, overlap and missing arms.
