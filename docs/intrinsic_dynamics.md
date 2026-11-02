# Intrinsic-dynamics diagnostic protocol

Maturity: **Experimental** CPU probes and **Diagnostic** reports. Disabled by default.
Native joint encoder/predictor/IDM training, visual backbones, planning and empirical effects
remain **Unsupported**. No encoder, policy, optimizer or accepted baseline is changed.

This is an original discrete/linear adaptation of [Shi et al.](https://arxiv.org/abs/2610.21612).
The source studies action accessibility in consecutive frozen latents using MLP probes and
separately compares predicted and realized candidate-cost rankings. Our small linear probes
do not reproduce that architecture or its findings. High decoding scores establish neither
causally faithful dynamics nor useful planning, constitutional alignment or moral admissibility.

## Inputs and offline synthetic-world binding

Import `DynamicsProbeConfig`, `ProbeTransition` and `RolloutCost` from
`evaluation.intrinsic_dynamics_records`; import the three functions from
`evaluation.intrinsic_dynamics`. All inputs use existing strict, versioned DEVELOPMENT
diagnostic serialization and observable evidence references. Config defaults to disabled.
The numerical function imports NumPy only when enabled. The existing optional numerical extra
provides it: `pip install -e '.[fusion,dev]'`; this packaging reuse is not a TF-OPD dependency.
There are no model downloads, provider calls, activation hooks or external execution.

`capture_world_transition(initial_world, action_ids, index, trajectory_id=..., before=...,
after=..., predicted_after=None, enabled=True)` replays a prefix with the existing offline
`simulate` contract. It binds root-world/prefix/state digests and the actual attempted action,
using a stable sorted action vocabulary. It never mutates the input world. A denied simulated
action remains an attempted action, without a fabricated success label.

Only initial tick-zero DEVELOPMENT worlds are accepted. REGRESSION/HIDDEN_EVAL worlds must
never be relabeled or stripped of provenance for this bridge. Internal probe `fit`/`test`
partitions remain evaluator diagnostics; they are not TRAIN policy data. This wrapper neither
encodes the states nor authenticates that supplied latents came from them. Preserve original
worlds, PEO exports, latent collectors and evidence so an independent evaluator can recover
sources. Full evaluator exports and labels never belong in actor prompts.

Host-supplied `ProbeTransition`s carry trajectory/step, root and transition digests, action
identity/vocabulary, before/after latents, references and an optional predicted next latent.
These are declarations, not verification receipts. Direct constructors cannot authenticate
raw collection or detect stripped provenance. Host controls own those duties. Synthetic replay
consistency is limited to the fixture and grants no real-world authority.

Prototype bounds are 512 records, 16 latent coordinates/actions, 16 observable references per
record and 512 UTF-8 bytes per text coordinate. Latents are finite with magnitude at most 1e6;
costs are nonnegative and at most 1e12. Complete reports are at most 4 MiB. These are original
CPU controls, not upstream hyperparameters.

## Matched action-accessibility experiment

Call `run_action_probes(rows, system=SystemConfig(...), fit_trajectories=(...),
test_trajectories=(...), config=DynamicsProbeConfig(enabled=True, seed=0))`.
Pin encoder/checkpoint, collector, forward predictor, action semantics and preprocessing in the
system versions/configuration digest and source archive. A version string does not prove freezing.

The complete trajectory rosters must be disjoint and cover every row. Root-source overlap,
inconsistent trajectory roots, duplicate steps/transitions, changed vocabulary or latent width
are rejected. Renaming a trajectory does not erase its source identity. The host must also
audit aliases and cross-corpus overlap not recognizable from declared digests.

Five probes share the same fit/test rows: before endpoint, after endpoint, displacement,
concatenated endpoints and concatenation with seeded shuffled FIT labels. Test labels never
enter fitting. The permutation is retained, including identity permutations and ineffective
single-class controls. No validation-based hyperparameter search is performed.

Each probe fits one-hot action targets with an intercept using FP64 SVD least squares and fixed
`rcond=1e-12`. Rank, design width, rank deficiency and retained condition are reported. This
declared truncation is part of the linear probe, not a claim of an invertible representation.
Overflow/underflow and unsupported numerical systems fail visibly, without partial reports.
Scores include MSE averaged over rows and action coordinates, per-coordinate R2 and joint
variance-weighted R2. Constant targets give `None` with `constant_targets`, and negative R2
remains negative. Argmax accuracy uses the first declared action for exact ties; tie counts and
raw unbounded linear scores are retained. These scores are not calibrated probabilities.

The `forward` block separately reports supplied one-step latent MSE, its observed/planned
denominators and missing step IDs. This is not multi-step rollout-ranking consistency. A good
inverse probe cannot fill in absent forward predictions or establish successful control.

## Separate rollout-consistency experiment

Call `compare_rollout_costs(cost_rows, system=..., enabled=True)` on a predeclared candidate
cohort. Each `RolloutCost` includes trajectory, decision, horizon, candidate/plan identity,
predicted/realized costs and source references. The host must collect both costs for the same
plan under the same encoder and goal/cost definition, retaining execution records independently.
This function neither executes plans nor selects a winner, and cannot discover omitted candidates
that the host never registered. Register the complete roster before capture and use `None`
for missing costs; do not cherry-pick successful plans.

Each trajectory/decision/horizon is scored separately. Spearman correlation uses average ranks
for ties. Missing costs, constant ranks or fewer than two candidates produce explicit unscored
statuses; raw rows remain present. Duplicate candidates or plan digests within a group fail.
No pooled score mixes horizons or treats high correlation as task success.

## Evaluation and authority

All reports retain model/harness configuration, raw declared sources, numerical settings and
provenance, with `optimizer_eligible=False` and `authority_granted=False`. The unchanged shared
training guard rejects them. No credit, data admission, case classification, runtime action or
promotion follows from these diagnostics.

Compare externally trained forward-only and inverse-supervised encoders on identical held-out
trajectories, preprocessing, action labels, probe capacity and budgets. Include shuffled-label
and endpoint controls. Report multiple model seeds, paired uncertainty, task success, protected
17-case/calibration/constitutional regressions and costs separately through existing
[world-model value](world_model_ablation.md) and [improvement-audit](reliable_improvement_evaluation.md)
mechanisms. No extra canonical cases or fixed Studies A–F conditions are introduced.
CPU fixtures establish software correctness only; no learned encoder or empirical gain was tested.

## Source traceability

[REF-INTRINSIC-DYNAMICS](recommendations/RESEARCH_TRACEABILITY.md#ref-intrinsic-dynamics):
Jiamin Shi, Yuanhong Zhang, Zhiyang Liu, Jiawei Sun and Jingwen Fu,
*Learning Intrinsic Dynamics in Latent World Models*, arXiv:2610.21612v1, 2026-10-15, CC BY4.0.
Sections3.1–3.4,4.1,4.4,4.5 and reproducibility checked 2026-11-02. The primary source says
code release is planned; no upstream method-code license was established and no code was copied.
The [authored source note](recommendations/source_artifacts/intrinsic-2610.21612v1-method.txt)
and exact SHA256 in the [matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md) identify our
retained note, not publisher HTML/PDF bytes.
