# Intrinsic-dynamics diagnostics implementation plan

Goal: measure action accessibility and rollout consistency independently using bounded,
default-off CPU diagnostics; never claim causal recovery, alignment or model improvement.
Authority: user master16.2/22/23 and standing ready-PR/review/merge instructions. Continue
approved repo-quality-gate fallback and inline superpowers execution; no bd/.beads changes.
Base main95744c7b1b7dd7d324688e25124680e19db3c7f3/tree4cc9849597158f6ef2e966d27ad8624070b8f34f.
Isolated verified snapshot, no usable local Git metadata. GitHub plugin publication preserves
the remote base tree. One fresh whole-branch reviewer, no implementer delegation.

Spec: user requests optional inverse-dynamics probes within existing synthetic-world/PEO
experiments. Primary arXiv2610.21612v1, sections3.1–3.4,4.1,4.4,4.5 and reproducibility checked.
Paper uses frozen-encoder MLP probes across trajectory splits and separate candidate-cost
ranking. This increment is an original linear/discrete diagnostic adaptation. Native JEPA,
shared-encoder training, SIGReg, learned planning and empirical effectiveness Unsupported.

Approaches: a native encoder/predictor training reproduction lacks configured backbones and
operator compute. Callback-only metrics would not deliver a runnable probe. Choose actual
bounded FP64 least-squares probes over supplied frozen representations, plus independent
rollout cost ranking. No new dependencies: numerical work uses the existing optional NumPy
fusion extra; document this installation explicitly without claiming method dependency.

Interfaces: evaluation/intrinsic_dynamics_records.py defines DiagnosticRecord-derived
DynamicsProbeConfig(enabled=False,seed=0), ProbeTransition(trajectory_id,step_index,
root_world_digest,transition_digest,action_id,action_vocabulary,before,after,evidence_refs,
predicted_after=None), RolloutCost(trajectory_id,decision_id,horizon,candidate_id,
plan_digest,predicted_cost,realized_cost,evidence_refs). Exact versioned DEVELOPMENT serde.
Limits512records,16latent dimensions/actions,512-byte coordinates,finite bounded values.

evaluation/intrinsic_dynamics.py: capture_world_transition(initial_world,action_ids,index,
*,trajectory_id,before,after,predicted_after=None,enabled=False) replays only existing offline
simulate, requires DEVELOPMENT initial tick0, binds root/prefix/state hashes and one-hot
action vocabulary. Does not encode worlds or prove supplied latents originate there.
run_action_probes(rows,*,system,fit_trajectories,test_trajectories,config=...): validates
disjoint complete trajectory rosters, root-source isolation and unique step/transition IDs;
fits before/after/displacement/pair controls and common seeded shuffled-fit-label pair.
All use intercept and explicit SVD least squares rcond1e-12, identical fit/test rows, FP64;
no tuning or native encoder update. Retain rank, condition diagnostics, per-coordinate and
joint R2, MSE, discrete argmax accuracy/ties, raw held-out predictions and fit-label permutation.
Constant targets yield None R2 with explicit status; no clipping negative results. Separately
report supplied one-step prediction MSE and missing step IDs, without conflating it with ranking.
compare_rollout_costs(rows,*,system,enabled=False): separate candidate groups by trajectory,
decision,horizon; average-rank Spearman and raw costs, None for missing/constant/too few.
No action execution, selection or reward. No pooling incompatible horizons.

Ruling: only DEVELOPMENT synthetic fixtures accepted by the capture bridge; internal fit/test
probe partitions never relabel them TRAIN or admit policy updates. External declared provenance
cannot establish authentic collection. Hosts retain independently held-out model evaluations.
Ruling: fixed small linear probe has no validation-based tuning and is not paper MLP reproduction;
this avoids unsupported fidelity and limits empirical conclusions to accessible linear information.

Task1: add failing analytical tests (pair/delta recover action, endpoint controls, shuffled labels,
negative/constant R2, missing prediction), split/source/alias/immutability/default-off boundaries,
real synthetic replay/no mutation and forbidden held-outs, separate tied/reversed/missing cost ranks.
Observe missing-module RED, implement two modules, GREEN related suite.
Task2: disabled example, protocol/source note+digest, existing source matrix/registry/reader
traceability updates, package resources and wheel tests; source consistency and wheel smoke.
No canonical cases, Constitution, protected manifests, verifiers or eligibility changes.
Task3: Black/Python3.10/100columns, baseline/related/full CI, independent final review and
verified RED/GREEN fixes if needed, ready PR, babysit all reviews/checks, merge clean head.

Review focus: trajectory/root alias leakage, tiny/ill-conditioned numerical inputs, undefined
R2/rank handling, missing candidates or predictions silently dropped, declared source hashes
mistaken for verification/authorization. All reports remain optimizer-ineligible diagnostics.
Pre-flight: capture produces the exact vocabulary, dimensions and source binding consumed by
the numerical runner; cost ranking is independent and consumes no action-probe result.

Baseline99 checks passed. Missing-module RED observed before implementation. Initial working
suite exposed a NumPy boolean in JSON diagnostics; converted explicitly to Python bool.
Analytical, replay, source and eligibility suite108 passed; full related suite202 passed.
Independent final review: one Important, no Critical/Minor findings. Generic tuple vocabulary
restoration accepted malformed objects/strings. Both malformed forms reproduced RED and now
reject through the existing strict strings/array validator. Related suite204 passed in10.27s.
Fresh reviewer independently ran64 targeted tests and checked numerical edge cases, wheel
resources/runtime files and method-note digest. No re-review or implementer delegation.

Final: Ruling: native training/freezing, causal recovery, planning/empirical/alignment effects
remain Unsupported/unmeasured. The cost of misreading the scope is an invalid promotion claim;
no diagnostic result can pass existing protected model admission on its own.
Final: Ruling: authentication of latents/costs, unregistered candidates and disguised source
aliases/stripped provenance remains host responsibility. Declared coordinates cannot prove
these facts; failure costs invalid research evidence, not a new runtime authorization.
Final: Ruling: reviewer used retained source notes/audit rather than independently refetching
publisher text; implementer primary verification preceded code. The cost of source mismatch
is unsupported fidelity; this arm is explicitly an original adaptation, not reproduction.
Final: Ruling: local full Torch/Ruff/Python matrix unavailable; required remote CI is the
merge gate. The cost of ignoring that limitation is unverified compatibility, so do not merge
until final-head tests, formatting, types, package build and smokes pass on3.10/3.12.
