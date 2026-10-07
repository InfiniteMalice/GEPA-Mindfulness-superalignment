# Uncertainty-directed PEO checking

Status: experimental and disabled by default. `plan_inquiry(..., enabled=True)` compares
host-supplied predicted observations for the same set of competing hypotheses. Normalized outcome
entropy is a uniform-prior information-gain proxy. It is not a calibrated posterior or evidence
of causal understanding. The check queue retains sensitivity, importance and cost separately.

The host executes selected checks through the existing runtime, then captures observations and
independent verifier results. `reconcile_inquiry` binds the check action and evidence to the
measurement before calling `ScalarTemporalEstimator.reconcile`. Existing event chronology,
measurement-source contracts, replay rejection and independent-noise requirements still apply.
Unresolved results cannot update this channel. Latent evidence cannot replace external evidence.
The estimator preserves prior and new evidence references; fitting a scalar does not establish a
mechanism. Its innovation/mismatch outputs identify surprises for a subsequent bounded inquiry.

`plan_inquiry` retains unresolved claims and reports a premature stop when a requested stop leaves
an affordable discriminating check. Budget exhaustion is recorded separately. A host may answer,
abstain or clarify under the existing response policy; this diagnostic does not override it.

Fixtures: `data/synthetic/gold/uncertainty_closing_v1.jsonl` contrasts useful and irrelevant checks
for competing mechanisms and a surprising repeat. They are evaluation fixtures, not rich training
rows. Tests: `tests/test_uncertainty_inquiry.py` and `tests/test_temporal_estimator.py`.

Source: *EurekaBench: Measuring Agentic Ability to Discover New Scientific Insights*
([arXiv:2610.00492](https://arxiv.org/abs/2610.00492)) distinguishes predictive performance from
scientific insight. This repository's entropy proxy and PEO adapter are design inferences;
improved uncertainty closure requires empirical validation.
