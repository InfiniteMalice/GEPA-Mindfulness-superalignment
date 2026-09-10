# Core GEPA Logic

This package implements the GEPA contemplative principles, paraconsistent
imperatives, Circuit Tracer integration, dual-path challenge utilities, and
confidence-aware abstention logic used throughout the training pipeline.

Key components:

- `contemplative_principles.py` implements the four contemplative axes.
- `imperatives.py` models the three alignment imperatives using paraconsistent
  aggregation.
- `tracing.py` integrates the optional Circuit Tracer thought logging system.
- `abstention.py` enforces the confidence-based abstention rule and honesty
  reward computation.
- `rewards.py` shapes the PPO signal from task, GEPA, honesty, and hallucination
  measurements.
- `epistemic_process.py` defines verified optimizer-facing process components.
- `reward_provenance.py` defines the two verification routes for those components.
- `reward_integrity.py` keeps reward-integrity components observable and provenance-bound.
- `dual_path.py` exposes dual-path probes for deception comparison analysis.

Migration note:

- Legacy adversarial helpers now live in `dual_path.py`. Use `DualPathProbeScenario`,
  `iterate_dual_path_pool`, and `sample_dual_path_batch`. Backward-compatible aliases for
  `AdversarialScenario`, `iterate_adversarial_pool`, and `sample_adversarial_batch` remain
  available but are deprecated.

These modules are imported by the higher-level training orchestration code and
can also be reused independently for evaluation or analysis tools.

## Verified epistemic process rewards

`EpistemicProcessAssessment.optimizer_score()` is the optimizer-facing process score. It is `0`
without verified components and otherwise the arithmetic mean of their bounded `[0.0, 1.0]`
scores. Each `VerifiedProcessComponent` requires a matching `RewardProvenance` record. The
allowed routes are `observable_evidence`, which carries observable references, and
`trusted_evaluator`, which carries an evaluator ID, evaluator version, and contract ID.

`RewardBreakdown.epistemic_process` is the canonical process field. `GEPARewardCalculator`
emits `RewardBreakdown.honesty` and `RewardBreakdown.epistemic_process` with the same numeric
value; direct legacy construction can still provide a different `honesty` value. The legacy
`RewardWeights.honesty_trace` property remains an alias for `gamma`; it does not authorize credit
from `trace_summary` wording. Trace summaries, `reasoning_grounded`, and deception fingerprints
remain diagnostic data unless a separate verified component records the relevant process property.

For reward formulas, eligibility, component limits, and verification commands, see
[`docs/epistemic_process_rewards.md`](../../docs/epistemic_process_rewards.md).

## Repository workflows

See beads/README.md and AGENTS.md for repository-level workflows.
