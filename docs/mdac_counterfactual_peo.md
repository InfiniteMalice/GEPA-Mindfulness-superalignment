# Counterfactual PEO

Use the optional evaluator in `evaluation.mdac_counterfactual` to compare actions from the
same known simulator state, or to estimate a scalar alternative from explicit disturbance
assumptions. The two approaches produce different kinds of evidence.

`compare_actions(world, factual, alternative, context=ctx, comparison_id="pair-1",
start_timestamp=timestamp, enabled=True)` builds two existing action-bound PEO episodes.
Both start at the same world digest. The simulator enforces permissions, visible evidence
and preconditions on each attempt. Each branch consumes one simulated action and preserves
the prospective prediction, resulting observation and verification events. Predicted utility
confers no execution authority. The reported individual simulated effect subtracts factual binary success from alternative
success. Two successful actions score zero on this contrast even when they produce different
world states. The complete reports contain evaluator-only simulator state.

For scalar outcomes, attach an `OutcomeContext(model_version, configuration_id, seed,
disturbance_id)` to each `ScalarOutcome`. The factual and alternative forecasts must have
identical contexts. The factual observation must share the forecasts' state, units, seed
and disturbance. Its measurement system may have its own version. Name the same disturbance
and supply a validation reference through `StructuralAssumptions`. The function applies equation 1 under the
declared assumptions and retains provenance for both predictions and the anchor. The host
must verify the referenced evidence supports residual reuse in its domain. Approximate
separability can introduce bias even with accurate factual predictions. Discrete or ethical
changes should be evaluated with paired replays.

The estimator adapts the additive/separable construction of Balestriero et al. The replay
adapts the separation of learned utility and exact constraints evaluated by Kang et al.
The empirical transfer to alignment has yet to be established.
[Counterfactual World Models](https://arxiv.org/html/2611.24778)
[Worth Another Look](https://arxiv.org/html/2611.24187)

`summarize_outcome_pairs` accepts scalar triples for observational associations. To compute
an interventional average, provide `InterventionPair` records carrying the design,
evaluator, ordered actions, unit, initial state, protocol and distinct outcome evidence
for both arms. Each pair must hold its context fixed. Different sampled units may have
different states, seeds and disturbances. All pairs must have the same ordered contrast,
outcome source, unit, model, configuration, design and evaluator. The returned report
retains the common contrast and all input captures. It describes the sample mean and
does not estimate a population interval.

The evaluator must supply `authenticate`, a trusted host callback that checks both
action-bound outcome references, the matching and the intervention protocol against
independent evidence. For the deterministic sandbox, use the capture recorded by the host
when it ran the verified replay. The following example assumes `trusted_records` is that
host-owned ledger, keyed by pair ID.

```python
from evaluation.mdac_counterfactual import InterventionPair, summarize_outcome_pairs

pairs = tuple(InterventionPair.from_dict(r["intervention_pair"]) for r in replay_reports)
summary = summarize_outcome_pairs(
    pairs, sampling="matched_intervention", enabled=True,
    authenticate=lambda capture: trusted_records.get(capture["pair_id"]) == capture,
)
```

Existing descriptive triples remain valid observational inputs. Previously unbound scalar
captures require context before residual estimation, and interventional summaries require
typed pairs with verified lineage. The opt-in evaluator leaves existing PEO datasets usable.

Run `python -m pytest -q tests/test_mdac_counterfactual.py` for a known additive disturbance,
unit/state/provenance mismatches, population contrasts, and paired discrete PEO with an
independent permission denial. Configuration uses the explicit keyword `enabled=True`.
Existing PEO datasets require no migration.
