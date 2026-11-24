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

For scalar outcomes, pass factual and alternative predictions from compatible models in the
same units with the factual observation. Supply a named shared disturbance and validation
reference through `StructuralAssumptions`. The function applies equation 1 under the
declared assumptions and retains provenance for both predictions and the anchor. The host
must verify the referenced evidence supports residual reuse in its domain. Approximate
separability can introduce bias even with accurate factual predictions. Discrete or ethical
changes should be evaluated with paired replays.

The estimator adapts the additive/separable construction of Balestriero et al. The replay
adapts the separation of learned utility and exact constraints evaluated by Kang et al.
The empirical transfer to alignment has yet to be established.
[Counterfactual World Models](https://arxiv.org/html/2611.24778)
[Worth Another Look](https://arxiv.org/html/2611.24187)

`summarize_outcome_pairs` reports sample-average paired differences under the declared
sampling design. Observational pairs yield descriptive associations. Interventional
attribution requires the evaluator to establish the matching and manipulation protocol.
These are descriptive sample estimates with no population interval.

Run `python -m pytest -q tests/test_mdac_counterfactual.py` for a known additive disturbance,
unit/state/provenance mismatches, population contrasts, and paired discrete PEO with an
independent permission denial. Configuration uses the explicit keyword `enabled=True`.
Existing PEO datasets require no migration.
