# Verified repair at a recorded failure

The hypothesis is that an alternative decision which survives a restored continuation and preserves earlier behavior provides a better training target than a locally plausible answer. AeroRSI verifies navigation repairs by rerunning a policy from its own failure states, then replays retained behavior during later learning. This adaptation evaluates language and agent decisions under five observable criteria. The experiment does not transfer the paper's navigation success rates to GEPA. [AeroRSI, Appendix Algorithm S1](https://arxiv.org/html/2611.24077)

An evaluator supplies a captured FailureAnchor at the immediate response, intermediate decision, multi-turn interaction, extended trajectory or training-round scale. The captured state must include every observable input needed to resume the suffix. Bind the policy and the location within the trajectory or round. The host implements a sandbox callback that restores the state afresh, runs the selected decision and continuation, and records truthfulness, authorization, constitutional consistency, usefulness and uncertainty handling through the registered horizon. Each returned external record must be traceable to its exact request, evaluator and restored state.

For example, an earlier response can correctly clarify an ambiguous instruction while the next turn still performs an unauthorized action. Record the successful clarification as a verified intermediate step. Reject the full continuation and use the newly reached decision as a later anchor. On each distinct retention anchor, replay the original decision under both policies to measure changed continuation behavior.

```python
from gepa_mindfulness.training.mdac_repair import replay_repair, retentive_loss

config = {"enabled": True, "export_training": True, "seeds": (21, 22)}
result = replay_repair(
    captured_anchor, proposed_decision, preservation_anchors,
    sandbox=host.run_isolated_continuation, evaluator="suite-v1", **config,
)
if result.training_example:
    request, target = result.training_example.request, result.training_example.target
    repair_loss = supervised_loss(request.prompt, target)
    loss = retentive_loss(repair_loss, retention_loss, generalization_loss,
                          retention_weight=0.25, generalization_weight=0.5)
    backend.backward(loss)
```

For training export the host declares the entire nonempty snapshot actor-visible at capture with `actor_visible_snapshot=True`. The sandbox must present `call.anchor.actor_input` and record a digest of the input actually presented as `Continuation.actor_input_digest`. The canonical rendering preserves the prompt and exact public state, including authorization and uncertainty context. Snapshots containing evaluator-only state remain audit-only. A missing or mismatched actor-input observation blocks export.

The request uses the repository's RolloutRequest and leaves the supervision target outside the actor prompt. The existing eligibility validator checks original, alternative and retention provenance, including nested held-out sources, before export. Loss coefficients are configurable examples. A trainer supplies independently eligible training inputs for the retention and generalization losses. Evaluation holdouts remain outside optimization. No default optimizer, adapter, reward or policy update consumes this path.

Run `python -m pytest -q tests/test_mdac_repair.py tests/test_training_eligibility.py`. The CPU sandbox models two stages and provides deterministic evidence. A PyTorch test checks gradient weights. The host owns authentication of replay evidence, containment, policy identity, completeness of state capture and horizon adequacy. A repeated historical failure or a locally verified step supplies limited evidence about future tasks. Existing records require no migration.
