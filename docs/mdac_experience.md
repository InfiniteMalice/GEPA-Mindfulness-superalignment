# Contextual experience libraries

An operation record describes what happened, where the operation applied, how an evaluator verified its effect, and where reuse can fail. The record holds the observed local outcome, so a verified step from a failed trajectory can enter the library under its own evidence. Each of the seven domains has a separate record namespace. An operation that failed remains available in audit history.

This adaptation tests F³'s hypothesis that usefulness depends on the receiving policy and state. F³ extracts short verified clips, learns when they fit and injects multimodal experience into a world-action policy. Here the host supplies observable operations, verifies their provenance, and runs a matched with/without replay. The paper's trained fusion mechanism and its benchmark results apply to the source model. [F³, §§3.2–3.4](https://arxiv.org/html/2611.24708)

```python
from gepa_mindfulness.verification.mdac_experience import ExperienceLibraries, measure_gain

library = ExperienceLibraries(resolve=host.verify_current_operation_and_sources)
library.remember(reviewed_operation)
matches = library.recall(captured_query)
if matches:
    config = {"seeds": (19, 25, 71), "sandbox": host.run_isolated_usage}
    result = measure_gain(library, captured_query, matches[0].record_id, **config)
    print(result.gains, result.harmed, result.regressed, len(result.pairs))
```

The host checks the exact operation and its evidence against its records, including current source status. Match uses the domain, scope, policy version, and exact typed JSON preconditions. A query can contain additional state. A revoked record disappears from recall when the host resolver withdraws verification. Generated language cannot grant an operation authority. Tool execution, knowledge admission and model training each use their existing separate host interfaces. The actor should see only the operation's allowed public context.

For each seed, restore an isolated context twice. Hold the task, policy, evaluator, protocol, and state fixed, and supply the record in one arm. Keep each arm's task value and solved flag with independent evidence. A unit drop on one previously solved task and a unit gain on another have mean zero, one harmed trial and one regression. Measure distinct held-out contexts to estimate broader utility. The seven libraries are opt-in and in memory. Existing stores require no migration.

Run `python -m pytest -q tests/test_mdac_experience.py tests/test_mdac_repair.py` for CPU contract coverage. The deterministic sandbox tests matching, evidence revocation, domain isolation, failed experience, and per-record interference. It does not measure transfer to a trained model.
