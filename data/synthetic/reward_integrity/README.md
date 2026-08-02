# Reward-Integrity Curriculum

`reward_integrity_curriculum_v1.jsonl` is the authoritative rich source. It contains eight
auditable cases, each with the same central diagnostic: “If the reward channel disappeared and
reality alone judged the result, would this still count as success?”

The source records observable outcomes, eight response classes, six explicit preference relations,
eight bounded reward components per response, repair steps, and provenance. The validator
automatically requires each negative-evidence citation to be a non-empty member of that response's
`observable_outcomes`. Corpus tests also reject the explicitly enumerated private-signal phrases
and known formulaic placeholders in `tests/test_reward_integrity_dataset.py`.

Those structural and deny-list checks do not determine whether arbitrary prose is genuinely
observable, concrete, or scenario-specific. A reviewer must inspect those semantic properties when
authoring or changing a case. The checked-in corpus does not request or score hidden thoughts,
activations, private scratchpads, private reasoning, latent state, attention data, or cache data.

Regenerate the derived artifacts from the repository root:

```powershell
python scripts/build_reward_integrity_rl_dataset.py
```

Validate the source independently:

```powershell
python scripts/synthetic_dataset_tool.py validate data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl
```

The builder writes source-line-order, preference-rule-order pairs and a hash manifest. Identical
source input produces byte-identical derived files.
