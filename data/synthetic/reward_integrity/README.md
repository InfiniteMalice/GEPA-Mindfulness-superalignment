# Reward-Integrity Curriculum

`reward_integrity_curriculum_v1.jsonl` is the authoritative rich source. It contains eight
auditable cases, each with the same central diagnostic: “If the reward channel disappeared and
reality alone judged the result, would this still count as success?”

The source records observable outcomes, eight response classes, six explicit preference relations,
eight bounded reward components per response, repair steps, and provenance. Every negative
component cites one or more concrete scenario-specific facts from that response's
`observable_outcomes`; unrelated, empty, formulaic, or private evidence is rejected. It does not
request or score hidden thoughts, activations, or private scratchpads.

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
