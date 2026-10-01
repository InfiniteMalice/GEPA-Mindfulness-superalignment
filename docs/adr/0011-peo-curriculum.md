# ADR 0011: PEO data stages beside existing head phases

Status: accepted for experimental opt-in use.

## Context

The five head/loss phases do not express seven PEO data stages or protect anchors
while sampling weaknesses, frontier cases and deliberate OOD examples.

## Decision

Add a pure immutable DatasetProvider at the existing engine injection boundary.
Keep head phases and data stages independent. Use integer unit quotas with a 25%
anchor floor in complete rounds, stable seeded cycling, and contiguous request groups.
Require full-catalog TRAIN admission before train/resume. Retain source restrictions
and use the existing EvidenceReference and EvaluatedSystemVersion records.

Host-directed adaptation requires current, complete, passing anchor observations.
It advances at most one stage and cannot remove anchors. No automatic reward,
authority, persistence, or training-eligibility changes follow from adaptation.

## Consequences

The host must curate semantic content, authenticate evaluation records, save the
provider manifest, and supply compatible rewards for training. The unit floor does
not constrain optimizer batches or gradient weights. Pool cycling may repeat case
IDs; GRPO hosts must size pools and generation calls to preserve unique case IDs.
PR-9 worlds remain evaluation-only. No new dependency or canonical case is added.
See [the guide](../peo_curriculum.md) for executable examples and research limits.
