# ADR 0013: opt-in contrastive hard-negative training

Status: accepted for experimental use.

## Context

PR-12 needs staged public state/answer ranking and a fair comparison of classifier,
JEV, CLM and CLM with causal negatives. Existing CPT records already represent
preference pairs. Existing worlds supply deterministic counterfactuals but prohibit
training admission. No JEV/CLM checkpoint is configured in this repository.

## Decision

Reuse CPT records, snapshot public inputs, and enforce recursive admission before
model callbacks. Add a two-candidate differentiable loss and optimizer loop with
fixed family ordering and an equal-exposure pooled control. Model architecture,
initialization and persistence belong to the caller. No new dependency is required.

Generate causal/laundering pairs from existing decisive relation pairs and trajectory
negatives through the validated PEO simulator. Retain complete source provenance
and non-TRAIN restrictions. Require separately admitted training data.

Compare available backends on the same labeled catalog in both answer orders.
Report missing arms, missing families, declared split-check status and raw scores.
Do not translate contrastive margins into runtime authority, reward or a mechanistic
claim. Raw margin scales remain specific to each backend.

## Consequences

The API can run real gradient updates without prescribing a pretrained architecture.
CPU tests establish integration behavior, not learned generalization. A controlled
four-model study needs admitted training data and host-supplied checkpoints. This
bounded PR implements the experiment machinery while preserving the 17-case suite,
default training behavior and existing safety boundaries.

See [usage and research interpretation](../contrastive_training.md) for contracts,
primary sources, hypotheses and validation commands.
