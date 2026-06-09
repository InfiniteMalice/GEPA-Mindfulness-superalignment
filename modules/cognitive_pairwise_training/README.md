# Cognitive Pairwise Training

This package provides a lightweight, opt-in implementation inspired by Cognitive Pairwise Training (CPT). It is a metacognitive mid-training scaffold: it builds pairwise comparisons over public, structured reasoning traces so a trainer can learn a reusable boundary between trustworthy and flawed reasoning before downstream PPO, GRPO, DAPO-hybrid, or other outcome-heavy optimization.

The implementation is not surface refusal SFT, not a deception penalty, and not a requirement to expose private chain-of-thought. It uses public reasoning summaries, structured reasoning units, verifier status, confidence, abstention state, and trace metadata.

The current repository implementation is honest scaffolding for dataset construction, diagnostics, JSONL export, and trainer-facing loss terms. Full fine-tuning attachment should consume the exported examples or `CPTBatchMetrics` in the relevant trainer stage when `cognitive_pairwise_training.enabled` is true. The default is disabled.
