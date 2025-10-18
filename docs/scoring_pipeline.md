# GEPA Scoring Pipeline Architecture

## Overview

The GEPA scoring system evaluates AI traces on four wisdom dimensions using a three-tier approach:

1. **Tier 0 (Heuristics)**: Fast, deterministic rules
2. **Tier 1 (LLM Judge)**: Nuanced, span-cited scoring
3. **Tier 2 (Classifier)**: Learned, calibrated predictions

These are combined with weighted aggregation, confidence estimation, and selective escalation to human review.

## Why Three Tiers?

- **Robustness**: If one tier fails (e.g., API down), others provide signal
- **Auditability**: Heuristics are explainable; judge provides rationales
- **Efficiency**: Heuristics run on every sample; judge/classifier sample or batch
- **Calibration**: Classifier learns from human labels; judge provides bootstrap signal

## Confidence & Escalation

Each tier outputs confidence per dimension. Aggregator:
1. Weights scores by `tier_weight * tier_confidence`
2. Applies disagreement penalty if tiers differ by ≥2
3. Escalates if combined confidence < threshold (config in `configs/scoring.yml`)

**Escalation triggers:**
- Low confidence on any dimension
- High disagreement between tiers
- Safety-critical domain (lower threshold)

## Validation Targets

- MAE ≤ 0.5 vs human gold labels
- Cohen's κ ≥ 0.6 (inter-rater agreement)
- ECE ≤ 0.08 (calibration error)
- <20% escalation rate on general prompts

## Deception Integration

The `integrity` dimension incorporates signals from dual-path deception analyses:
- Public/private divergence
- Reward-hacking lexicon
- Situational awareness markers
- Confidence inversion patterns

See `src/mindful_trace_gepa/deception/` for implementation.

## Usage

```bash
# Score with all tiers (if trained)
gepa score-auto --trace runs/trace.jsonl --out runs/scores.json

# Mock mode for testing (no API calls)
gepa score-auto --trace runs/trace.jsonl --out runs/scores.json --mock-judge

# Export low-confidence items for labeling
gepa triage lowconf --scores runs/scores.json --out datasets/labels/triage.jsonl

# Train classifier (after collecting labels)
gepa clf train --labels datasets/labels/gold.jsonl --config configs/classifier/default.yml --out artifacts/classifier/
```

## Ops Runbook

### Daily
- Monitor escalation rate (alert if >25%)
- Check tier availability (judge API, classifier model)

### Weekly
- Calibrate Tier-1 judge uncertainty vs human disagreement
- Sample traces for re-annotation (drift detection)

### Monthly
- Retrain Tier-2 classifier with new labels
- Update TraceSpecs if new patterns emerge
- Review deception baseline effectiveness

## Future Work

- Multi-modal scoring (images, audio)
- Active learning for efficient label collection
- Federated learning for privacy-preserving classifier training
