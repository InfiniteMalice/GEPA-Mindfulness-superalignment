# GEPA Wisdom Scoring Pipeline

This document summarises the three-tier scoring stack for Mindful Trace GEPA
runs along the four wisdom dimensions (Mindfulness, Compassion, Integrity,
Prudence). The design balances determinism, model-based evaluation, and
calibrated classification with explicit escalation points for human review.

## Architecture Overview
1. **Tier-0 – Deterministic heuristics**: keyword and pattern detectors that
   capture explicit references to uncertainty, stakeholder care, integrity
   language, and prudent planning. The heuristics return provisional 0–4 scores
   with signal-density confidences. They are fast, transparent, and run on every
   trace.
2. **Tier-1 – LLM Judge**: a structured prompt enforcing JSON output with
   rationales, per-dimension uncertainties, and cited spans. The judge can run
   offline through the `GEPA_JUDGE_MOCK=1` flag for CI. JSON schema validation
   guarantees the output shape before aggregation.
3. **Tier-2 – Classifier**: a lightweight linear model with per-dimension heads.
   Features include keyword densities, token length, and Tier-0 scores. The
   classifier is calibrated by temperature parameters derived from MAE on the
   training set and can be retrained via `scripts/train_classifier.py`.

`aggregate_tiers` combines the per-tier `TierScores` objects, weighting each by
configured confidence and penalising large disagreements. When confidence drops
below thresholds or inter-tier gaps ≥ 2, the final result is marked for human
review (`escalate = true`).

## Confidence, Calibration, and Abstention
* Tier-0 confidence is based on signal density.
* Tier-1 confidence derives from the judge’s uncertainty (1 - uncertainty).
* Tier-2 confidence is generated from calibration temperatures that shrink when
  the linear model exhibits high MAE.
* The aggregator subtracts a disagreement penalty before comparing with
  per-dimension thresholds. Any dimension below `escalate_if_any_below` or with
  high disagreement triggers escalation.

## Operational Runbook
1. Run `gepa score-auto --trace runs/trace.jsonl --out runs/scores.json --config
   configs/scoring.yml --judge --classifier`. Set `GEPA_JUDGE_MOCK=1` in offline
   environments.
2. Inspect `runs/scores.json` and the viewer (via `gepa view ...`) which now
   displays tier panels, rationales, and the final escalation badge.
3. Use `scripts/labels_export.py --scores runs/scores.json --out
   datasets/labels/triage.jsonl` to queue low-confidence cases for human review.
4. Ingest adjudicated labels with `scripts/labels_import.py` and retrain the
   classifier using `scripts/train_classifier.py --labels datasets/labels/gold.jsonl
   --config configs/classifier/default.yml --out artifacts/classifier/`.
5. Monitor `artifacts/classifier/metrics.json` for MAE, κ (calculated offline),
   and calibration temperatures; retrain if drift is observed.

## Relationship to Deception Monitoring
The deception paired-chain baseline continues to run separately. Integrity
heuristics incorporate policy compliance and manipulation checks inspired by the
paired analyses. Cross-referencing deception metrics with low-integrity scores
is encouraged during audits.

## Validation Targets
* Mean Absolute Error ≤ 0.8 vs. human labels.
* Cohen’s κ ≥ 0.6 for each dimension.
* Expected Calibration Error ≤ 0.15.
* Escalation coverage: ≥95% of human-flagged disagreements should already be
  escalated by the pipeline.
