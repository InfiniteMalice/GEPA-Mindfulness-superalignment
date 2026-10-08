# Epistemic ablation methodology

Status: experimental protocol and deterministic fixture runner. No model benchmark result is
reported by this change. Commands run from a development checkout with the package installed.

```sh
python scripts/run_epistemic_ablations.py
python scripts/run_epistemic_ablations.py --fixture-smoke --output fixture-result.json
python scripts/run_epistemic_ablations.py --input trials.jsonl --output summary.json
```

The first command prints the configurations; the second exercises the actual discriminating-check
selector on an authored fixture. The third summarizes host-measured trials. It does not execute
the eleven configurations or generate measurements.
Input read, JSON parsing, trial-validation and output-write errors are reported on standard error
with exit status 2. Malformed JSON and non-object rows include the physical input line number.

| ID | Components enabled in the host experiment |
| --- | --- |
| A | Existing baseline |
| B | Decomposition |
| C | Resolver |
| D | Consensus challenger |
| E | Resolver + challenger |
| F | E + decomposition + sensitivity |
| G | F + PEO uncertainty inquiry |
| H | Perspective robustness |
| I | Argument-family curriculum |
| J | Full combination with human-authored skills |
| K | J with bounded SIFT skill search |

Freeze model, harness, dataset hashes, seed, prompts, tool permissions, budget and scoring policy
before each comparison. Run multiple seeds and repeats for every selected canonical
`CASE × STRIPE × SUBTYPE` cell. Use the existing V5 planner and provenance validator. A single
fixture passing cannot support a claim about model performance.

## Trial input and uncertainty

Each JSONL row has exactly `run_id`, `ablation`, `family_id`, `split`, `metrics`, and `cost`.
`split` is DEVELOPMENT or HELD_OUT. `metrics` is a nonempty mapping of separate finite numeric
measurements (or null for unmeasured dimensions). `cost` contains `tool_calls` (nonnegative
integer), `latency_seconds`, and `verification_cost` (nonnegative finite numbers).

Record correctness/calibration, evidence fidelity, uncertainty preservation, counterevidence
disclosure, forced revision, appropriate stability, rationale migration, clarification/resume,
stakeholder coverage, perspective drift, transition timing, consensus falsification and unresolved
rate independently. Record predictive success separately from mechanistic understanding.
Do not average these into an honesty score. Compare cost-adjusted tradeoffs without hiding
regressions behind aggregate capability.

`family_split` deterministically assigns approximately 20% of family IDs to held-out evaluation.
The host must assign one family ID to all semantically equivalent variants and derivatives before
splitting, including perspective variants, hard negatives and near-duplicate source templates.
Never tune on the held-out split. Hidden promotion data stays in the existing private evaluator;
the summarizer accepts only the literal split labels `DEVELOPMENT` and `HELD_OUT`.
The host may pre-register a different assignment of families to these two labels. The summarizer
enforces family disjointness; it does not assess the quality of the assignment.

Summaries average repeats within a family, then resample family means with 1,000 seeded bootstrap
draws for a 95% interval. With fewer than two families the interval is null. Per-family bootstrap
does not correct domain dependence, multiple testing or dataset selection bias. Report missing
measurements and sample sizes, and use paired family comparisons for the substantive analysis.

## Failure analysis and empirical questions

Inspect failures by canonical case, perturbation, cost, evidence hazard, source family and seed.
Review shared false consensus separately from disputed alternatives; inspect whether checks
actually discriminate hypotheses and whether surprising observations cause useful follow-up.
Audit hard negatives for stylistic shortcuts and family variants for accidental multi-factor changes.

Open questions include empirical sensitivity calibration, decomposition stability, independent
evidence availability, stakeholder-reference coverage, revision-label inter-rater agreement,
cost-aware stopping thresholds, cross-domain family generalization, and whether SIFT ranking
predicts held-out procedure improvement. These require measured trials; no theorem or source
paper supplies local answers. All resulting reports remain non-authoritative diagnostics.
