# Criterion-level directional regret

For an independently specified criterion, compare the incumbent and candidate on matched
tasks, conditions and repeats. `evaluate_regret` accepts existing `PairedObservation`
records, a frozen pair roster and a `ResamplingPolicy`. It calls the existing cluster-aware
paired estimator for uncertainty. It also retains each criterion's source schema, rubric
digest, measurement unit, pair identity, baseline score and candidate score. Missing pairs
are disclosed. Estimate each criterion separately for each evaluation partition.

Configure a gate with `CriterionGate(metric, "calibration", pass_threshold=0.1,
target=0.05, severity=1, hurt_weight=2, gap_weight=1)` for a lower-is-better
error metric, and call the evaluator with `enabled=True`. Thresholds and targets are in
the metric's reported units. The four transitions use the threshold. Hurt and gap use
the continuous difference from the incumbent and target, respectively. For a lower-is-better
metric their signs reverse. Weights change diagnostic priority only.

`compare_candidates` reports best and worst improvement on a single complete criterion
panel with identical incumbent outcomes. The report retains candidate-specific evidence
so a reviewer can inspect ties and individual regressions. It expects trusted, unmodified
output from the evaluator. The existing private promotion store and admission workflow
own candidate rejection and rollback decisions.

This adapts GUARD's positive hurt and gap terms to the existing paired audit interface.
The extra severity weights and application to decomposed ethical criteria are research
extensions. Passing a finite buffer gives evidence only for the tested scope. Use the
existing independent audit, final-test and OOD partitions to examine transfer.
[GUARD](https://arxiv.org/html/2611.23476)

The CPU tests exercise all four transitions, opposite metric directions, missing pairs,
and incumbent mismatches. Run `python -m pytest -q tests/test_mdac_regret.py`.
Existing GEPA records and honesty reward configurations require no migration.
