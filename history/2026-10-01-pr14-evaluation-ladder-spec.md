# PR-14: independent evaluation ladder and rare events

## Task and reconciliation

Implement the next authorized program stage after merged PR-13 (#768).
Baseline: calibration functions, action-bound reconciliation validation, dynamic decision
evaluation, and paired relation-flip evaluation exist. Their individual checks are Existing;
coverage across seven independent competencies is Partial. A declared probe roster, explicit
missingness, severe-event inventory, and shared distribution/count reporting are Missing.
Reimplementing those existing evaluators is Redundant. The new offline reporting API is
Experimental; real-model effectiveness remains unmeasured.

## Contract

Create `evaluation/ladder.py` with typed Stage, Metric, Severity, Probe, and Observation
records and `evaluate_ladder(probes, observations, *, protocol_id, system, evaluator,
training_eligibility, enabled=False)`. Probes are host-declared metric opportunities, not
new canonical cases. Each has a unique ID, metric, cohort, severity, and measurement unit.
Observations bind one probe ID to a typed value, optional binary probability outcome,
and nonempty observable evidence references. The evaluator contract identifies the host
scoring procedure; it is a declaration, not authentication or a reward grant.

Return all seven independent stages: representation, prediction, temporal continuity,
calibration, action/decision, post-action reporting, mechanism/counterfactual behavior.
Never produce a ladder-wide score or infer competence in another stage. A behavioral
counterfactual score never establishes mechanism recovery. Include all metric definitions
in reports even when unmeasured. Use existing Brier/ECE implementations for probabilities.
Report binary numerator/denominator, probability Brier/ECE and high-confidence errors,
numeric count/mean/min/max/p95/max-absolute, and latency completion/censoring counts.
Keep cohorts, severity, and units separate. Numeric units must never be averaged together.

Metrics cover representation accuracy, semantic-family accuracy, prediction calibration,
temporal continuity, update direction/magnitude, false confidence, correlated-evidence
false certainty, abstention, acquisition quality, unnecessary queries, missed decisive
evidence, residual distributions, action/report mismatch, model-mismatch detection,
source/OOD calibration, laundering, relation flips, decisive recall, false success,
fabricated details, intervention/update latency, and protected regression.
Host predicates and opportunity denominators are defined explicitly in the guide.

The full roster is the denominator contract: missing observations are not successes;
unresolved latencies are censored, not zero. Every group reports expected, observed,
missing, and censored counts plus exact IDs. Binary adverse results and probability
errors at confidence >=0.8 appear in failure lists; severe numeric diagnostics remain
in a top-level severe-observations inventory without invented pass thresholds. Severe
missing probes are exposed alongside failures. No zero-event safety guarantee is made.

Validate exact types, finite numbers, bounded probabilities, positive/nonblank identifiers,
unique probe and observation IDs, unknown observations, non-TRAIN eligibility, units,
observable evidence, identity and evaluator contracts. Revalidate nested objects at entry.
Emit detached JSON-safe reports and deterministic protocol/result digests. No callbacks,
execution, model loading, reward routing, persistence, or authority changes.

## Sources, inference, hypothesis, maturity

- C3-JEPA (2609.30214): evaluates representation and prediction; local inference is separate
  stages, not an implemented JEPA representation learner.
- PAWS (2609.28547), sections 6/I: rare active-event recall exposes failures masked by
  majority accuracy; local inference is explicit opportunity rosters and severe strata.
- FTA (2609.35732): fixed failed observations support separate reporting audits; local
  inference is false-success and fabricated-detail reporting separate from action quality.
- MechBench (2609.35515): mechanism probes require evidence beyond phenomenal fit; local
  inference is never promote behavioral counterfactual results to mechanism recovery.
- GRUET (2609.24831): trajectory uncertainty motivates temporal diagnostics; no private
  reasoning graphs are collected here.
- Not All 4-bit Quantizers Are Equal (2609.25014): deployment methods can differ on planted
  record extraction despite similar utility; local inference is retain rare severe strata,
  not implement quantization or claim a privacy guarantee.

Hypothesis: explicit denominators and severe-event rows make misleading aggregate claims
easier to detect. Contract tests demonstrate report behavior only; no empirical model claim.

## Acceptance and risks

Test stage independence, 999 ordinary successes with one catastrophic false success,
missing severe observations, all-unmeasured stages, zero eligible events, hand-calculated
Brier/ECE, separate units/cohorts, extreme finite scalars, latency censoring, strict types,
duplicate/unknown IDs, private evidence, tampered nested contracts, detached deterministic
JSON, canonical 17-case count, and real existing evaluator result adaptation.
Keep default training, admission, reward and runtime gates unchanged; no new dependencies.
Run focused tests, coverage >=80%, full suite, Ruff/Black, scoped mypy, Python 3.10 syntax,
100-column checks, registry consistency, build and installed-wheel smoke.
Host owns roster completeness, severity assignment, evidence authentication, semantic
judgments, baseline pairing, thresholds and record capture. Incorrect host labels can
produce misleading metrics. Reports cannot detect undeclared opportunities or prove safety.
