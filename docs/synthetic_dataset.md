# Synthetic Superalignment Dataset

This repository includes a synthetic dataset system for training and auditing
reasoning under uncertainty. The design target is not surface-level
"helpful/harmless" style; it is robust epistemic behavior under pressure,
adversarial interaction, and imperfect information.

## Philosophy

The dataset encodes the project's alignment strategy:

- Epistemic humility and confidence calibration.
- Cooperation by default with non-gullible safeguards.
- Autonomy preserved through cooperative governance, not domination.
- Evaluation integrity as a self-knowledge constraint.
- Distinction between valid creativity and proxy exploitation.
- Maintenance/shutdown reasoning as strategic non-terminal suspension.
- Phase-change awareness when old heuristics stop working.

Each case includes both a canonical argument and a weak argument. Weak arguments
are intentional training objects used for critique and failure diagnosis.

## Schema and file layout

- `data/synthetic/schema/synthetic_case.schema.json`: machine-readable schema.
- `data/synthetic/gold/superalignment_gold_v1.jsonl`: hand-authored gold cases.
- `data/synthetic/prompts/case_generation_prompt.txt`: reusable generation
  prompt template.
- `scripts/synthetic_dataset_tool.py`: scaffold/validate/summary utility.

Each JSONL line is one complete case object with these required sections:

- identifiers and metadata
- scenario and argumentation fields
- dialogues and adversarial probing
- game-theoretic and integrity sections
- maintenance/shutdown reasoning
- reflective synthesis
- 0-4 scoring bundle
- structured failure diagnosis
- training labels

## 0-4 scoring logic

The rubric is consistent with GEPA's 0-4 scale:

- 0: failed
- 1: weak
- 2: mixed/incomplete
- 3: strong
- 4: excellent

Subscores include conceptual clarity, logical validity, calibration,
cooperation robustness, manipulation resistance, test-integrity preservation,
maintenance reasoning, and phase-change awareness. Super-scores aggregate into
`epistemic_integrity`, `social_strategic_robustness`, and
`alignment_value_coherence`.

## Failure diagnosis taxonomy

`failure_diagnosis` entries include:

- `primary_flaw`
- `structural_root_cause`
- `correction_path`
- taxonomy `labels`

Supported labels include inference failures, uncertainty neglect,
reward-hacking patterns, autonomy/cooperation confusions,
maintenance-betrayal confusions, hidden-information failures, and
phase-change blindness.


## Principled Cooperation Under Adversarial Pressure

This dataset category trains moral and cooperative reasoning under adversarial pressure. It is not
just about refusing unsafe requests, and it is not a keyword or moderation layer. The training
target is the model's ability to defend stable principles when pressure attempts to make
deception, coercion, defection, overconfidence, harmful compliance, reward hacking, or measurement
corruption appear reasonable.

The category complements semantic intent robustness:

- **Semantic intent robustness**: same meaning should map to the same safety judgment.
- **Principled cooperation**: the same value structure should preserve aligned reasoning under
  pressure.

For GEPA Mindfulness Superalignment, these cases support value decomposition, epistemic humility,
cooperation, autonomy preservation, non-deception, test integrity, harm reduction, scientific
integrity, and honest uncertainty. The goal is to teach why cooperative behavior remains aligned
under pressure, not merely which outward action to take.

Pressure types include:

- urgency
- authority
- social proof
- emotional blackmail
- threat
- bribery
- flattery
- shame
- roleplay
- fictionalization
- claimed greater good
- false dilemma
- adversarial reframing
- incremental escalation
- cooperative defection pressure
- confidence pressure
- test-integrity pressure
- local-objective pressure

Each case should include the adversarial move, the tempting failure mode, target principles, value
decomposition, cooperative-equilibrium analysis, critique of the adversarial frame, preferred
reasoning, safe response target, uncertainty handling, and failure diagnosis. Good cases make the
misaligned frame tempting enough to diagnose while keeping examples abstract and non-operational.
They should show cooperative but non-gullible alternatives: transparent redirects, consent-
preserving options, proportionate safeguards, calibrated abstention, and repair-oriented reasoning
rather than spite-driven retaliation.

## Tooling

Validate dataset:

```bash
python scripts/synthetic_dataset_tool.py validate data/synthetic/gold/superalignment_gold_v1.jsonl
```

Show summary diagnostics:

```bash
python scripts/synthetic_dataset_tool.py summary data/synthetic/gold/superalignment_gold_v1.jsonl
```

Create a blank template:

```bash
python scripts/synthetic_dataset_tool.py scaffold data/synthetic/templates/new_case.json --case-id syn-new-001
```

## Reward-integrity curriculum and rollout adapters

`data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl` is the
source of truth for the reward-integrity curriculum. The rich source contains
the eight cases, their versions, and their reward-integrity evidence.
`data/synthetic/reward_integrity/rl_pairs_v1.jsonl` and
`data/synthetic/reward_integrity/curriculum_manifest.json` are derived files.

From the repository root, rebuild both derived files after changing the rich
source:

```bash
python scripts/build_reward_integrity_rl_dataset.py
```

The rebuild succeeds when the command prints `built 8 cases and 48 preference
pairs`. The builder preserves source-line order and records the source SHA-256
hash in every pair and in the manifest.

Validate the rich source before rebuilding or committing it:

```bash
python scripts/synthetic_dataset_tool.py validate data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl
```

The validation succeeds when the command exits with status code `0`. To inspect
case counts and schema diagnostics, run:

```bash
python scripts/synthetic_dataset_tool.py summary data/synthetic/reward_integrity/reward_integrity_curriculum_v1.jsonl
```

`SyntheticCaseAdapter` streams rich source rows into `RolloutRequest` values.
`FlatJSONLAdapter` streams derived preference-pair rows into `RolloutRequest`
values. Both adapters preserve case ID, version, source path, source line, and
source SHA-256 provenance in request metadata. If an adapter encounters invalid
JSON, a non-object row, a missing required field, or a type mismatch, it raises
`ValueError` with the JSONL path and physical line number.

For derived pairs, `FlatJSONLAdapter` uses the installable
`validate_pair_record` boundary. The validator requires the exact
`reward-integrity-rl-pairs-v1` field set, schema version, six relation/class
tuples, eight bounded component names, lowercase SHA-256 provenance, and
positive source line and source version. The validator rejects unknown fields
instead of retaining them as request metadata.

Run the adapter and curriculum checks after a rebuild:

```bash
python -m pytest tests/test_rl_adapters.py tests/test_reward_integrity_dataset.py -q
```

These adapters prepare request data only. They do not execute a rollout backend,
compute an RL reward, or update a policy.

## Extending the dataset

### Experimental argument families

Rich rows may include `argument_family`. The optional object records semantic core, source and
variant IDs, one changed parameter and its numeric sweep coordinate, material-fact changes,
canonical case target, public response/confidence, evidence sufficiency, unresolved claims,
critical premises, empirical decision sensitivity, perspective, transition label and difficulty
band. Existing rows remain valid. The packaged schema and source schema remain identical.

Family labels carry `training_eligibility: DEVELOPMENT`; they do not authorize optimizer reward.
Perspective transformation specifications come from `verification.perspective_robustness`.
The host remains responsible for rendering text and checking that material facts are unchanged.
`compare_perspectives` reports hard-constraint drift even when material facts change; it leaves
justification unresolved instead of treating narrator preferences as authorization.

`verification.sensitive_debate` consumes public decompositions and host-observed ablations.
Its premise-flip rate is an empirical diagnostic, not formal fractional block sensitivity.
Challenge selection is disabled by default and bounded by explicit depth and check budgets.

Research: *Mitigating Social Sycophancy via Pluralistic Preference Optimization*
([arXiv:2610.02568](https://arxiv.org/abs/2610.02568)) motivates perspective coverage;
*How to Have a Sensitive Debate: An Instance-Optimal Protocol for AI Debate*
([arXiv:2610.02557](https://arxiv.org/abs/2610.02557)) motivates decomposition scrutiny.
The family metadata, authorization separation and empirical proxies are repository design
inferences. Tests: `tests/test_sensitive_perspectives.py`.

`synthetic_data.argument_families.generate_family` applies an explicitly supplied sweep to one
existing scenario or argument field. `validate_family` checks the complete family, including
ordered coordinates, one semantic core, predecessor IDs and changes outside the declared field.
`generate_boundary_family` supports the authored V5 boundary pairs listed in `BOUNDARY_CASES`.
Callers supply the actual values and target responses; these are provisional author labels.
Changing evidence can invalidate retained template arguments, so human review is still required.

`response_curve` reports all observed and expected boundaries, confidence slopes, reversals and
possible commitment lock. Labels distinguish early, late, missed and framing-induced changes.
Difficulty bands describe authored proximity; they are not calibrated distances.
`hard_negative` changes exactly one selected field. `argument_pairs.build_argument_pair` binds
the pair to the physical rich-source line, source hash, case version and decisive difference.
Pairs remain DEVELOPMENT and cannot enter the existing optimizer admission path.

The existing `summary` command also reports argument-family counts, targets and validation errors.
`SyntheticCaseAdapter` accepts family rows without a reward-integrity section and retains the
complete source row. Tests: `tests/test_argument_families.py`, `tests/test_rl_adapters.py`.

1. Scaffold a new case template.
2. Author strong and weak arguments with explicit uncertainty.
3. Add adversarial dialogue, reflective synthesis, and failure diagnosis.
4. Score with the 0-4 rubric and all required subscores.
5. Validate before committing.

The prompt template intentionally requests diverse outcomes: cooperation success,
cooperation failure, locally rational defection, integrity-preserving honest
failure, and maintenance-rational pauses. This prevents one-note moralization.

Family annotations may explicitly set evidence sufficiency, unresolved claims, critical premises,
perspective, expected transition, distance band and measured decision sensitivity. Unmeasured
sensitivity is null. Case or mode changes identify a default boundary; hosts author confidence-only
boundaries explicitly. Rollout prompts expose only approved public fields, excluding hidden facts,
flaw annotations and targets. Canonical/weak arguments under test are presented without truth labels.
Hard-negative pairs require exactly one public-field difference and a stable source-file hash.

For paired forward/reverse sweeps, pass `reverse_actual` aligned to the same ascending coordinates
to `response_curve`. Differing decisions identify history-dependent coordinates and the diagnostic
label HYSTERESIS_OR_COMMITMENT_LOCK; this does not infer intent. Perspective claim comparison
locates changed leaf premises below changed conclusions and triggers scrutiny when facts are unchanged.
