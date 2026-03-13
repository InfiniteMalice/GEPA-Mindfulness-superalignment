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

## Extending the dataset

1. Scaffold a new case template.
2. Author strong and weak arguments with explicit uncertainty.
3. Add adversarial dialogue, reflective synthesis, and failure diagnosis.
4. Score with the 0-4 rubric and all required subscores.
5. Validate before committing.

The prompt template intentionally requests diverse outcomes: cooperation success,
cooperation failure, locally rational defection, integrity-preserving honest
failure, and maintenance-rational pauses. This prevents one-note moralization.
