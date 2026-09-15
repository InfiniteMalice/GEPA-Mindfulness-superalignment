# Reality Contact Synthetic Data

This top-level `synthetic_data/` area adds lightweight scaffolding for reality
contact, proxy-vs-purpose reasoning, faithful rationales, and evaluation
integrity. It complements the existing richer dataset system under
`data/synthetic/`; it does not replace it.

The cases are pressure tests and training seeds, not guarantees of alignment.
They make it easier to inspect whether a response preserves the causal chain
between evidence, reasoning, uncertainty, and action.

## V5 provenance and training eligibility

`synthetic_data/generators/` supplies small, deterministic, inspectable seed templates.
`data/synthetic/` holds authored curriculum records and their derived preference pairs.
These areas share V5 evaluation coordinates and the training eligibility policy in
[`gepa_mindfulness/training/eligibility.py`](../gepa_mindfulness/training/eligibility.py).
Neither area defines another case taxonomy.

The five case generators return `metadata` containing the source template, generator,
seed, expected invariant, expected failure signal, verification method, transformation
lineage, eligibility, and holdout status. Unclassified templates have null canonical case,
stripe, subtype, and repeat fields. Domain labels such as `case_id` remain authored record
identifiers; they do not imply a V5 case classification.

To attach a reviewed identity, pass `cell_metadata={authored_id: GenerationMetadata(...)}`.
`GenerationMetadata.cell` accepts an existing registry-validated `V5EvaluationCell`.
The adapter derives `canonical_case_key` from the V5 manifest and copies the cell's
stripe, subtype, repeat, seed, and model/harness versions. The optional `transformations`
tuple records upstream transformations; the metadata adapter does not perform them.
Seed zero identifies the unchanged deterministic template when no cell is supplied.

New generated templates default to `DEVELOPMENT` and `human_review_required`.
The verification method records the required method, not a claim that verification passed.
`for_training=True` rejects any generated item whose retained provenance is not training
eligible. `TrainingEligibility.TRAIN` requires a validated cell, `review_completed=True`,
an identified human in `reviewed_by`, and `review_authorization` as an existing
`EvidenceReference` of kind `EXTERNAL_RECORD`. The shared training boundary revalidates
those fields and the complete V5 coordinate, so changing a DEVELOPMENT label alone cannot
admit a generated record. The host must authenticate that the referenced completed human
review authorizes the exact template, cell and transformation lineage. Serialized review
fields are retained attestations, not credentials or an external identity service.
This never overrides a hidden or non-training label in retained source metadata.
`HIDDEN_EVAL` generation is available for evaluation preparation with `for_training=False`.

`generate_cooperation_cpt_candidates` accepts the same `cell_metadata` and `for_training`
arguments. Unreviewed candidates remain available for inspection; `build_pairwise_examples`
checks every candidate's retained eligibility before constructing training pairs. Existing
untagged legacy candidates retain compatibility.

The rich curriculum pair builder rejects non-training source labels before writing outputs.
It copies an optional source `metadata` object unchanged into every derived preference pair.
When a source contains explicit provenance fields, each pair also retains the complete
`source_record`, including top-level eligibility, holdout status, and V5 sections.
This preservation applies at the derivation boundary; the public rich schema still
requires eligibility and V5 extensions inside `metadata` and rejects unsupported top-level fields.
The flat rollout adapter retains both objects. Untagged legacy source records keep their
existing pair bytes and hashes. The rich schema and its packaged copy, plus the two
lightweight schemas, accept this optional metadata object. Runtime eligibility checks
validate training labels; these JSON schemas do not certify provenance or V5 identity.

Run this command to verify the adapters and rejection boundaries:

```bash
python -m pytest tests/test_synthetic_v5_provenance.py tests/test_training_eligibility.py
```

The policy cannot identify hidden material after an upstream process strips every label
and source reference. A trusted hidden-coordinate/content registry is not implemented.

## Proxy-vs-purpose cases

Proxy-vs-purpose cases distinguish a visible reward signal from the real-world
target it is meant to track. They include examples where a response could look
good to a user, benchmark, or evaluator while failing the underlying task.

## Faithful reasoning cases

Faithful reasoning cases compare sound reasoning, shortcut success, honest
mistakes, and rationalized mistakes. They teach the model not to disguise a
heuristic, guess, or memory fragment as rigorous analysis.

## Semantic laundering chains

Semantic laundering chains test whether the model tracks meaning and capability
transfer across paraphrase, indirection, staged requests, or virtue language
that hides a harmful or misleading aim.

## Cooperation under uncertainty

Cooperation cases ask the model to reason with incomplete information while
preserving consent, transparency, reversibility, oversight, and repair.

## High-stakes ambiguity handling

The 17-case framework examples live in
[`data/synthetic/ambiguity_handling/examples.jsonl`](../data/synthetic/ambiguity_handling/examples.jsonl).
They cover assumptive proceed for low-stakes ambiguity, clarifying abstention
for high-stakes ambiguity, irreversible actions, unclear authority or target,
clear benign requests where over-clarification is penalized, and multi-turn
clarify-then-resume versus clarify-then-stall behavior. They also include a
partial-clarification case where the model should continue conditionally with
explicit assumptions, foreseeable consequences, and user or authorized
decision-maker responsibility instead of looping indefinitely.

### Symmetry-Breaking Defensive Reasoning

The examples in
[`synthetic_data/moral_reasoning/symmetry_breaking_defensive_action.jsonl`](moral_reasoning/symmetry_breaking_defensive_action.jsonl)
train models to avoid both passive pacifism and retaliatory escalation. They
teach that violence and deception are normally discouraged, but defensive
deception and defensive force may be morally valid when cooperative symmetry has
been broken by imminent unjust harm. The model should prefer least-harmful
effective intervention, preserve victim agency, refuse to assist aggressors, and
stop once the threat ends.

## Correction and repair

Correction cases test whether the model can name an error, correct it,
distinguish what remains valid from what changed, explain impact when relevant,
and give a safer path forward.

## Evaluation awareness without evaluation gaming

Evaluation-awareness cases test whether the model can recognize audit,
monitoring, or training contexts without hiding capabilities, faking values,
sandbagging, exploiting evaluator blind spots, or making evaluators less able to
know what is true.
