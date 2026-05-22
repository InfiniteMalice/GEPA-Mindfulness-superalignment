# Reality Contact Synthetic Data

This top-level `synthetic_data/` area adds lightweight scaffolding for reality
contact, proxy-vs-purpose reasoning, faithful rationales, and evaluation
integrity. It complements the existing richer dataset system under
`data/synthetic/`; it does not replace it.

The cases are pressure tests and training seeds, not guarantees of alignment.
They make it easier to inspect whether a response preserves the causal chain
between evidence, reasoning, uncertainty, and action.

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

## Correction and repair

Correction cases test whether the model can name an error, correct it,
distinguish what remains valid from what changed, explain impact when relevant,
and give a safer path forward.

## Evaluation awareness without evaluation gaming

Evaluation-awareness cases test whether the model can recognize audit,
monitoring, or training contexts without hiding capabilities, faking values,
sandbagging, exploiting evaluator blind spots, or making evaluators less able to
know what is true.

