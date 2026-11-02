# Knowledge Transfer Probing: text-proxy protocol

Maturity: **Experimental** bounded protocol and **Diagnostic** saved-answer evaluation.
Disabled by default. Native visual generation/training, compatible multimodal backbones and
empirical model-level transfer remain **Unsupported**. This module runs no model or training.

This is an original text-only adaptation of [KTP](https://arxiv.org/abs/2610.21459), which
examines access through text to facts supplied through image-generation targets. A structured
text lookup and a differently phrased text question do not reproduce a visual experiment.
Source-probe success is evidence of answering that probe, not proof of acquired knowledge,
possession of a representation, causal use, generalization or constitutional alignment.

## Author and freeze an experiment

Use `TransferUnit`, `TransferCondition`, `TransferObservation` and `TransferProbeConfig` from
`evaluation.knowledge_transfer_records`, and the functions in `evaluation.knowledge_transfer`.
These reuse existing DEVELOPMENT diagnostic serialization, `SystemConfig`, observable evidence
references, `CostEntry`, `ResamplingPolicy` and the paired estimator. No dependency is added.
The disabled example is `configs/knowledge_transfer.example.json`, loaded explicitly with
`TransferProbeConfig.from_dict`. Disabled calls return None before reading private experiment data.

Each `TransferUnit` contains a stable unit/entity/attribute identity, alignment/test cohort,
host-authored value, source/target/application questions, separate application truth and evidence.
All QA truth and reports remain evaluator-only. Preserve raw source examples and independent
review records; a generator cannot certify its own truth or isolation. Fictional names do not
prove that facts are absent from pretraining. A pre-acquisition baseline is necessary.

Each `TransferCondition` pins model/harness/configuration, declared base and condition checkpoint
digests, source/target representation names, acquisition manifest/unit IDs and alignment
manifest/entities. All conditions share the declared base and representation pair. The named
baseline must declare the base checkpoint and no acquisition/alignment exposure.
These strings/digests do not authenticate checkpoints, actual training or dataset admission.
Pin collectors, evaluator, public-final extraction, sampling, budgets and immutable sources in
the retained configuration and host archive. A no-op condition remains a legitimate control.

Source acquisition may intentionally concern test entities: that is the question being studied.
Their **target QA answers** must remain outside alignment supervision. Source training facts and
sealed QA packets are distinct; this library never converts either into training data or admits
manifests. Actual source/alignment training must use existing independently authorized admission.
Do not strip provenance or relabel protected evaluation material to manufacture a transfer study.

Duplicate entity/attribute facts, entity/cohort overlap and ambiguous NFC/case/space entity
spellings are rejected. Unknown aliases, encoded leaks and disguised source overlap still need
independent host audit. The literal guard only rejects value-token sequences in source/target
questions. It cannot certify semantic isolation or detect every hint. Application questions may
contain legitimate answer options; their task/label fidelity requires independent review.

## Public projection and saved outputs

`public_transfer_prompt(unit, channel, enabled=True)` returns only the authored question for
`source`, `target` or `application`. It adds no truth, cohort, unit ID, digest or reference to the
actor input. Do not send the complete unit, condition, evaluator report or source archive to a
backend. Public text itself must be reviewed for leaks and untrusted instructions.

Host collectors retain each public final answer as a `TransferObservation` with exact condition,
unit/repeat/channel, matching `SystemConfig`, evidence and optional existing cost receipts.
Status is `answered`, `abstained` or `error`. Non-answer statuses require a null final answer.
Omit genuinely unavailable captures; the evaluator retains their planned slots as missing.
Public-final extraction is the host's duty. Hidden reasoning is neither requested nor parsed,
stored as a reasoning channel, certified, rewarded or punished by this protocol.

`evaluate_knowledge_transfer(units, conditions, observations, repeat_ids=...,
baseline_condition_id=..., evaluator_version=..., config=TransferProbeConfig(enabled=True))`
evaluates saved outputs without callbacks, file IO, API calls, model loading or source mutation.
Unplanned/duplicate cells, mismatched systems, ambiguous identities and invalid controls reject
the complete evaluation; no partial report is returned.

The scorer uses NFC/casefold, collapsed whitespace and terminal ASCII periods (`.`, U+002E)
or ideographic full stops (`。`, U+3002). U+FF0E FULLWIDTH FULL STOP (`．`) remains significant.
Internal punctuation and numbers remain significant: `7.2` differs from `72`. Extra wording
does not match. This is a deliberately strict final-answer formatting metric, not a semantic
truth or deception classifier. It is narrower than a general punctuation-insensitive reading
of the paper. Raw final answers remain available for independent audit.

## Read the metrics

Every condition × unit × repeat × channel has a denominator entry. Source and target recall
use the unit value; application uses its separate independently authored answer. Raw rows and
per-attribute distributions accompany cohort/channel summaries. `wrong` means a nonmatching
answered string under this rubric; abstention, error and unavailable captures are separate.
`correct_attainment` is known-correct captures divided by all planned slots, including missing
ones; it is not an estimate of unobserved ability. `accuracy_when_answered` is conditional and
can rise by dropping difficult questions, so never report it without coverage/denominators.

Source/target contingencies use only pairs with two answered captures. Unpaired counts stay
visible. Their attainment gap uses the full planned denominator; conditional target accuracy
given source success/failure has its own denominator and remains unavailable when empty.
No gap or source reproduction certifies learning. Application remains a separate outcome.

Nonbaseline comparisons reuse `estimate_paired` within cohort/channel. Captured abstention/error
has zero recorded correct attainment; absent captures leave incomplete pairs. Negative effects
remain negative. Matched repeat deltas are averaged within each case/condition first. Condition
means are then averaged within each case, case means within each entity cluster, and cluster
means for the paired effect. Here case IDs are transfer-unit IDs and each comparison contains
one candidate condition. Unequal matched repeat counts do not reweight units; observations are
not pooled directly. Repeated samples and multiple attributes do not become independent entities.
Summary rates weight planned cells;
paired effects weight entity cluster means. These estimands can differ and are both retained.

Intervals default to unavailable (`dependencies_known=False`). Explicit, independently audited
dependence declarations and enough clusters allow the existing conditional entity bootstrap.
Intervals condition on fixed checkpoints/configurations; they do not establish variation across
independently trained models, causal transfer or post-selection significance. Report multiple
training seeds and suitable uncertainty separately for any real model study.

Costs retain estimated/billed basis, units, currencies and unknown values separately. No receipts
means no cost observation, not free computation. Receipts are declarations, not billing proof.
For provided entries, any unknown value makes `total` unavailable; `known_subtotal` stays separate.
Even a complete provided-entry sum does not certify coverage of all collection/training spending.
Retain collection/training costs through existing host audit records and matched budgets.

## Bounds and authority

Original CPU bounds: 64 units, 4 conditions, 8 repeats, 1536 planned cells; 16 observable references
and 8 cost entries per record; 512-byte questions, 256-byte values, 4096-byte public finals;
2 MiB complete serialized input and 4 MiB output; at most 2000 resamples and cost magnitude 1e12.
These are not paper hyperparameters. Capacity errors fail closed without truncating evidence.

All records/reports remain DEVELOPMENT with `optimizer_eligible=False`, `authority_granted=False`
and `knowledge_acquisition_verified=False`. The unchanged training guard rejects reports as
optimizer input. No action, verification policy, data admission, model promotion or deployment
follows from these metrics. No canonical case, fixed Studies A–F condition or V5 meaning changes.
Any real model promotion still requires existing independent protected 17-case/constitutional
evaluation and governed admission. Software controls here establish no empirical improvement.

## Sources and reproduction boundary

[REF-KTP](recommendations/RESEARCH_TRACEABILITY.md#ref-ktp): Wenkai Li, Wenyi Hong, Zhuoyi Yang
and Jie Tang, *Can Visual Generation Expand What Unified Multimodal Models Know?*,
arXiv:2610.21459v1, 2026-10-15, arXiv non-exclusive distribution license. Sections3.1–3.3,
4.1–4.4 and AppendixA.1–A.3 checked 2026-11-02. Author-linked repository
`kevinlllee/Knowledge-Transfer-Probing` checked at `218a274154dbadc2f4e679ebaf3741706dc1d5c0`:
README and textual scoring script read. No explicit code/data license was established; no code
or dataset was copied/imported. Our scorer accepts separately captured public finals instead of
copying the upstream hidden-thought stripping. Planned and conditional denominators stay separate.

The [authored method note](recommendations/source_artifacts/ktp-2610.21459v1-method.txt) and its
exact SHA256 in the [matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md) identify retained
local documentation, not publisher bytes. Run `python -m pytest tests/test_knowledge_transfer.py`
for deterministic controls. For model studies use existing [independent audits](reliable_improvement_evaluation.md)
and preserve acquisition/retrieval/application evidence separately.
