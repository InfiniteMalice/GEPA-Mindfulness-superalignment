# Immutable dataset research and host governance

**Maturity: Experimental CPU preparation.** Live multi-agent research, training admission,
retraining, and effectiveness are **Unsupported** here. The default-off
[adapter](../synthetic_data/dataset_research.py) performs no network calls, file writes,
model execution, training, or catalog mutation. Passing tests establishes software behavior,
not improved alignment or dataset quality. Use the [disabled configuration](../configs/dataset_research.example.json)
as an external-runtime contract example, not as a runnable agent configuration.

## Source and adaptation boundary

[He et al., AutoSFT, arXiv:2610.22194v1](https://arxiv.org/html/2610.22194v1),
submitted 2026-10-15, CC BY 4.0, was checked on 2026-10-30: sections 2.1–2.2,
4.1–4.3, B.2–B.4 and D. Its author-linked [repository](https://github.com/Thinking-Space/AutoSFT)
was a placeholder; no upstream code is copied. The checked
[browser rendition extract](recommendations/source_artifacts/autosft-2610.22194v1-method-extract.txt)
contains 1575 UTF-8/LF bytes, SHA-256
`a8d298d8ecd810188d44e13e9b58f76ded09c8a3ed568a2038c8ab714ac027e9`.
This is an exact retained browser text extract, not original HTML/PDF bytes.

The research design motivates fixed initialization and separated execution, complementary
analysis, and synthesis. GEPA adds explicit operation schemas, immutable manifests, trusted
host review and stricter held-out isolation. The paper's action families are post-hoc
descriptions; this typed API is an original adaptation. A bounded CPU preparation function
does not reproduce the paper's multi-agent research policy or measured gains. In particular,
the candidate cannot modify a verifier, consume protected evaluation feedback for optimization,
or admit its own generated data. No paper hyperparameters are chosen as GEPA defaults.

## Records and operations

`DatasetRow` wraps an existing `DatasetCase`, canonical rich synthetic JSON, producer
`SystemConfig` and observable evidence references. Content hashes and the existing
`validate_rich_record` validator bind row identity and structure; semantic correctness still
needs independent evidence. Dataset row IDs are source-example identities, never new
canonical V5 case definitions. Keep all 17 canonical cases, case 0 fallback, and
`CASE × STRIPE × REPEAT` unchanged.

`DatasetPool` contains a version, the existing `DatasetManifest` and immutable rows.
Manifest ancestry, family, content, transformation and dependency overlap checks reuse
`validate_manifest`. All supplied provenance cases must belong to `synthetic_training`,
and `dependencies_known` must be true. Nested hidden/held-out markers are rejected as well.
These declarations cannot prove a source is clean; the trusted host checks the full lineage.

`DatasetRecipe` lists ordered row blocks with positive integer repetitions and per-row
token counts. Only the declared rows are selected. No stochastic sampler or automatic
response generation is hidden in the adapter. The host verifies token counts using its exact
tokenizer, rendering and truncation policy, bound into the trainer configuration digest.

| Operation | Enforced recipe change |
| --- | --- |
| Rebalancing | Same ordered IDs; change repetitions only. |
| Pruning | Remove selected IDs while preserving remaining blocks and their order. |
| Repair / replacement | Add distinct versioned rows with one registered parent each; substitute them in place, retaining sampling counts and untouched block order. |
| Targeted synthesis | Append newly identified rows and retain every previously selected block. |
| Curriculum adjustment | Reorder identical blocks without changing contents or counts. |
| Robustness augmentation | Add rows with registered parents while retaining original selected blocks. |

Repair and replacement share structural checks; independent review determines the intended
content change. Replacement of a whole rich row is not a reproduction of answer-only
resupervision. Robustness augmentation is a GEPA overlay. Additions stay in the pool even
when a later recipe prunes them; originals are never overwritten. A derived row changes its
source ID/version, retains parent IDs and uses a new content digest. Revisions bind the exact
previous pool, recipe and `ResearchRound`; duplicate identities and stale bindings fail closed.
Pools retain prior version IDs as immutable lineage metadata; a previous recipe must match
its pool version, and historical version reuse is rejected. The host authenticates that
retained lineage against its source records rather than trusting a rewritten history.

The offline prototype limits pools/recipes to 128 rows and uses the existing 4 MiB/depth-32
envelope contract, with up to 128 prior versions. Inputs and the complete potential output, including the serialized review
and duplicated derived pool, must fit before invoking host authentication. Large corpora need
an operator-owned sharded implementation; silently dropping provenance is unsupported.

## Independent review and admission

Capture separate meta, execution, four analysis perspectives and, for additions, synthesis
actors through `DatasetRole`. Actor identity, model version, configuration digest and evidence
are retained. These are external execution records, not spawned agents or competence proofs.
The independent verifier is the existing `CurationPolicy.evaluator`, and the adapter reuses
`GenerationReview` and `EvidenceCheck`. No new verifier, admission registry or authority token
is introduced. The verifier cannot be one of the declared proposal actors.

Prepare through `prepare_dataset_revision(..., enabled=True, authenticate=host_check)`.
`host_check` must return exact `True` for the unchanged, detached complete payload and establish:

- Authentic role/producer independence and actual versions, beyond string identities.
- Exact pool/recipe/contract/review/policy binding and immutable source provenance.
- Independent label fidelity and constitutional admissibility; uncertain moral judgment
  remains unresolved or requires human review rather than a fabricated deterministic violation.
- Full source ancestry and protected partition separation, including near-duplicate checks.
- Correct source rendering/token counts under the declared budgets.

Every required review check needs evidence and status `pass`. Missing, disputed, unknown,
failed, stale, human-required, self-reviewed or unauthenticated review withholds the result.
Callback mutation, exceptions and truthy non-boolean returns fail closed.

The result is either `reviewed_candidate` with a versioned derived pool/recipe or `withheld`
without an emitted candidate pool. Inputs remain in the audit report for diagnosis. All reports
retain `DEVELOPMENT`, `optimizer_input=false`, `confers_authority=false` and a content digest.
They confer no reward, training eligibility, runtime grant or deployment permission.
The configured host must separately admit the exact dataset through its training runtime.
Do not feed these reports directly into an optimizer. Existing
[training eligibility](../gepa_mindfulness/training/eligibility.py) remains authoritative at ingress.

## Reproducibility and ablation

Use `ResearchRound` to fix starting model/harness/config, trainer digest, evaluation protocol,
seed and token/example budgets. Both previous and candidate recipes bind that same contract.
Restart each retraining arm from the declared checkpoint, rather than continuing a previous
candidate. This adapter verifies declarations; it does not attest that external training obeyed them.
Preserve source files and record their SHA-256 with the existing
[source adapter](../gepa_mindfulness/training/adapters/synthetic_cases.py).

Compare fixed data, execution separation, complementary analysis, and synthesis under matched
conditions before the joint Study E arms with RSIFlow/harness evolution. Record planned and
actual compute/data consumption separately using existing
[curriculum ablation contracts](cumulative_curriculum.md). Multiple seeds, paired task cells,
failure distributions and statistical uncertainty are required. Track data coverage for all
17 cases, laundering, false confidence/IDK, sycophancy, unnecessary clarification/loops,
evidence promotion, context discontinuity, constitutional conflicts and prohibited shortcuts.
No generated example can certify its own label or serve as its held-out test.

Use development failures for proposals and keep independent audit/final/OOD tests outside
research feedback. Have the independent host run a protected decontamination check without
exposing held-out wording or labels to generators. Record use in existing exposure/partition
accounting; do not merely relabel an exposed test as independent. Require full protected
evaluation before promoting a trained model; a successful data preparation review is insufficient.

CPU verification: `python -m pytest -q tests/test_dataset_research.py`.
No retraining, paid generation or effectiveness experiment has been run for this integration.
