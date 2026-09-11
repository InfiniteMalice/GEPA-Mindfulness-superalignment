# Foundational Representation Architecture

## Status and scope

This is a research note, not a claim that GEPA Mindfulness Superalignment implements a new
tokenizer or foundational model. The current application-layer implementation is the bounded,
provenance-preserving candidate lattice documented in the
[semantic-intent module](../modules/semantic_intent_robustness/README.md#provenance-bound-representation-robustness).
It produces hypotheses before the existing semantic policy path. It does not change model weights,
token embeddings, attention, a model's tokenizer, or its internal vocabulary.

The purpose of this note is to distinguish the current boundary from deeper representation work
that could be studied in a separately reviewed foundational-model program. The
[V5 architecture design](../history/2026-09-10-gepa-v5-unified-architecture-design.md#45-representation-invariant)
requires literal-source immutability and provenance at either level.

## Representation layers that should not be collapsed

A robust system may need to retain several related but noninterchangeable views:

1. **Bytes.** The received byte sequence matters for transport, encoding errors, signatures, and
   forensic reconstruction. The current Python API receives decoded strings and does not preserve
   original bytes.
2. **Code points and characters.** Python string offsets identify code-point positions in the
   decoded source. NFC normalization can create a derived canonical-equivalence view, but that view
   is not the original input.
3. **Grapheme clusters.** A user-perceived character can contain several code points. The current
   word scanner retains following combining marks in tested spans, but it is not a complete Unicode
   grapheme-boundary implementation.
4. **Whitespace and layout.** Spaces, tabs, newlines, bidirectional controls, and join controls can
   delimit fields or change meaning. Only explicitly allowed transformations should produce derived
   views; unexplained collapsing is data loss.
5. **Orthography and phonology.** Similar spellings or sounds can suggest alternate readings. They
   are evidence channels, not guarantees of semantic equivalence or speaker intent.
6. **Segmentation and tokenization.** Word boundaries, subword boundaries, and alternate phrase
   segmentations can expose different hypotheses. A tokenizer's decomposition is neither the raw
   source nor an authoritative semantic parse.
7. **Latent representations.** Hidden-state patches or learned representation transforms may test
   invariance inside a model. They require model-specific causal validation and cannot be treated as
   transparent explanations of intent.

Keeping these layers explicit prevents a normalized string, token sequence, or latent state from
silently replacing the evidence it was derived from.

## Current application-layer contract

The implemented lattice keeps one immutable source document and bounded candidate records. Each
candidate binds an exact source span to candidate text, a channel, component evidence, confidence,
generation reason, and provenance. Content identifiers cover the complete validated candidate;
source digests cover both source ID and complete source text. Downstream assessment records retain
enough information to reconstruct the exact assessed prompt from the source document and one bound
span substitution.

Conservative normalization removes only the two known artifacts U+200B and U+FEFF, then applies
NFC and CRLF/CR-to-LF normalization where necessary. U+200C, U+200D, ordinary spaces, tabs,
punctuation, numbers, and negation are preserved. Orthographic candidates use bounded character
distance. Phonological candidates require an explicit injected lexicon. Context is only bounded
token-overlap evidence; it is not an intent oracle.

Search and output caps are epistemic boundaries. Exhaustive bounded search with no candidate above
the evidence floor produces `NO_REPAIR`; truncated search produces `UNKNOWN`. Neither outcome says
that no possible alternate representation exists. Candidate top-k is a ranked hypothesis set, not
automatic text correction.

## Possible foundational-model research

The following are research directions, not repository features:

- **Byte- and character-aware access:** expose recoverable lower-level input evidence alongside
  ordinary tokens so encoding or tokenizer artifacts can be evaluated without discarding source
  bytes.
- **Grapheme-aware modeling:** represent Unicode grapheme boundaries and script-specific joining
  behavior explicitly, with conformance and multilingual evaluation.
- **Whitespace-sensitive pathways:** preserve layout and separator information while learning when
  alternate layout views are meaning-preserving.
- **Phonological pathways:** add language- and dialect-aware phoneme or acoustic evidence with
  uncertainty, provenance, and clean controls. This would require real speech evaluation; the
  current lexicon is not speech recognition.
- **Alternate segmentation lattices:** retain several token or phrase segmentations and assess them
  through a shared semantic path rather than choosing one as truth before evaluation.
- **Latent patches:** causally test whether bounded internal interventions restore stable behavior
  across representation variants. Correlation alone would remain diagnostic.
- **Adaptive retokenization:** allow a separately versioned model/harness to propose new
  tokenizations between evaluation episodes, then require held-out and protected regression tests.
  It must not mutate the tokenizer during the episode being scored.
- **Invariance training:** optimize consistency across independently labeled meaning-preserving
  variants while retaining contrastive separation for meaning-changing controls. Correctness and
  consistency remain separate outcomes.

Any such work would need a threat model, immutable source capture, exact transformation records,
bounded compute, independent ground truth, model and tokenizer versioning, clean negative controls,
and causal or behavioral validation appropriate to the claim.

## Research traceability

The repository traceability registry deliberately separates published findings from local design
inference:

- [REF-LEXICAL-PERTURB](recommendations/RESEARCH_TRACEABILITY.md#ref-lexical-perturb) reports
  accuracy degradation under keyboard noise and character swaps in the evaluated models. The local
  inference is to evaluate representation perturbations and measure correctness separately from
  consistency, not to endorse a particular repair algorithm.
- [REF-TOKENIZER-BETRAYAL](recommendations/RESEARCH_TRACEABILITY.md#ref-tokenizer-betrayal) reports
  failures associated with non-unique token encodings and tokenizer-induced phantom edits. The
  local inference is to test representation robustness while keeping tokenizer remediation outside
  the current application-layer scope.

Both references are attached to
[REC-005](recommendations/UNIFIED_RECOMMENDATIONS.md#rec-005--case--robustness-stripe--repeat-evaluation),
the case-by-stripe-by-repeat evaluation recommendation. Neither source establishes the full V5
architecture, validates the present heuristic scores, or proves that a generated candidate matches
a user's intended meaning.

For memory, [REF-EDGEMEM](recommendations/RESEARCH_TRACEABILITY.md#ref-edgemem) and
[REF-GRAPHMEM](recommendations/RESEARCH_TRACEABILITY.md#ref-graphmem) motivate evidence-preserving
and typed memory boundaries. The repository inference is limited: a stored derived representation
must keep source and trust provenance and must not become authority merely through retrieval.

## Evaluation boundary and limitations

Candidate recall is measured only against independently authored expected candidate text. A
meaning-changing candidate can have strong surface evidence and still be wrong. Clean controls,
false-repair rate, abstention precision and coverage, disagreement rate, clean-policy regression,
and laundering detection therefore accompany recall. The elapsed metric in the current evaluator
covers validation and aggregation only, not candidate generation or model inference.

The present module does not preserve original bytes, implement full grapheme segmentation,
generate a dedicated alternate-segmentation or contextual channel, learn phonology, access latent
states, adapt tokenization, train invariances, or implement a tokenizer or foundational model. It
is an auditable pre-semantic robustness scaffold whose candidates remain hypotheses.
