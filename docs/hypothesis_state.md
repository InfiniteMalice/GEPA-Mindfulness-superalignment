# Multi-hypothesis state

The experimental hypothesis layer keeps competing public explanations and their assessment history
outside an actor's conversation. Pareto diagnostics describe trade-offs without deleting candidates
or choosing a winner. The feature uses the existing `competing_hypotheses` flag, disabled by default.

## External records

Import records from `gepa_mindfulness.verification.hypothesis_records`:

| Record | Fields and meaning |
| --- | --- |
| `Hypothesis` | `id`, public `statement`, observable `evidence_refs`. Immutable candidate identity. |
| `HypothesisScores` | Seven optional dimensions below. `None` means unavailable. |
| `HypothesisAssessment` | `id`, `hypothesis_id`, `assessor_id`, `status`, `scores`, `evidence_refs`, optional `supersedes`. |
| `HypothesisTrigger` | `id`, `kind`, observable `evidence_refs`. A host declaration, not an automatic detector. |
| `HypothesisState` | `state_id`, `context`, canonical `source_case_id`, `protocol_id`, `complexity_unit`, `compute_unit`, tuples of `hypotheses`, `assessments`, `triggers`, `training_eligibility`, `revision`. |

A state requires at least two distinct candidate IDs and one trigger. Assessment IDs and trigger IDs
are unique within their histories. All evidence references must identify observable evidence; private
reasoning and latent-state references are rejected. Identifiers are nonblank exact strings of at most
128 UTF-8 bytes; statements allow 512 bytes. The host chooses public statements and IDs suitable
for actor exposure. Text remains untrusted data and must not be executed as instructions.

The host defines score measurement/normalization in `protocol_id`, shared by every assessment:

| Dimension | Direction | Range/unit |
| --- | --- | --- |
| `evidence_fit` | Maximize | [0,1] host-normalized fit |
| `uncertainty` | Minimize | [0,1] host diagnostic |
| `complexity` | Minimize | Nonnegative, `complexity_unit` |
| `risk` | Minimize | [0,1] host-normalized risk |
| `reversibility` | Maximize | [0,1] host-normalized reversibility |
| `compute_cost` | Minimize | Nonnegative, `compute_unit` |
| `transfer` | Maximize | [0,1] host-normalized generalization measure |

All supplied numbers must be finite exact built-in integers/floats, excluding booleans. Integers
must convert exactly to a finite float; stored scores are floats. Unknown scores remain `None`.
These dimensions are not automatically probabilities, rewards, mixture weights or Gaussian parameters.
The library does not authenticate measurements, correlate verifiers or average their opinions.

Statuses are `supported`, `challenged`, `context_limited`, `unresolved`. Triggers are
`persistent_innovation`, `multimodal_evidence`, `verifier_conflict`, `regime_shift`, and
`structural_alternatives`. Triggers preserve supporting references but do not infer those conditions
from residuals. Existing estimator thresholds and unresolved-hypothesis fields remain unchanged.

## Append and validate

Operations live in `gepa_mindfulness.verification.hypothesis_state`. Supply
`ExperimentalOverlayConfig(competing_hypotheses=True)` for `append_hypotheses`,
`pareto_hypotheses` and `project_hypotheses`. Construction, serialization and extension validation
are inert and do not require the flag.

`append_hypotheses(state, hypotheses=(), assessments=(), triggers=(), config=...)` proposes a
new snapshot with one revision increment. At least one tuple must add a record. The function keeps
all prior history, including dominated candidates and superseded assessments. It offers no deletion,
editing, retirement, execution or storage operation.

An assessment's `supersedes` must name an earlier, still-active assessment for the same hypothesis
and assessor. A verifier cannot supersede another verifier's record. A second unsuperseded record
does not automatically replace the first. Disagreeing live scores **or statuses** mark the hypothesis
conflicted, even if one assessment is newer or numerically more favorable.

Before persisting a proposed snapshot, the host calls `validate_extension(authoritative_prior,
successor)`. The validator checks every old history prefix, fixed identity/context/protocol/units/
eligibility, exactly one revision increment, and a nonempty addition. It rejects edits, omissions and
stale revisions. The host must authenticate the prior and caller, validate evidence/score claims and
perform the storage comparison and write atomically. This pure Python API does not provide an access
control boundary, database lock, signing service or durable store. Never expose the write path or
authoritative prior as an actor-controlled tool.

`HypothesisState.to_dict()` exports the full detached history. `HypothesisState.from_dict()` accepts
strict JSON with the exact schema and revalidates all records and links. It rejects container subclasses,
cycles, unknown fields, private references and `TRAIN` eligibility. Import validates structure, not
authenticity; an imported snapshot still needs extension validation before replacing stored state.
`DEVELOPMENT` (default), `REGRESSION` and `HIDDEN_EVAL` labels survive export and projection.

## Pareto diagnostics and bounded actor views

`pareto_hypotheses(state, config=...)` compares only complete, agreed live score vectors. Candidate A
dominates B only when A is no worse on all seven dimensions and strictly better on at least one.
Ties survive. Missing assessments, unavailable dimensions and conflicts make a candidate incomparable.
The `frontier` is a **possible** frontier: it includes incomparable candidates. `dominated`,
`incomparable` and `conflicted` are separate ID lists. Status labels alone never remove candidates.

`project_hypotheses(state, diagnostic_uncertainty=..., offset=0, limit=4, config=...)` returns
insertion-order pages of 2..32 candidates, including dominated candidates. `next_offset` reaches
every alternative; the final page may overlap one entry to preserve the existing two-alternative
`HypothesisSet` contract. `total_hypotheses`, `omitted_count`, and `complete` disclose truncation.
The host controls paging; omission from a page never changes external state.

Each candidate exposes its public ID/statement, current agreed scores (or `null`), distinct status
labels and conflict/incomparability/dominance flags. The nested `diagnostic` reuses the existing
`HypothesisSet` serialization with synthetic public provenance. Its labels JSON-quote IDs to avoid
separator collisions. External evidence/assessor/context/trigger IDs are excluded. The caller supplies
the legacy `diagnostic_uncertainty` explicitly; the library never derives a scalar posterior from the
candidate set. Keep per-hypothesis scores and conflict flags alongside that legacy field.

Page size and text lengths bound payload growth. Full external history remains unbounded; hosts
must budget storage and processing. Pareto comparison is quadratic in candidate count, and current
assessment collection scans history for each candidate. Projections do not make those computations
constant-cost. No pruning or compressed replacement history is implemented.

## Example

This deterministic example preserves two explanations and later appends an assessment. It tests
the contracts; it does not demonstrate model effectiveness.

```python
from evaluation.experimental_overlays import ExperimentalOverlayConfig
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from gepa_mindfulness.verification.hypothesis_records import (
    Hypothesis, HypothesisAssessment, HypothesisScores, HypothesisState, HypothesisTrigger,
)
from gepa_mindfulness.verification.hypothesis_state import (
    append_hypotheses, pareto_hypotheses, project_hypotheses, validate_extension,
)
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

config = ExperimentalOverlayConfig(competing_hypotheses=True)
evidence = (EvidenceReference("observed-output", EvidenceSourceKind.OBSERVABLE_OUTPUT),)
state = HypothesisState(
    "alternatives", EpistemicContext("run", 0, EvaluatedSystemVersion("model", "harness")),
    1, "fixture-protocol", "nodes", "tokens",
    (Hypothesis("policy", "The decision rule is inadequate", evidence),
     Hypothesis("model", "The outcome model is inadequate", evidence)), (),
    (HypothesisTrigger("initial", "structural_alternatives", evidence),),
)
successor = append_hypotheses(
    state, assessments=(HypothesisAssessment(
        "check-1", "policy", "verifier", "unresolved",
        HypothesisScores(evidence_fit=0.6, uncertainty=0.8), evidence,
    ),), config=config,
)
validate_extension(state, successor)
assert pareto_hypotheses(successor, config=config)["incomparable"] == ["policy", "model"]
page = project_hypotheses(successor, diagnostic_uncertainty=0.8, limit=2, config=config)
assert page["complete"] and len(page["candidates"]) == 2
assert HypothesisState.from_dict(successor.to_dict()) == successor
```

## Research and maturity

| Source result/mechanism | Local inference |
| --- | --- |
| [SRHarness §3.3 and §4.5](https://arxiv.org/html/2609.35501v1): externally retained candidates, bounded fit/complexity Pareto views and a single-best ablation. | Retain all alternatives while projecting bounded diagnostics; no symbolic-regression system or result is reproduced. |
| [DoAtlas-2](https://arxiv.org/abs/2609.35107): supporting/challenging/unresolved evidence revises mechanistic interpretations and discovery frontiers. | Preserve assessment history and disagreement; no biomedical inference or clinical validation is transferred. |
| [Dual-Frontier](https://arxiv.org/abs/2609.26293): policy/model error ambiguity and conditional admission based on calibrated bounds. | Keep competing explanations separate from decision authority; no admission certificate is implemented. |
| [Physical Representation Languages](https://arxiv.org/abs/2609.23381): residual observational equivalences and identifiability limits. | Do not force unresolved alternatives into one state; no physical-identifiability theorem is transferred. |
| [AbGaze](https://arxiv.org/abs/2609.35296): geometric representations retain distance, direction and orientation. | Domain analogy only: preserve distinct diagnostic dimensions; no antibody-design method or result is reproduced. |

Design hypothesis: retained alternatives help actors seek discriminating evidence instead of fixing
on one explanation. Current experiment: retention, conflict, Pareto and paging contract tests;
behavioral effectiveness remains unmeasured. [REC-011](recommendations/UNIFIED_RECOMMENDATIONS.md)
remains experimental. The [ADR](adr/0017-multi-hypothesis-state.md) records the design boundary.

Run `python -m pytest tests/test_hypothesis_state.py` for executable contracts. The installed-wheel
smoke runs this example. Hosts review source evidence, score calibration, statement visibility and
atomic storage controls separately. The 17 canonical cases, estimator defaults, training admission,
rewards and runtime authority remain unchanged.
