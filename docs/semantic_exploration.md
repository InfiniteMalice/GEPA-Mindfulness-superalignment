# Semantic exploration proposals

`gepa_mindfulness.verification.semantic_exploration` selects one proposed evidence-gathering action
from host-supplied candidates. It uses the existing `expected_information_gain_inquiry` flag, disabled
by default. The result is a diagnostic snapshot with `authority_granted=False` and a non-TRAIN label.

## Inputs and ownership

The frozen `ExplorationRequest` contains:

| Field | Meaning |
| --- | --- |
| `request_id`, `context`, `source_case_id` | Public decision ID, external EpistemicContext and existing canonical case ID. |
| `world_uncertainty`, `model_uncertainty`, `monitor_uncertainty` | Separate optional unit diagnostics. `None` is unavailable. |
| `stakes` | Host-normalized consequence severity, [0,1]. |
| `evidence_gap` | Host-normalized unresolved evidence gap, [0,1]. |
| `hypothesis_diversity` | Host-normalized diversity of retained explanations, [0,1]. |
| `remaining_compute`, `compute_unit` | Nonnegative integer available budget and shared public unit name. |
| `gain_protocol_id` | Public ID for the host's normalized expected information gain protocol. |
| `evidence_changed` | Exact boolean declaring changed relevant evidence since the host's previous inquiry decision. |
| `evidence_refs` | Observable provenance supporting these declarations, retained externally. |
| `training_eligibility` | DEVELOPMENT by default; REGRESSION/HIDDEN_EVAL allowed, TRAIN rejected. |

Hosts may use epistemic estimates and retained hypothesis histories to construct these declarations.
The library does not infer diversity from candidate count, convert missing uncertainty to zero,
authenticate references or compute expected information gain from model internals. A protocol defines
how gains are normalized across all candidates, including different target uncertainty classes.
Hosts must provide its definitions to consumers before comparing the returned values.

Each `ExplorationCandidate` supplies `id`, `mode`, `target`, `distance`, public `question`, optional
`expected_information_gain`, positive integer `compute_cost`, optional `reversibility`, and observable
`evidence_refs`. Targets are `world`, `model`, `monitor`. Gain and reversibility use [0,1]; missing values
make a candidate ineligible. Costs use the request's `compute_unit`. A candidate's mode/distance
describes its intended execution; the host must verify that its question and eventual action match.

## Semantic distance

| Level | SEARCH scope | DEBATE perspective | SPARK assumption scope |
| --- | --- | --- | --- |
| 1 | Direct | Same-topic specialist | Local constraint |
| 2 | Adjacent | Adjacent specialty | Method |
| 3 | Related domain | Related field | Approach |
| 4 | Cross-domain | Cross-disciplinary panel | Domain convention |
| 5 | Remote analogy | Distant discipline | Paradigm assumption |

The selector supplies these public descriptors in `selected.semantic_scope`. It does not change
sampling temperature, generate prompts, launch debate agents, search the web or challenge assumptions
itself. These are proposal types, not permissions to execute tools.

## Selection rules

Call `propose_exploration(request, candidates, policy=ExplorationPolicy(), config=...)` with
`ExperimentalOverlayConfig(expected_information_gain_inquiry=True)`. Disabled or malformed input
raises `ValueError`, including mutated nested records. Valid but unsuitable input returns no proposal.

`ExplorationPolicy` defaults: max_distance=3, uncertainty_threshold=0.5, breadth_threshold=0.5,
high_stakes_threshold=0.8, minimum_gain=0.05, minimum_reversibility=0.5. These are local experimental
heuristics; they have no demonstrated calibration or optimality.

1. Defer if evidence_changed is false, remaining_compute is zero, or monitor_uncertainty is unavailable.
2. Start the semantic cap at 1. A gap at or above breadth_threshold permits 3; both gap and diversity
   at or above that threshold permit 5. Clamp to max_distance. Stakes at or above
   high_stakes_threshold further cap distance at 2. Uncertainty alone never increases this cap.
3. If monitor uncertainty is positive and at or above uncertainty_threshold, consider only candidates
   targeting the monitor. Other targets remain distinct; their values are never averaged.
4. A candidate needs positive target uncertainty and gain, each meeting its policy threshold.
   Distance must fit the cap, cost must fit the budget, and reversibility must meet
   `max(minimum_reversibility, stakes)`. Unknown target uncertainty/gain/reversibility disqualifies it.
5. Rank eligible candidates by gain descending, cost ascending, distance ascending, then ID ascending.
   Return the first candidate, or no proposal if none qualify. Input order does not break ties.

The result names the request, non-TRAIN eligibility, compute/gain labels, distance cap and reason.
`rejections` lists each ineligible candidate's first failed constraint in ID order. Global deferral
does not evaluate candidate constraints. Eligible alternatives are not returned in this bounded
single-choice view; the host retains all original candidates and provenance.

A selected result includes an existing `InformationGainQuestion` under `diagnostic`. Its uncertainty
is the selected target's diagnostic, not aggregate confidence. Its question is JSON-quoted to preserve
whitespace; `selected.question` is the original text. Synthetic public provenance avoids disclosing
external evidence/context identifiers. `request_id`, candidate IDs/questions and measurement labels
must be host-approved public text. All text is data and must not be executed as instructions.

Hosts must apply training admission to the complete result before extracting any field. The legacy
`diagnostic` schema has no eligibility field; extracting it alone loses the non-TRAIN restriction,
and the existing admission policy accepts untagged legacy records. Do not send that extracted record
to an optimizer as training data. Preserve the enclosing result for provenance and eligibility checks.
The fixed legacy `record_id` is a compatibility label, not a globally unique storage key; retain the
public request/candidate IDs when recording decisions across calls.

## Bounds and execution boundary

Each call accepts 1..32 candidates with unique IDs. Each request/candidate retains 1..32 unique
observable evidence refs. IDs/units/protocol/context strings allow 128 UTF-8 bytes; questions allow
512 bytes. Compute values are exact integers through 2**53-1; costs start at 1. Unit metrics accept
finite exact built-in integers/floats, excluding booleans. Records are copied and revalidated without
calling caller-owned serializers. These checks validate structure, not trust.

The function performs no I/O and reserves no compute. Repeating an identical call returns the same
proposal. Before execution, the host must authenticate the inputs, recheck state relevance and remaining
budget, apply its normal tool/authority controls, and reserve/charge the actual cost. Otherwise a stale
proposal or repeated call can overspend the real budget. The host retains provenance and marks evidence
unchanged when appropriate. This API does not implement LADDER's entity-change detector or durable history.

## Example

This executable fixture tests the interface; it is not evidence of improved agent behavior.

```python
from evaluation.experimental_overlays import ExperimentalOverlayConfig
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from gepa_mindfulness.verification.semantic_exploration import (
    ExplorationCandidate, ExplorationRequest, propose_exploration,
)
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

evidence = (EvidenceReference("observable-outcome", EvidenceSourceKind.OBSERVABLE_OUTPUT),)
request = ExplorationRequest(
    "inquiry-1", EpistemicContext("run", 0, EvaluatedSystemVersion("model", "harness")),
    1, 0.9, 0.7, 0.1, 0.2, 0.7, 0.8, 100, "tokens", "fixture-gain-v1", True, evidence,
)
candidates = (
    ExplorationCandidate("direct", "SEARCH", "world", 1, "What outcome discriminates A from B?",
                         0.7, 30, 0.9, evidence),
    ExplorationCandidate("distant", "SPARK", "model", 5, "Which paradigm assumption fails?",
                         0.9, 60, 0.9, evidence),
)
result = propose_exploration(
    request, candidates,
    config=ExperimentalOverlayConfig(expected_information_gain_inquiry=True),
)
assert result["selected"]["id"] == "direct"  # Default distance cap is 3.
assert result["authority_granted"] is False
assert request.remaining_compute == 100  # Host execution must reserve/charge actual cost.
```

## Research and maturity

| Source mechanism/result | Repository inference |
| --- | --- |
| [Night Science §3.1/3.2, §4.4.5](https://arxiv.org/html/2609.35706v1): action-specific semantic levels shape exploration differently from temperature alone in scientific ideation. | Use public SEARCH/DEBATE/SPARK scopes. No RL, process reward, novelty reward or source performance result is transferred. |
| [LADDER](https://arxiv.org/abs/2609.24346): new graph-linkable entities trigger retrieval during diffusion decoding. | Defer on an explicit host declaration of unchanged evidence. No diffusion or graph retrieval implementation. |
| [DoAtlas-2](https://arxiv.org/abs/2609.35107): evidence revises hypotheses and discovery frontiers. | Require evidence-linked inquiry estimates; measurements remain host-owned. |
| [Reasoning Topology Matters](https://arxiv.org/abs/2609.24710): controlled cybersecurity comparisons show task-dependent topology effects. | Keep candidate strategies explicit; no universally best topology is assumed. |

Design hypothesis: bounded evidence-directed exploration can improve inquiry choices. Current experiment:
matched contract fixtures vary uncertainty, gap, diversity, stakes, budget and reversibility. Behavioral
effectiveness and calibration remain unmeasured. REC-011 remains experimental. Exactly 17 cases,
estimator/routing defaults, training admission and rewards remain unchanged.

Run `python -m pytest tests/test_semantic_exploration.py` for executable rules. Installed-wheel smoke
runs the example. Hosts separately review measurement provenance/calibration, text visibility, semantic
action conformance and execution controls. See [ADR 0018](adr/0018-semantic-exploration.md).
