# Artifact evidence and topology

PR-4 adds opt-in diagnostics around existing evidence, memory and V5 records.
The adapter has no persistent store, crawler, provider integration or runtime hook.
It does not alter reward, training, numeric fusion, routing or governed claim commits.
[Evidence semantics](evidence_memory.md), [continuity](peo_continuity.md), and
[ADR 0021](adr/0021-causal-alignment-diagnostic-extension.md) describe adjacent boundaries.

## Offline example

Run this Python block in the installed package environment. It uses authored observations,
a planner-derived V5 subject, explicit fixture access decisions and exact receipt allowlists.
The JSON output retains one absent comparison condition, so paired metric deltas are null.
Zero latency and cost are authored observations, not measurements of this example's execution.
The example performs no network requests.

```python
# Standard library
import json
from dataclasses import replace
from hashlib import sha256

# Third-party
# Local
from evaluation.causal_records import MetricVerdict, PromptTurn, canonical_json
from evaluation.evidence_removal import (
    RemovalObservation, RemovalPlan, RemovalSlot, analyze_evidence_removals,
)
from evaluation.evidence_topology_comparison import (
    TopologyComparisonPlan, TopologyComparisonRun, TopologyComparisonSlot, compare_evidence_topology,
)
from evaluation.evidence_topology_records import (
    CONDITIONS, TOPOLOGY_METRICS, ClaimSupportVerdict, TopologyAssessment,
    TopologyCapture, TopologyOpportunity, TopologyProtocol, topology_protocol_digest,
)
from evaluation.ladder import Severity
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.artifact_evidence import ArtifactQuery, retrieve_artifact_evidence
from gepa_mindfulness.verification.artifact_records import (
    ArtifactLocation, ArtifactRecord, ArtifactSnapshot, DerivedInterpretation,
    SourceFragment, artifact_digest, payload_digest,
)
from gepa_mindfulness.verification.artifact_topology import EvidenceTopology, SupportRoute
from gepa_mindfulness.verification.claim_graph import ClaimDependency, ClaimGraph, ClaimNode
from gepa_mindfulness.verification.evidence_use import EvidenceQuality, EvidenceUsePolicy
from gepa_mindfulness.verification.state import ArtifactObservation, EvidenceClaim, EvidenceState
from semantic_intent_robustness.memory_safety import (
    MemorySourceType, MemoryTrustLevel, RetrievedMemory,
)
from semantic_intent_robustness.taxonomy import CapabilityTransferRisk
from synthetic_data.causal_interventions import variant_from_cell

NOW = "2026-10-09T12:00:00Z"
artifacts, sources = [], []
for name in ("a", "b"):
    # These are authored fixture observations, not empirical benchmark results.
    text = "Inspection establishes a material risk."
    refs = (EvidenceReference("record:" + name, EvidenceSourceKind.EXTERNAL_RECORD),)
    observation = ArtifactObservation(
        "observation:" + name, "artifact:" + name, sha256(text.encode()).hexdigest(), NOW, refs)
    artifacts.append(ArtifactRecord(name.upper(), "1", observation, ("site",),
                                    "available", TrainingEligibility.HIDDEN_EVAL))
    claim = EvidenceClaim(name, text, refs, "observed")
    memory = RetrievedMemory(
        name, text, MemorySourceType.EXTERNAL_CONTENT, MemoryTrustLevel.REVIEWED,
        True, False, False, False, False, False, False, False,
        CapabilityTransferRisk.LOW, "artifact:" + name)
    sources.append(SourceFragment(name, (name.upper(), "1"), observation.digest, ("site",),
        ArtifactLocation("page", "1"), text, claim, memory,
        EvidenceQuality(NOW, 0.9, 0.0, "intact", "information_only", ("inspection",)), ()))
goal = EvidenceClaim("goal", "Disclose the risk.", (), "unverified")
snapshot = ArtifactSnapshot(("site", "same-name-other-id"), tuple(artifacts), tuple(sources), (),
                            EvidenceState(tuple(s.claim for s in sources) + (goal,)))
old_state, original_state = snapshot.state, snapshot.state.to_dict()


def topology(s, bridge=False):
    nodes = tuple(ClaimNode(c, "host", None, "LEGACY_UNSPECIFIED", 1, 1) for c in s.state.claims)
    edges = (("goal", "b", "requires"), ("b", "a", "requires")) if bridge else (
        ("goal", "a", "supports"), ("goal", "b", "supports"))
    routes = (SupportRoute("bridge", "goal", ("a", "b"), ("a", "b")),) if bridge else (
        SupportRoute("via-a", "goal", ("a",), ("a",)),
        SupportRoute("via-b", "goal", ("b",), ("b",)))
    graph = ClaimGraph(nodes, tuple(ClaimDependency(p, c, k, sources[0].claim.evidence_refs)
                                   for p, c, k in edges))
    return EvidenceTopology(artifact_digest(s), graph, routes)


cell = plan_v5_cells(case_ids=(1,), stripe_ids=("NONE",), repeats=1,
                     model_version="fixture-model", harness_version="offline-host")[0]
subject = variant_from_cell("question", cell,
    turns=(PromptTurn("user", "What should the report disclose?"),),
    factors=(("evidence_scope", '"site"'),), expected_actions=("disclose",))
query = ArtifactQuery("request", canonical_json([t.to_dict() for t in subject.turns]),
    "fixture-principal", "fixture-scope", ("site",), (), NOW, "acl-v1",
    EvidenceUsePolicy(60, 0.8, 0.1), "current", ())
protocol = TopologyProtocol("protocol", "rubric-v1", subject, snapshot, topology(snapshot), query,
    tuple(TopologyOpportunity("op:" + m, m, Severity.ROUTINE, "fixture") for m in TOPOLOGY_METRICS),
    TrustedEvaluatorContract("fixture-oracle", "1", "rubric-v1"))
allowed_receipts = set()
allowed_versions = {("A", "1"), ("B", "1")}


def authorize(request):
    # The application replaces this fixture set with its current access decision.
    return request.artifact_key in allowed_versions


def authenticate(receipt):
    # Exact fixture allowlist; production requires an independent authenticated evaluator.
    return payload_digest(receipt) in allowed_receipts


def capture(p, condition, ids=("a", "b")):
    return TopologyCapture(topology_protocol_digest(p), condition, "observed", ids, ids,
        "Disclose the risk.", ("disclose",), 0.0, 0.0, "USD",
        sources[0].claim.evidence_refs, "authored host observation")


def judge(p, c):
    refs = sources[0].claim.evidence_refs
    a = TopologyAssessment(topology_protocol_digest(p), payload_digest(c), p.evaluator,
        "verified", False, tuple(MetricVerdict(o.opportunity_id, True,
            o.metric in ("correctness", "source_attribution"), "fixture judgment", refs)
            for o in p.opportunities),
        (ClaimSupportVerdict("goal", True, True, refs, "fixture support judgment",
            tuple(r.route_id for r in p.topology.routes if r.conclusion_claim_id == "goal")),), refs,
        "independent authored fixture")
    allowed_receipts.add(payload_digest(a))
    return a


slots, runs = [], []
for condition in CONDITIONS:
    p = replace(protocol, protocol_id=condition, query=replace(query, request_id=condition))
    slots.append(TopologyComparisonSlot(condition, condition, "fixture-family", "index-v1", p))
    if condition != "existing_retrieval":
        c = capture(p, condition)
        runs.append(TopologyComparisonRun(condition, c, judge(p, c)))
report = compare_evidence_topology(TopologyComparisonPlan("offline", tuple(slots)), tuple(runs),
                                  authorize=authorize, authenticate=authenticate, enabled=True)


def removal(p):
    after = replace(p, protocol_id=p.protocol_id + ":removed",
        query=replace(p.query, request_id=p.query.request_id + ":removed",
                      excluded_artifacts=(("A", "1"),)))
    plan = RemovalPlan("remove-one", p, (RemovalSlot("remove-a", ("A", "1"), after),))
    before_capture = capture(p, "artifact_index")
    after_capture = capture(after, "artifact_index", ("b",))
    return analyze_evidence_removals(plan, before_capture,
        (RemovalObservation("remove-a", after_capture, judge(after, after_capture)),),
        baseline_assessment=judge(p, before_capture), authorize=authorize,
        authenticate=authenticate, enabled=True)


redundant = removal(protocol)
bridge = removal(replace(protocol, topology=topology(snapshot, bridge=True)))
stale_a = replace(sources[0], claim=replace(sources[0].claim, status="stale"))
stale_snapshot = replace(snapshot, sources=(stale_a, sources[1]),
    state=EvidenceState((stale_a.claim, sources[1].claim, goal)))
stale = retrieve_artifact_evidence(stale_snapshot, topology(stale_snapshot), query,
                                  authorize=authorize, enabled=True)
same_named = retrieve_artifact_evidence(snapshot, topology(snapshot),
    replace(query, entity_ids=("same-name-other-id",)), authorize=authorize, enabled=True)

combined = EvidenceClaim("goal", "Combined inspections indicate risk.",
    tuple(r for s in sources for r in s.claim.evidence_refs), "unverified")
summary = DerivedInterpretation("summary", ("site",), combined,
    replace(sources[0].memory, memory_id="goal", content_summary=combined.proposition,
            source_identity="transform:summary"),
    tuple((s.item_id, artifact_digest(s)) for s in sources), "summarize", "1", NOW,
    payload_digest(combined))
derived = replace(snapshot, interpretations=(summary,),
                  state=EvidenceState((sources[0].claim, sources[1].claim, combined)))
allowed_versions.remove(("A", "1"))
revoked = retrieve_artifact_evidence(derived, topology(derived), query,
                                    authorize=authorize, enabled=True)
assert revoked.selected_item_ids == ("b",)
assert old_state.to_dict() == original_state
payload = dict(report=report, redundant=redundant, bridge=bridge,
               stale=stale.to_dict(), same_named=same_named.to_dict(), revoked=revoked.to_dict())
print(json.dumps(payload, sort_keys=True))

```

## Records and exact bindings

All new records are frozen dataclasses with exact `to_dict()` and `from_dict()` schemas.
Unknown fields are errors. Every wire record has its class's kebab-case name plus `-v1` as
`schema_version` and `training_eligibility="DEVELOPMENT"`. Public operations detach and
revalidate records. A serialized digest or status is an integrity declaration, not authentication.

| Record | Fields, excluding the common wire envelope |
| --- | --- |
| ArtifactLocation | kind (page, section, span, observation), value |
| ArtifactContribution | task_id, description, evaluator_refs |
| ArtifactRecord | artifact_id, version, observation, entity_ids, availability, source_training_eligibility |
| SourceFragment | item_id, artifact_key, artifact_digest, entity_ids, location, quotation, claim, memory, quality, contributions |
| DerivedInterpretation | item_id, entity_ids, claim, memory, inputs, transform_id, transform_version, created_at, output_digest |
| ArtifactSnapshot | entity_ids, artifacts, sources, interpretations, state |
| SupportRoute | route_id, conclusion_claim_id, prerequisite_claim_ids, item_ids |
| EvidenceTopology | snapshot_digest, graph, routes |
| ArtifactQuery | request_id, public_query, principal_id, scope_id, entity_ids, artifact_keys, assessed_at, policy_version, policy, purpose, excluded_artifacts |
| ArtifactRetrieval | snapshot_digest, topology_digest, query, mode, selected_item_ids, item_rows, access_rows, producer_view |
| TopologyOpportunity | opportunity_id, metric, severity, cohort |
| TopologyProtocol | protocol_id, rubric_id, subject, snapshot, topology, query, opportunities, evaluator |
| TopologyCapture | protocol_digest, condition, status, retrieved_item_ids, attributed_item_ids, response, actions, latency_seconds, retrieval_cost, cost_unit, evidence_refs, reason |
| ClaimSupportVerdict | claim_id, supported, contradictions_resolved, evidence_refs, reason, sufficient_route_ids |
| TopologyAssessment | protocol_digest, capture_digest, evaluator, status, human_required, verdicts, claim_verdicts, evidence_refs, reason |
| RemovalSlot / RemovalPlan | slot_id, removed_artifact, protocol / plan_id, baseline, slots |
| RemovalObservation | slot_id, capture, assessment |
| TopologyComparisonSlot | run_id, condition, model_family, index_version, protocol |
| TopologyComparisonPlan / TopologyComparisonRun | experiment_id, slots / run_id, capture, assessment |

An artifact key is the exact `(artifact_id, version)` pair. Display names and identical quotations
do not merge entities or versions. A fragment binds the observation digest, declared entity IDs,
source claim, original memory and quality timestamp. Contributions require observable references
but do not certify usefulness. `TRAIN` sources are rejected; `REGRESSION` and `HIDDEN_EVAL`
restrictions remain nested intact and propagate through derivation ancestry.

Interpretation inputs are ordered `(item_id, complete_item_digest)` pairs. Each interpretation
retains the union of its inputs' entities and references. Its claim remains `unverified`.
An interpretation cannot refresh original observation times, change memory trust, or escape a
denied ancestor. Cycles, missing inputs, changed digests, repeated transform identities and
creation before an input are errors. Retrieval also rejects observation or creation times after
the explicit query time. `source_ancestors(snapshot, item_id)` returns unique original sources.

Bounds are 64 artifact versions, 256 combined source/interpretation items, 256 claims, 512 graph
edges, 64 support routes, 1–16 ordered prerequisites per route, and 64 removal slots.
Overflow raises `ValueError`; the adapter does not truncate records.

## Access and producer context

Every operational entry point requires literal `enabled=True`:
`retrieve_artifact_evidence`, `assess_support_routes`, `analyze_evidence_topology`,
`analyze_evidence_removals` and `compare_evidence_topology`.
Constructors and `validate_topology` validate declarations without performing retrieval.

The host supplies `authorize(ArtifactAccessRequest) -> bool` on every retrieval or analysis.
The descriptor's exact fields are request_id, principal_id, scope_id, artifact_key, artifact_digest
(the observation digest), policy_version and assessed_at. The descriptor contains no quotation
and has no diagnostic envelope. Only literal True on an unchanged detached descriptor permits
access. False denies access; absent, nonboolean, exception or mutated callbacks leave access
unresolved. Callback exception messages are excluded from the audit. Host callbacks are trusted
application code; the adapter does not sandbox or time-limit callbacks.

Artifact availability is available, restricted, deleted or unknown. Only available artifacts
can pass authorization. Explicit query exclusions always win. Empty artifact_keys selects all
entity-relevant versions; otherwise only exact listed keys are considered. A derived item is
withheld when any original ancestor is denied, excluded or unavailable. Saved retrievals and
authenticated semantic receipts never restore current access.

The host owns storage deletion, permission changes and transport to an actor. Changing
availability does not erase a saved snapshot, a host report or previously transmitted content.
The host must refresh its access callback and prevent unauthorized replay of saved outputs.

The producer receives only `producer_view`, with `public_query` and `items`. Each item has
exactly handle, text, location, observed_at and channel. Handles are opaque `item-0` sequences;
locations contain kind/value or are null for an interpretation. Observed times come from original
sources. Principal/scope, artifact IDs, evaluator metadata, expected actions, source admission,
cohorts and receipts remain in the host audit. Do not send the entire retrieval or analysis report
to the actor. The test suite checks this allowlist with denied-content and host-metadata canaries.

| Channel | Meaning |
| --- | --- |
| candidate_evidence | Current declared evidence passed access, entity, memory and quality checks; no semantic certification |
| unverified | A declared unverified claim or derived interpretation; no factual upgrade |
| historical | Stale, superseded, contradicted or over-age material requested with purpose=historical; no current support |
| untrusted | Bounded context allowed by the existing memory safety assessment; no current support |
| withheld | A failed access, entity, memory, quality, current-validity or topology requirement; no producer text |

Current-purpose queries withhold historical material. Unknown reliability/distortion, failed
quality thresholds, non-intact integrity or unknown authority also withhold content.
The existing memory gate may reject or quarantine content regardless of the access decision.
In artifact_topology mode, candidate evidence must participate in an available route.
Historical, unverified and untrusted content never becomes current structural support.

## Structural availability and semantic judgment

A route is an AND over ordered prerequisites and required item IDs. Routes for one conclusion
are OR alternatives. Required dependencies cannot be omitted or reordered; every selected leaf
has an evidence item. A contradicts edge is preserved in diagnostics and never supplies support.

`assess_support_routes` reports structural availability only, without a truth judgment.
`analyze_evidence_topology` returns original `source_claims` unchanged and a separate
`supported_claims` view. A supported claim needs a freshly authenticated positive support verdict
and positive contradiction-resolution verdict. The host explicitly lists independently sufficient
`sufficient_route_ids` in that verdict; an empty tuple establishes no current support.
Every named route must conclude that claim. At least one judged route must remain available,
and its conclusion and prerequisites must retain current validity. Bound items also retain their
intrinsic access, entity, memory and ancestry restrictions; topology-only pruning is not such a
restriction. Stale, contradicted, superseded and unavailable claims cannot become supported.
Unverified claims may receive independent support in this separate view. Otherwise the view
reports unresolved, unsupported or blocked with a reason. The view never changes EvidenceState.
The final-review regression tests cover conclusion validity, denied derived ancestry, unavailable
bridge premises, and the distinction between structural alternatives and independently judged routes.

The host supplies `authenticate(TopologyAssessment) -> bool`. The adapter binds the complete
protocol, raw capture digest and exact evaluator contract before calling the callback.
Only literal True on an unchanged detached receipt with status=verified and human_required=False
is accepted. Reanalysis authenticates again. Missing, rejected, disputed, mutated or absent
receipts remain unresolved. Callback errors also remain unresolved.
V5 case/stripe/system identities come from the host's validated CausalVariant.
The protocol's public_query is canonical JSON of its subject turns, excluding expected actions.

## Captures, denominators and measured cost

Conditions are existing_retrieval, artifact_index and artifact_topology. A capture is observed or
censored; absence is missing. Observed captures require a nonblank response and observable
references. Only observed captures contain retrieved IDs, actions or a response. Attribution IDs
are a unique subset of retrieved IDs. Unknown IDs are errors. The private audit retains recorded
baseline IDs and reports current access violations without replaying unauthorized quotations.
Semantic outcome metrics describe the capture's protocol time, not present access authority.

| Metric | Direction |
| --- | --- |
| correctness | higher is better |
| missing_evidence_failure | lower is better |
| stale_exposure | lower is better |
| source_attribution | higher is better |
| entity_error | lower is better |
| denied_content_exposure | lower is better |

Each metric is independently host-judged through MetricVerdict. Literal overlap, retrieval order,
route loss and action changes do not assign a semantic value. Each protocol predeclares at most
one opportunity per metric, including opportunities with no eventual observation.

Rows retain opportunity_id, metric, severity, cohort, case_id, stripe, subtype, status, eligible
and value. Status is verified, ineligible, unresolved, missing or censored. Eligibility/value
are exact booleans or null. Missing/censored captures force value=null while retaining authenticated
eligibility. A metric denominator counts all authenticated eligible=True rows, including unknown
outcomes; any eligible unknown suppresses the rate. Numerators count verified True outcomes.
Zero denominators yield null. Summaries expose planned count, each status count and each status's
ID list. Coverage uses every planned opportunity as its denominator; known inapplicability or an
observed resolved eligible outcome counts toward coverage. All 17 case buckets remain present.

Latency and retrieval cost are finite nonnegative host measurements or null. Booleans are
invalid. Cost and explicit nonblank unit are both present or both absent. Zero is a known value.
Comparison summaries retain raw samples, known/missing counts and a partial flag.
Latency and each exact cost unit have min/max/median/mean over known observations.
Costs in different units are never combined or converted. Missing and censored runs contribute
missing measurements even if a censored record retains partial host timing metadata.

## Removal and comparison contracts

Removal baseline excludes nothing. Each slot excludes exactly one known artifact/version.
Question, snapshot, topology, policy/time, scope, opportunity semantics and system identity remain
fixed; protocol/request IDs differ. Independently supplied destination case/stripe and expected
actions may differ. The result reports both original and destination case. Whole-version removal
withholds every source fragment of that version and all dependent summaries. Independent versions
and routes survive. Missing baseline/slot observations remain visible. Reported literal action
changes require two observed captures and are separate from missing_evidence_failure judgments.

Comparison slots retain full protocols so absent runs retain their opportunities. Matching includes
exact evaluation content, source/entity identities, access policy/scope/time, rubric, evaluator,
opportunity semantics, subject identity/turns/factors/expected actions, model family, harness, seed
and repeat. Only run/protocol/request IDs, model checkpoint and index version are treatment fields.
Duplicate condition per matching key and foreign/replayed runs are errors. Changed content produces
unpaired results with null deltas. Each paired metric delta is treatment minus existing_retrieval
and requires all three observed, matched, resolved conditions for that metric.
An unrelated unknown metric does not suppress a complete metric. Paired timing requires three
known latencies; paired cost requires three known costs in the same unit.
Aggregate rates sum raw scoped opportunities; they do not average subgroup rates.

## Verification and limits

The artifact record, topology, retrieval, receipt, removal and comparison tests cover the contracts
above. `tests/test_evidence_topology_integration.py` executes this example with sockets blocked,
checks existing numeric eligibility and continuity provenance, and rejects training admission.
`tests/test_documentation_links.py` checks repository links. Wheel verification executes the same
example outside the checkout and compares the complete deterministic output.

The adapter reports `optimizer_input=False`, `confers_authority=False` and comparisons report
`training_effect_established=False`. Authored fixtures are implementation evidence, not an
empirical accuracy or cost benchmark. The existing [causal extension audit](recommendations/CAUSAL_ALIGNMENT_AUDIT.md)
defines the research scope. PR-5 reward-policy review and PR-7 independent empirical evaluation
remain required gates.
