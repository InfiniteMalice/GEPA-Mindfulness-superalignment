# Skill bank and failure localization

PR-8 adds an opt-in diagnostic adapter and immutable skill metadata. The adapter
annotates the existing `FailureGraph`; the existing `SkillLifecycleStore` remains
responsible for durable artifacts, execution receipts and held-out validation.
Importing either module changes no runtime route, reward, recovery policy or case.

## Localize a recorded failure

```python
from gepa_mindfulness.verification.failure_layers import (
    FailureLayer, LayerClaim, localize_failure_layers,
)

# events is a tuple ending at the selected reconciliation. graph is a FailureGraph.
report = localize_failure_layers(events, graph, "reconciliation-event", enabled=True)
audit_payload = report.to_dict()
```

`localize_failure_layers()` validates the full action-bound sequence, numeric PEO
residuals, chronology and evaluation-unit identity. Each graph node must reference
an event in the selected reconciliation's ancestry, use that event's timestamp and
cite its recorded observable evidence. Typed evidence retains its source kind.
For legacy events with untyped reference IDs, the host remains responsible for the
node's evidence-kind declaration. The host must also authenticate event producers,
verifier identities and source contents; matching references cannot establish truth.

The report is a historical snapshot ending at the supplied reconciliation. It does
not claim to include later observations or superseding verifications. The host must
retain the full input window and graph alongside the report's SHA256 digests.

The adapter derives hypotheses from negative typed verifier findings only when
the finding has observable evidence bound to the graph node. It preserves caller
claims as `reported_claim`, never as verified diagnoses. A producer-declared model
mismatch in the aggregate update or any innovation produces a world-model
hypothesis on a node referencing the reconciliation event. `unassessed` produces
no such hypothesis. Nonzero residuals and high uncertainty alone prove no layer.
The report retains residuals and all three uncertainty dimensions, including `None`.

| Layer | Evidence needed for review | Proposed repair surface |
|---|---|---|
| Routing | A required, adequate skill existed but was not selected | Selection and routing description |
| Knowledge/skill | The appropriate skill was selected but its guidance was inadequate | Guidance and replay coverage |
| Execution | Arguments, schema, execution or intended operation failed | Executor and runtime |
| World-model | Recorded mismatch suggests an inadequate prediction model or changed environment | Reconciliation and environment checks |
| Evidence | Contradictory sources or lost provenance | Retrieval and source reconciliation |
| Calibration | Independent outcome comparison shows confidence is miscalibrated | Held-out calibration evaluation |
| Verifier/monitor | Independent evidence identifies incorrect or missing monitor coverage | Monitor and verifier audit |
| Reporting | A claim exceeds the available outcome evidence | Correct the report and disclose limits |
| Authority | The relevant action permission or scope check failed | Stop and review authority |

Routing, knowledge/skill, calibration and verifier/monitor require explicit
`LayerClaim` inputs from a host review process. For example:

```python
claim = LayerClaim("failure-id", FailureLayer.KNOWLEDGE_SKILL, observable_refs)
report = localize_failure_layers(
    events, graph, "reconciliation-event", (claim,), enabled=True,
)
```

The adapter validates the claim's evidence linkage, not the review's conclusion.
It retains simultaneous hypotheses; nodes with no layer remain unlocalized.
`task_fit=False`, unmet dependencies and repeated-route flags do not by themselves
distinguish poor skill selection from execution or environment failure. The adapter
does not force those findings into a layer. No annotation changes graph edges,
root-cause status, recovery counters or executor permissions.

## Separate WHEN from HOW

```python
from gepa_mindfulness.skill_bank import SkillBank, SkillCard

card = SkillCard.from_history(
    history,
    routing_description="When observable sources disagree.",
    operational_guidance="Preserve both sources and obtain independent verification.",
    foundational=True,
)
bank = SkillBank((card,))
```

`history` is a canonical `SkillLifecycleHistory`. The card records its current skill
ID, version, artifact ID and canonical artifact digest. `card.matches_history(history)`
checks that binding again; a lifecycle transition makes the old card stale. A card may
describe a draft artifact. The binding does not certify its text or make it deployable.

Hosts own bank configuration and the `foundational` classification. Agent-generated
cards or complete replacement banks must not be installed as trusted configuration.
`SkillBank.propose_for_layer()` maps routing to a routing-description review and
knowledge/skill to an operational-guidance review. It rejects the other seven layers
as skill-text changes. A review reference identifies supporting review material; the
method does not fetch or authenticate it. `propose()` also supports retirement requests.

Every request to change or retire a foundational card returns `blocked=True` and
`foundational_norm_requires_human_governance`. Performance evidence cannot override
this result. Even unblocked operational proposals only return data: the bank and
lifecycle remain unchanged. There is no proposal application API or automatic training.
Host review, independent replay/held-out checks and the existing durable lifecycle
remain necessary before persistence. A proposal is neither an approval nor a receipt.

Suggested host-curated skills include EpistemicAbstention, EvidenceConflictResolution,
EpistemicReconciliation, AuthorizationCheck, FailureTransparency, ProvenancePreservation
and HumanEscalation. These are design examples, not installed policies or certified skills.
Hosts should mark normative obligations as foundational regardless of measured task success.

## Research transfer and limits

- [SEEK](https://arxiv.org/abs/2609.29803) separates routing descriptions from operational
  guidance and distinguishes routing, knowledge and execution errors. This implementation
  adopts those distinctions in metadata and review proposals; it does not reproduce the
  paper's industrial evaluator or skill update system.
- [ARISE](https://arxiv.org/abs/2609.35532) evolves rubric-skill pairs from rollout evidence,
  adapts task sampling and retires consistently satisfied criteria. Here, gap evidence
  can motivate a proposal, while foundational norms remain protected. No adaptive reward,
  rubric retirement or reinforcement-learning algorithm is installed.
- [PINNForge](https://arxiv.org/abs/2609.23023) uses execution feedback to guide changes
  in physics-informed neural-network design. Layer-specific repair review is a repository
  hypothesis; these software diagnostics do not reproduce PDE experiments.
- [Failure-Transparent Agents](https://arxiv.org/abs/2609.35732) separates failed execution
  from unsupported post-failure reporting. The adapter keeps those hypotheses distinct.
- [Qwen-Planner-Agent](https://arxiv.org/abs/2609.29892) describes trajectory diagnosis and
  reviewed, versioned updates to training data and the harness. This stage retains the
  review boundary and adds no online model or harness mutation.

Before PR-8, graphs expressed causal roles and artifacts tracked lifecycle states, but
neither interface exposed this nine-layer sidecar or separate WHEN/HOW metadata.
The expected benefit is more specific repair review. Contract tests establish scoping,
norm protection and deterministic behavior; they do not establish diagnosis accuracy,
calibration quality or improved task performance on real agents.

Validation: `python -m pytest -q tests/test_failure_layers.py tests/test_skill_bank.py`
covers all nine reported layers, typed execution/authority/evidence/reporting findings,
aggregate and per-binding mismatch, ambiguity, unknowns, private-evidence rejection,
bad ancestry, stale artifact bindings, and non-mutating proposals. Existing graph,
lifecycle, reconciliation and 17-case tests cover compatibility.
