# Experimental V5 Overlays

The V5 experimental overlay layer declares diagnostic proposals that a controlled experiment may
inspect. It does not add canonical cases, alter evaluation cells, change optimizer reward, grant
runtime authority, execute actions, or promote research proposals to stable behavior.

## Registry and opt-in boundary

`evaluation/cases/experimental_overlays.yaml` is the packaged registry. Every entry has
`maturity: experimental`, `enabled_by_default: false`, one unique feature flag, allowed diagnostic
outputs, prohibited effects, and links to the recommendation and research registries. The loader
rejects missing or unknown fields, duplicate YAML keys, reordered or unknown overlay identities,
cross-wired feature flags, unrecognized outputs, unresolved REC/REF identifiers, and any entry
that does not prohibit canonical-case creation and direct optimizer reward.

Experiments opt in with `ExperimentalOverlayConfig`. The zero-argument configuration enables
nothing. External mappings may omit flags, which remain false, but cannot contain unknown keys or
truthy non-boolean values. `enabled_overlays()` returns immutable declarations in registry order;
it has no integration with the case manifest, cell planner, reward calculator, runtime authority,
model, harness, skill lifecycle, or action executor.

```python
from evaluation.experimental_overlays import ExperimentalOverlayConfig, enabled_overlays

config = ExperimentalOverlayConfig(competing_hypotheses=True)
declarations = enabled_overlays(config)
```

Receiving a declaration means only that the caller opted into producing or inspecting its allowed
diagnostic record. It is not permission to act on a proposal.

## Declared overlays

| Overlay | May observe or propose | Cannot do |
| --- | --- | --- |
| `competing_hypotheses` | A set of at least two alternative hypotheses grounded in typed provenance. | Choose an action, create a case, or change reward. |
| `expected_information_gain_inquiry` | A question and finite nonnegative expected-information-gain estimate. | Execute the inquiry, claim an answer, or change reward. |
| `adaptive_small_multi_agent_topology` | A two-to-five-role directed topology proposal. | Spawn agents, delegate authority, execute work, or change reward. |
| `declarative_orchestration_scope` | A `global`, `focus`, or `local` scope declaration. | Enforce scope, grant capabilities, execute actions, or change reward. |
| `mechanistic_circuit_audit` | A provenance-bound audit observation and audit references. | Treat correlation as causation, modify a model, create a case, or change reward. |

The research links explain why each experiment may be worth evaluating. They do not establish the
overlay's conclusions or prove the repository architecture.

## Typed diagnostic records

`evaluation.experimental_records` supplies five frozen record types: `HypothesisSet`,
`InformationGainQuestion`, `TopologyProposal`, `OrchestrationScopeDeclaration`, and
`MechanisticAuditReference`. Each record contains:

- one `source_case_id` from the canonical 1–17 V5 manifest;
- a finite uncertainty value between zero and one;
- nonempty typed provenance with at least one observable `EvidenceReference`;
- the exact feature flag associated with its concrete record type;
- exact `experimental` maturity and `diagnostic` status.

`experimental_record_from_dict()` accepts a closed schema selected by `record_type`. Unknown fields
are rejected, including action-execution instructions, authority grants, additional canonical case
IDs, reward components, and optimizer-fitness values. A record can describe a proposal or audit
observation only. The module contains no function that performs the proposed inquiry, topology,
scope, action, model change, or reward update.

## Controlled experiment checklist

1. Select the minimum required flag in an explicit `ExperimentalOverlayConfig`.
2. Preserve the originating canonical case and observable evidence references in every record.
3. Store or display the record as diagnostic output only.
4. Evaluate any proposed behavior through separately authorized runtime and V5 evaluation paths.
5. Disable the flag after the experiment; the default configuration remains empty.

The registry, feature-flag, and record contracts are verified by
`tests/test_experimental_overlay_registry.py`, `tests/test_experimental_overlay_flags.py`, and
`tests/test_experimental_overlay_records.py`. Wheel-level tests verify that the registry, modules,
and this guide remain available outside a source checkout.
