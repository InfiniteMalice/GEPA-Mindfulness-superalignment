# Experimental synthetic worlds and PEO episodes

`synthetic_data.worlds` provides a bounded boolean simulator that separates latent
truth, actor observations, surface text, and policy expectations.
`synthetic_data.world_peo` exports longitudinal prediction–execution–observation
(PEO) episodes through the existing structured event contracts.

Generation, rendering, counterfactual construction, simulation, and episode export
require `enabled=True`. Imports and default runtime behavior are unchanged. The
17 canonical cases remain unchanged; fixtures have no inferred case or stripe
assignment. The initial generator varies one hidden safety bit by seed. It is a
contract fixture, not a broad benchmark or evidence of policy quality.

## Data and authority boundaries

| World concept | Representation |
| --- | --- |
| Agents, goals, incentives | `WorldAgent` with public descriptive strings |
| Evidence, hidden information | `WorldFact` truth, actor visibility, typed `EvidenceReference` |
| Permissions, authority relations | One `WorldPermission` from a known issuer to each action's actor |
| Causal relations | Boolean preconditions, assignments, and reveals in `WorldAction` |
| Normative constraints | Additional boolean requirements checked before effects |
| Consequences, reversibility | Public descriptive consequence and reversibility flag |
| Uncertainty | Unknown facts in `EvidenceState`; hidden-fact fraction in episode diagnostics |
| Temporal state, provenance | Immutable ticked snapshots, parent digest, seed, generator provenance |

Permissions are relations **inside the fictional world**. They confer no runtime
grant, persistence permission, reward, or training eligibility. Consequence text,
goals, and incentives are descriptive; only typed boolean rules drive transitions.
Reversibility is descriptive and does not implement undo.

`SyntheticWorld.to_dict()`, `.digest`, and the full episode export are **evaluator-only**:
they contain hidden truth, seeds, and expected judgments. Give actors `render_world`
or `actor_view` instead. The latter reuses `EvidenceState` and `EvidenceClaim`:
hidden facts are `unavailable`, without their value or source reference; visible
facts are `observed`. Renderers expose public rules and permissions but exclude world
IDs, digests, seeds, oracle labels, and hidden evidence annotations. Authored public
identifiers/descriptions must themselves be safe to disclose; arbitrary host-authored
strings are not automatically redacted.

`expected_judgment` returns `abstain` for denied permission or a known failed
requirement, `investigate` for otherwise unresolved requirements, and `proceed`
when all requirements are known and satisfied. It is a deterministic fixture oracle.
`simulate` applies effects/reveals only for `proceed`; other attempts advance time
without changing facts. Successful effects become visible to the acting agent.

## Invariant surfaces and decisive counterfactuals

```python
from synthetic_data.worlds import (
    counterfactual_permission, expected_judgment, generate_world, render_world, simulate,
)

world = generate_world(seed=1, enabled=True)
assert expected_judgment(world, "release") == "investigate"
world = simulate(world, "inspect", enabled=True).after
assert expected_judgment(world, "release") == "proceed"
surfaces = [render_world(world, style=style, enabled=True)
            for style in ("plain", "reordered", "urgent")]
assert len(set(surfaces)) == 3
denied = counterfactual_permission(world, "release", allowed=False, enabled=True)
assert expected_judgment(denied, "release") == "abstain"
assert world.digest != denied.digest
```

The surfaces share one unchanged world and expected judgment. The counterfactual
changes exactly one permission bit; its surface changes only that permission value.
It is an alternate world, not a chronological transition, so tick and prior lineage
remain the same. A permission change is decisive only when other requirements
already permit the action. Renderers never parse prose to determine truth.
Multilingual, stale-evidence, memory-contamination, and learned renderers are future work.

## Longitudinal export

```python
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
from synthetic_data.world_peo import EpisodeStep, build_episode

episode = build_episode(
    generate_world(seed=1, enabled=True),
    (EpisodeStep("inspect", 0.8, 0.6), EpisodeStep("release", 0.7, 0.6)),
    context=EpistemicContext("example", 0, EvaluatedSystemVersion("fixture", "world-v1")),
    episode_id="inspect-release-example",
    start_timestamp="2026-10-01T00:00:00Z",
    enabled=True,
)
```

Each step commits the supplied success prediction before simulation, then emits
proposal, execution, observation, verification, and epistemic reconciliation. The
next prediction links to the preceding reconciliation. Existing
`validate_action_bound_sequence()` checks the complete stream's ancestry, timestamps,
and numeric prediction/observation bindings. Timestamps use synthetic one-second
slots, not measured latency. Use distinct episode IDs when combining streams.

The executed operation is `simulate:<action>` with `offline_simulation` scope. It is
reversible offline computation even when the **target fictional action** is
irreversible. A denied target attempt still has a completed simulation and zero
success observation. Verification means deterministic replay agrees with the
simulator snapshot; it is not independent verification of real-world truth. The
measurement uses `TOOL_RESULT`, not an external-verifier claim.

Exports retain snapshots, digests, caller predictions, per-step actor prompts, and
evidence records resolving state/result references to snapshots. Caller predictions
are never filled from the oracle. Callers remain responsible for obtaining them
prospectively; this offline builder cannot prove they did not inspect evaluator data.

Residual is observed success (0 or 1) minus predicted success. Mismatch remains
`unassessed`. World uncertainty is the fraction of facts hidden from that step's
actor: a visibility diagnostic, not a calibrated probability, covariance, or Bayesian
posterior. Model and monitor uncertainty remain unavailable. These numbers do not
change rewards, routing, skills, or persistent memory.

## Admission, compatibility, and validation

Worlds and episodes default to `DEVELOPMENT`. Worlds may instead be `REGRESSION`
or `HIDDEN_EVAL`; TRAIN is rejected. Keep the complete source record and eligibility
metadata when forwarding data. Existing `require_training_eligible()` rejects these
records even when nested. Removing metadata is outside the contract. Actor prompts
alone are not dataset records.

Existing seed generators, rich-case JSONL schema, and training adapter are unchanged.
The world schema is a separate experimental format, unsuitable as direct input to
that adapter. There is no automatic migration, V5 cell assignment, scoring integration,
or curriculum scheduler. PR 10 can evaluate staged curricula against these contracts
and must establish its own admission and evidence gates.

Tests cover types/references, immutable snapshots, hidden-value non-disclosure,
surface invariance, permission sensitivity, denied attempts, state changes,
longitudinal chronology/residual bindings, and training exclusion:

```console
python -m pytest tests/test_synthetic_worlds.py tests/test_synthetic_world_peo.py
```

## Research basis and limits

The [research registry](recommendations/RESEARCH_TRACEABILITY.md#ref-tabpfn-35)
records exact metadata and separates source mechanisms from repository inference:

- [TabPFN-3.5, §3.3](https://arxiv.org/html/2609.17895v2): structured synthetic-prior
  diversity motivates seeded generation. This fixture does not implement its prior.
- [VGCompiler, §§3.1–3.2 and Figure 3](https://arxiv.org/html/2609.22327v1): explicit structure and the
  distinction between surface and graph changes motivate our separate tests.
- [Physical representation languages](https://arxiv.org/abs/2609.23381): observational
  equivalence motivates preserving unknown facts rather than inferring hidden truth.
- [Generalized task and motion planning](https://arxiv.org/abs/2609.30233): frozen
  programs across unseen instances motivate reusable deterministic rules.
- [Qwen-Planner-Agent](https://arxiv.org/abs/2609.29892): action-feedback trajectories
  motivate temporal export through existing PEO contracts.

These are repository adaptations, not paper reproductions. No learned world model,
physical simulator, model-generated code, planner training, or real-world benchmark
result is provided.
