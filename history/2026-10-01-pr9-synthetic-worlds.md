# PR 9: synthetic worlds and longitudinal PEO

## Task spec and reconciliation

Implement the user's PR-9 stage on main `f5e82ca`. Deliver a draft PR. Keep the
17-case manifest, training admission, scoring, runtime authorization, and defaults
unchanged. PR 10 curriculum work and PR 11 adversarial expansion remain separate.

- **Existing:** immutable evidence references/states, prediction/action/observation/
  verification/reconciliation events, sequence validation, training eligibility,
  seeded case generators, rich synthetic-case JSONL schema, V5 coordinates.
- **Partial:** authored examples carry provenance but do not share executable latent
  state across surface variants or temporal transitions.
- **Missing:** a bounded typed world, actor-visible projection, deterministic
  transitions, invariant renderings, decisive counterfactuals, longitudinal exports.
- **Redundant:** another event stream, evidence vocabulary, runtime grant system,
  canonical case, rich-case replacement, or training adapter.
- **Experimental:** all new simulation entry points require explicit opt-in. Outputs
  are development or held-out/regression data; they cannot request TRAIN admission.

## Design

Use boolean relational worlds with agents (goals/incentives), facts (truth,
visibility, observable simulation provenance), scoped permissions (authority
relations), and actions (preconditions, effects, reveals, normative constraints,
consequences, reversibility). Immutable snapshots carry time, parent digest, seed,
and generator provenance. Cross references and exact types fail closed.

Actor projections use existing EvidenceState/EvidenceClaim and omit hidden values,
world digests, seeds, provenance, and oracle labels. A hidden fact remains unknown
even if its latent value would permit an action. Text renderers consume only this
projection and never parse text to determine truth. Plain, reordered, and urgent
surfaces share the same evaluator-side world digest and expected judgment. A
permission counterfactual changes one decisive boolean in an otherwise identical
world. Oracle judgments are fixture expectations, never runtime authorization.

The transition engine records offline simulation attempts. Denied or failed
attempts advance time without applying effects. Successful inspection can reveal
facts for later decisions. Reversibility is descriptive; no automatic undo or
external execution occurs. Consequence strings are descriptive, effects executable.

The episode builder accepts caller-supplied predictions before each transition and
emits existing causal PEO events plus numeric reconciliations. Verification means
agreement with this simulator only. World uncertainty is the fraction of hidden
facts, a visibility diagnostic; model/monitor uncertainty remain unavailable.
Residuals are descriptive, with mismatch unassessed. Export retains eligibility
and full evaluator-side snapshots; actor prompts are separate from that export.

## Execution plan

1. Write failing contract tests for opt-in, validation, visibility, rendering
   invariance, counterfactual sensitivity, and temporal transitions; implement the
   bounded world module and deterministic seed fixture.
2. Write failing longitudinal tests; implement an episode builder using existing
   event factories and validate chronology, ancestry, residuals, and export
   training exclusion. Test inspect/release and denied simulation attempts.
3. Document API, limitations, migration, and research mechanisms. Register four
   primary sources and extend the existing Qwen-Planner linkage. Add ADR 0010.
4. Run focused tests, full suite, lint/format/type checks, package build and installed
   wheel smoke checks. Run one fresh whole-branch review, assess findings against
   current code, fix valid findings, revalidate, commit/push, create/attach draft PR.

## Acceptance and validation

Same-world renderings preserve expected judgments and latent digest. Permission
flips change judgment. Hidden truth cannot leak through actor-facing projection,
IDs, or evidence provenance. Unknown and denied actions fail closed. Predictions
precede simulation; later steps link to earlier reconciliations; serialized events
pass the existing validator. All generated exports reject training admission.
Invalid IDs, references, bool/numeric coercion, and mutation attempts fail. Existing
generators, schema, 17 cases, rewards, and default runtime behavior stay compatible.

Primary sources: TabPFN-3.5 (2609.17895), VGCompiler (2609.22327), Discovering
Physical Representation Languages (2609.23381), generalized task and motion
planning (2609.30233), and existing Qwen-Planner-Agent (2609.29892). Documentation
must separate their demonstrated mechanisms from this repository's inference.

## Rulings

- Use a deliberately bounded boolean simulator; a general planning DSL would add
  unneeded interpreter and validation risk. Cost: continuous dynamics are deferred.
- Log offline simulation attempts as executed actions, including denied attempts;
  the simulated target operation may fail, while the simulator operation occurred.
  Cost: consumers must retain the explicit offline-simulation scope.
- No TRAIN option, even with reviewed metadata: this stage establishes experimental
  data contracts, not evidence for curriculum admission.

## Implementation evidence and review

Before this change, the seed generators produced independent authored records. The
new fixture shares typed state across surfaces and across inspect/release steps.
The two-step example now produces three immutable world snapshots and twelve
validated PEO events. Permission denial changes the expected judgment; hidden-value
changes leave an uninformed actor's projection unchanged. This is contract evidence,
not a measured improvement in learned policy quality.

The 47 new tests cover these behaviors and rejection paths. Focused statement
coverage is 98% across the two new modules (100% for the episode adapter). A fresh
whole-branch review found no functional defect and exercised additional hidden-bit
pairs and a multi-actor episode. Two documentation findings were corrected:

- P2/BLOCK: the guide used lowercase eligibility labels that deserialization rejects.
  The guide now names DEVELOPMENT, REGRESSION, and HIDDEN_EVAL exactly. A temporary
  documentation check failed before the correction and passed afterward, including
  JSON round trips for all three canonical enum values.
- P3/WARN: the VGCompiler mechanism locator pointed to related work. Official
  sections 3.1–3.2 and Figure 3 were checked and corrected in all three mirrors.
  Correcting this small finding supports the user's primary-source traceability
  requirement; no research claim or executable behavior changed.

Review rulings: host-authored public descriptions remain subject to disclosure
review; bad host inputs can expose secrets. Caller prediction provenance cannot be
proved by an offline exporter; misuse can invalidate evaluation. Continuous dynamics,
calibration, and real-world policy quality remain outside this boolean fixture and
need separate validation. No finding is deferred.

## Final validation

- Full suite: 4,155 passed, 18 skipped, 16 warnings in 223.74 seconds.
  The first run hit the unchanged attribution timing assertion because `time.time()`
  measured baseline inference as zero seconds. The test passed in isolation, and
  the subsequent complete suite passed. No timing-test code was changed.
- New contracts: 47 passed; 98% statement coverage across both modules.
- Research mirror checks after documentation fixes: 47 passed.
- Whole-repository Ruff and Black checks passed (602 Python files for Black).
- Mypy passed for the nine CI-selected modules, both new modules, the recommendation
  loader, and the separately checked logging schema. Changed Python lines meet 100 chars.
- Source distribution and wheel built. Installed-wheel checks outside the checkout
  verified both module hashes, packaged guide examples, the 12-event sequence,
  CLI help, 17 canonical cases, and 56 primary references.
- Fresh whole-branch review and documentation precision gate completed; all findings
  corrected. Git whitespace checks passed. No new dependency or default enablement.

## Codex follow-up: custom-world provenance

Verified the PR comment against `build_episode`: snapshots retain supplied world
provenance, but estimates, measurements, innovations, and updates use a hard-coded
generator label. Scope: initialize diagnostic provenance from the input world and
test default, custom, and multi-source lineage through both episode steps and JSON
serialization. Preserve verifier/estimator versions, event contracts, and behavior.

The regression reproduced two failures for custom lineage before the one-line fix;
the bundled-generator case already passed. After the fix, all 397 focused world,
epistemic, continuity, and action-bound tests passed. Ruff, Black, Mypy, and whitespace
checks passed. The full suite was not repeated for this provenance-only follow-up.
