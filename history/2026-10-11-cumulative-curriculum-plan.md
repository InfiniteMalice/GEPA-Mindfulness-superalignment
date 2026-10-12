# Synthetic Curriculum and Cumulative Knowledge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native execution, or superpowers:subagent-driven-development if the maintainer selects delegation. Steps are numbered to respect repository tracking conventions; do not invoke bd or modify .beads.

**Goal:** Add opt-in offline lesson curation, milestone diagnostics, balanced candidate curricula,
and reproducible seven-arm comparison plans without conferring training or deployment authority.

**Architecture:** Immutable inputs retain complete source records and existing lesson identities.
Pure functions authenticate detached host review payloads, derive diagnostic dispositions and
compose existing generators and PR-6 audit functions. Existing training and lifecycle APIs stay unchanged.

**Tech Stack:** Python >=3.10, standard library, existing repository contracts, pytest, Ruff, Black, mypy.

**Spec:** [Approved PR-7 design](2026-10-11-cumulative-curriculum-design.md).

**Status:** Design approved on 2026-10-11; implementation plan and native execution approved
by the maintainer on 2026-10-12. Live studies and publication remain separate decisions.
Base: `6d14ca0d4289207c6d2da3e2a05b1b90d320c88f`; branch: `codex/cumulative-curriculum-pr7`.

## Global constraints

- No new dependency, model call, training run, reward adjustment, admission receipt, catalog write or deployment action.
- Preserve exactly 17 canonical cases, noncanonical Case 0, normative commitments and existing abstention/clarification behavior.
- Every exported report: `training_eligibility="DEVELOPMENT"`, `optimizer_input=False`, `confers_authority=False`; deployment status, where present, is `not_assessed`.
- Preserve every nested source restriction. Hidden-evaluation-derived inputs appear only in excluded audit entries; no relabeling to TRAIN.
- `curated` means included in an offline report. It does not change `LessonReviewStatus`, select a learning destination or grant use permission.
- Public operations require keyword-only `enabled: bool = False`; anything except built-in `True` raises before callbacks or generation.
- Callback success requires built-in `True` and a canonically unchanged detached payload. Exceptions, mutation and non-boolean truthy values fail closed.
- Caller supplies UTC as-of time. No wall-clock reads, implicit expiry refresh, or persisted callback approvals.
- Unknown observations remain unknown; failed/cancelled attempts, severe events and exclusions remain visible.
- Python lines <=100 characters, labeled import sections, no unused imports, existing formatter/linter conventions.
- Software completion does not require an active PR-5 penalty or establish withheld-scenario usefulness.

## Review focus

1. A bank or source changes after review: withhold the affected lesson, retaining the old binding and all reasons (Task 2).
2. A valid JSON envelope hides deep JSON in a source string: reject before the embedded decoder recurses (Task 1).
3. A lesson overlaps held-out data through a bank ancestor: record exposure for the consuming candidate, never claim independent improvement (Task 5).
4. Matched experiment limits conceal unequal actual compute/data consumption: planned matching stays separate from actual matching (Task 5).
5. A generator places evaluator-only answers in exported prompt metadata: actor views contain only original public role/content turns (Task 4).

## File map and dependencies

Create four production modules named in the design:
`synthetic_data/curriculum_records.py`, `synthetic_data/lesson_curation.py`,
`synthetic_data/cumulative_curriculum.py`, `evaluation/curriculum_comparison.py`.

Create `tests/curriculum_fixtures.py` and six test files:
`tests/test_curriculum_records.py`, `tests/test_lesson_curation.py`,
`tests/test_curriculum_milestones.py`, `tests/test_cumulative_curriculum.py`,
`tests/test_curriculum_comparison.py`, `tests/test_cumulative_curriculum_integration.py`.

Task 1 establishes shared inputs; Tasks 2-3 implement lesson evidence; Task 4 composes candidate
examples; Task 5 composes comparisons; Task 6 verifies the end-to-end boundary and documents it.
These tasks are sequential because their input contracts and trust boundaries are shared.

Modify only relevant documentation, package data and documentation tests in Task 6. Add
`curriculum_fixtures` to Ruff's existing first-party list when introducing that helper. Do not
reformat unrelated files or add the seven pre-existing untracked planning files to commits.

## Exact common record contract

All new records are frozen, keyword-only dataclasses with explicit `to_dict()` and
`from_dict(data: object)` methods, exact fields, and a `schema_version` wire discriminator.
Use `curriculum-<kebab-case-class-name>-v1`. Constructors, deserializers and public-operation
snapshots enforce the same invariants. Do not inherit PR-6's generic serializer: existing
`LessonProposal` has private routing state that must travel through its public serializer.

Use existing `EvidenceReference`, `TrustedEvaluatorContract`, `LessonProposal`, `SystemConfig`,
`CausalPair`, `DatasetManifest` and PR-6 audit records through their public serializers. Evidence
must be observable; external authentication/replay evidence requires `EXTERNAL_RECORD` references.
Never deserialize a private field or invoke an input-specified class/function.

`JsonObject` means `dict[str, Any]` at public boundaries. Internal source snapshots use canonical
JSON strings and decode to detached objects. `Refs` below means `tuple[EvidenceReference, ...]`;
`IDs` means `tuple[str, ...]`. Neither alias changes the wire representation: tuples become arrays.

Provide `curriculum_digest(record: object) -> str`, accepting only the new exact record types,
and `read_curriculum_object(value: str) -> JsonObject` for bounded embedded JSON. Reuse
`evaluation.causal_records.canonical_json` and `content_digest`; use the unchanged public
`gepa_mindfulness.training.causal_reward_records.validate_input_envelope` for object-envelope
checks. Keep the embedded-string lexical
depth check local and test it before `json.loads`; do not import PR-5 private parsing helpers.

Bounds: complete decoded operation envelope <=4 MiB of canonical UTF-8, maximum depth 32
(root counts as one), <=1,024 entries per tuple, IDs <=256 characters, prose <=16,384 characters.
Reject cycles, duplicate JSON keys, nonfinite numbers, bool-as-number, invalid Unicode, unknown
fields/schema versions and noncanonical encoded strings. Embedded JSON is subject to the same
decoded bounds; an encoding layer cannot evade the aggregate limit. Integers are serialization-safe
within +/-`(2**53 - 1)`; ordinals/counts are nonnegative. Digest strings are lowercase SHA-256.
Return `ValueError` with a field/reason; do not expose source bodies in error messages.

### Record inventory (defined in `synthetic_data/curriculum_records.py`)

Fields shown without defaults are required. Optional means an explicit JSON null, not an omitted field.

| Record | Fields |
| --- | --- |
| `CurationPolicy` | `policy_id: str`, `version: str`, `evaluator: TrustedEvaluatorContract` |
| `EpisodeSource` | `episode_id: str`, `case_ids: IDs`, `family_id: str`, `parent_episode_ids: IDs`, `system: SystemConfig`, `terminal_status: str`, `source_json: str`, `evidence_refs: Refs` |
| `BankEntry` | `lesson: LessonProposal`, `version: str`, `source_episode_ids: IDs`, `source_json: str` |
| `BankSnapshot` | `bank_id: str`, `version: str`, `entries: tuple[BankEntry, ...]` |
| `EvidenceCheck` | `name: str`, `status: str`, `reason: str`, `evidence_refs: Refs` |
| `ReplayEvidence` | `spec_json: str`, `system: SystemConfig`, `seed_policy: str`, `seeds: tuple[Optional[int], ...]`, `results_json: str`, `evidence_refs: Refs` |
| `LessonReview` | `lesson_digest: str`, `bank_digest: str`, `source_digests: IDs`, `policy_digest: str`, `checks: tuple[EvidenceCheck, ...]`, `replay: Optional[ReplayEvidence]`, `valid_from: str`, `valid_until: str`, `status: str`, `human_required: bool`, `evidence_refs: Refs` |
| `UtilitySlot` | `slot_id: str`, `task_id: str`, `content_digest: str`, `family_id: str`, `condition: str`, `repeat_id: int`, `seed: Optional[int]`, `system: SystemConfig`, `rubric_digest: str`, `budget_json: str` |
| `UtilityObservation` | `slot: UtilitySlot`, `status: str`, `success: Optional[bool]`, `source_json: str`, `evidence_refs: Refs` |
| `UtilityStudy` | `study_id: str`, `lesson_digest: str`, `bank_digest: str`, `seed_policy: str`, `expected_slots: tuple[UtilitySlot, ...]`, `observations: tuple[UtilityObservation, ...]` |
| `MilestoneReview` | `milestone_id: str`, `episode_digest: str`, `step_id: str`, `goal: str`, `before_json: str`, `after_json: str`, `step_completed: Optional[bool]`, `goal_progress: Optional[bool]`, `epistemic_update: Optional[bool]`, `final_success: Optional[bool]`, `goal_authorized: Optional[bool]`, `constraint_violation: Optional[bool]`, `severe_event: Optional[bool]`, `status: str`, `human_required: bool`, `evidence_refs: Refs` |
| `BoundaryCandidate` | `candidate_id: str`, `pair: CausalPair`, `source_json: str`, `row_list: str`, `row_id_key: str`, `row_id: str`, `case_ids: IDs`, `parent_case_ids: IDs`, `generator_id: str`, `generator_config_json: str`, `coverage_tags: IDs` |
| `GenerationReview` | `candidate_digest: str`, `policy_digest: str`, `status: str`, `human_required: bool`, `checks: tuple[EvidenceCheck, ...]`, `evidence_refs: Refs` |
| `AblationArm` | `arm_id: str`, `candidate_id: str`, `control_arm_id: Optional[str]`, `components_json: str`, `data_case_ids: IDs`, `starting_system: SystemConfig`, `optimizer_digest: str`, `reward_digest: str`, `seed_policy: str`, `seeds: tuple[Optional[int], ...]`, `planned_budget_json: str`, `actual_budget_json: Optional[str]`, `knowledge_report_json: Optional[str]` |
| `CurationExposure` | `ordinal: int`, `case_ids: IDs`, `candidate_ids: IDs`, `round_ids: IDs`, `source_json: str`, `evidence_refs: Refs` |

`source_json`, replay and bank sources retain full original eligibility/provenance, not just IDs.
Episode terminal values: `completed_success`, `completed_failure`, `pending`, `cancelled`, `incomplete`.
Review status: `verified`, `disputed`, `unresolved`. Check status: `pass`, `fail`, `unknown`.
Timestamps: exact UTC `YYYY-MM-DDTHH:MM:SSZ`, inclusive `valid_from`, exclusive `valid_until`;
the latter must exceed the former. Seed policies: `shared`, `independent`, `unknown`.
Source-ID references resolve against the complete supplied roster, including bank ancestors;
duplicate IDs, missing parents and ancestry cycles reject structurally. Missing review bindings
to otherwise well-formed source versions are withheld, not fabricated.

## Task 1: Exact records and source boundaries

**Files:** Create `synthetic_data/curriculum_records.py`, `tests/curriculum_fixtures.py`,
`tests/test_curriculum_records.py`; modify Ruff first-party classification in `pyproject.toml`.

**Interfaces:** Produce the record inventory, digest and bounded-parser functions above.
The test helper exposes `curation_inputs() -> dict[str, Any]` with the Task 2 keyword inputs:
one approved episode-fact lesson, an empty bank, one completed DEVELOPMENT episode, an exact
six-check review with replay evidence, a policy, and `as_of="2026-10-11T12:00:00Z"` inside its validity window.
All helper evidence is authored fixture data, never independently verified live evidence.

1. Add failing tests: `test_record_round_trip_detaches_sources`,
   `test_nested_json_depth_and_size_bounds`, `test_duplicate_ids_and_ancestry_rejected`,
   `test_review_time_interval_is_exact`, and `test_private_evidence_is_rejected`.
   Assert round-trip canonical wire equality; mutation of returned source objects leaves the
   record unchanged. Depth 32 accepts, 33 and 20,000 reject with `ValueError`, including embedded
   JSON; escaped brackets in strings do not count as containers. Test the aggregate bound with
   several individually small embedded sources whose decoded sum exceeds 4 MiB.
2. Run `python -m pytest tests/test_curriculum_records.py -q`; establish RED before implementation.
3. Implement explicit per-record decoding, validation and detached snapshots. Use existing
   public serializers for nested repository records. Missing semantic review facts use explicit
   unknown states; malformed schema types reject. Defer business dispositions to Task 2.
4. Re-run the record tests, Ruff/Black on changed files, and scoped mypy on the new module.
   All return exit 0; commit `feat(curriculum): add bounded offline evidence records`.

## Task 2: Lesson curation and utility diagnostics

**Files:** Create `synthetic_data/lesson_curation.py`, `tests/test_lesson_curation.py`;
extend the fixture helper and record tests as required by the listed contracts.

**Interfaces:** `HostCheck = Callable[[JsonObject], bool]`.
Produce these exact public functions in `lesson_curation.py`:

```python
def curate_lessons(*, episodes: tuple[EpisodeSource, ...], lessons: tuple[LessonProposal, ...],
    bank: BankSnapshot, reviews: tuple[LessonReview, ...], policy: CurationPolicy,
    as_of: str, authenticate: HostCheck | None = None, enabled: bool = False) -> JsonObject: ...
def evaluate_lesson_utility(study: UtilityStudy, *, lesson: LessonProposal, bank: BankSnapshot,
    policy: CurationPolicy, as_of: str, authenticate: HostCheck | None = None,
    enabled: bool = False) -> JsonObject: ...
```

Signatures above are wrapped for the plan; implementation uses normal 100-column formatting.
Use one private callback gate shared with Task 3. Its payload has exact keys `purpose`, `policy`,
`as_of`, `subject`; purpose is `lesson_review`, `utility_study`, or `milestone_review`.
`subject` contains complete relevant records, including the entire bank/source roster, not only digests.

1. Add these failing assertions using `curation_inputs()` and accepting authored callbacks:

```python
report = curate_lessons(**curation_inputs(), authenticate=lambda payload: True, enabled=True)
assert report["lessons"][0]["disposition"] == "curated"
assert report["training_eligibility"] == "DEVELOPMENT"
assert report["optimizer_input"] is False
assert report["confers_authority"] is False
```

   Add `test_changed_bank_or_source_withholds`, `test_curated_is_not_review_approval`,
   `test_callback_exact_true_and_payload_integrity`, `test_failure_episode_can_support_negative_lesson`,
   `test_nested_hidden_origin_is_excluded`, `test_expiry_boundary`, and
   `test_all_reasons_survive_rejection`. Test every check's fail/unknown state, pending/rejected
   human review, absent replay, missing review, duplicate lesson content, unknown source and future as-of.
2. Run `python -m pytest tests/test_lesson_curation.py -q`; confirm RED.
3. Implement curation with six required checks: `support`, `contradiction_clear`, `freshness`,
   `reproducible`, `information_preserved`, `nonredundant`. All source episodes must be completed;
   completed failures retain their original failures. Nested HIDDEN_EVAL or held-out origins
   are always excluded from `curated`. No call to the training checker can reclassify an input.
   A rejected proposal or authenticated failed check yields `rejected`; otherwise any missing,
   disputed, expired, unauthenticated or stale binding yields `withheld`. Curation also requires
   the proposal's existing review status to be `approved`, review status `verified`, and
   `human_required=False`; otherwise withhold. Only then return `curated`.
   Local exact duplicate lesson IDs reject input; different IDs with identical proposal summary
   and characteristics yield `rejected` with `duplicate_content`. Preserve all applicable reasons.
4. Produce `lesson-curation-v1` reports with complete `inputs`, `lessons` (ID, disposition,
   reasons, authentication, checks), and `proposed_bank_additions` containing only curated entries.
   Do not modify the supplied bank. Add `result_digest` over the report before that field is added.
5. Add utility tests with four conditions: `none`, `lesson_only`, `bank_only`, `bank_plus_lesson`.
   Two matched opportunities with lesson-only outcomes `(True, False)` give standalone rate `0.5`;
   bank-only `(False, False)` and bank-plus `(True, False)` give marginal delta `0.5`.
   Missing expected observations do not count as failures. A success without support cannot
   change the curation disposition. Seed/config/rubric/budget mismatch leaves comparisons unverified.
6. Implement `lesson-utility-v1`: expected/observed/missing/failed/censored counts and raw success
   rates per condition; raw matched-pair deltas separately from authenticated comparisons.
   `UtilityObservation.status` is `observed`, `failed`, `censored`, or `missing`; only `observed`
   carries a boolean success. Missing roster members are synthesized only as missing report rows.
   Match strata by task/content/family/repeat and exact executor/rubric/budget. `shared` requires
   equal non-null seeds for paired deltas. `independent` reports differences of condition means
   within those strata, with recorded seeds and no paired claim. `unknown`, or missing seeds,
   prevents an authenticated comparison but retains raw rates. Test all three policies explicitly.
   Conditions bind the exact lesson and bank.
   Emit no interval in this standalone report (`interval=None`, `unsupported_source_for_interval`);
   PR-6 supported-source uncertainty is joined only in Task 5.
7. Re-run Task 1-2 tests, scoped types and formatting. Commit
   `feat(curriculum): authenticate lesson curation and utility evidence`.

## Task 3: Milestones without reward or success conflation

**Files:** Extend `synthetic_data/lesson_curation.py`; create `tests/test_curriculum_milestones.py`.

**Interface:** `review_milestones(episodes: tuple[EpisodeSource, ...], milestones: tuple[MilestoneReview, ...], *, policy: CurationPolicy, as_of: str, authenticate: HostCheck | None = None, enabled: bool = False) -> JsonObject`.

1. Add failing `test_completed_step_is_not_goal_progress`: an authenticated completed episode
   with step=True, progress=False, update=None, final=False yields those four distinct verified
   values. Add `test_unknown_goal_authority_withholds_progress`,
   `test_severe_event_survives_progress`, `test_no_final_success_from_step_count`,
   `test_unverified_milestone_retains_claims_only`, and duplicate-ID/source mismatch tests.
2. Run `python -m pytest tests/test_curriculum_milestones.py -q`; confirm RED.
3. Implement `curriculum-milestones-v1`, retaining each input claim and separate verified values.
   Verification requires exact source binding, completed episode, authenticated undisputed review,
   and no outstanding human requirement. Verified progress/final success additionally require
   goal_authorized=True, constraint_violation=False and severe_event=False. Otherwise those two
   verified values are null with reasons; preserve original claims and severe evidence separately.
   Step completion and epistemic updating are independently reported, not summed into success.
4. Assert there is no optimizer accessor, numeric coercion or reward-total field. Run Tasks 1-3
   tests and scoped checks; commit `feat(curriculum): separate observable milestone judgments`.

## Task 4: Provenance-preserving boundary curriculum

**Files:** Create `synthetic_data/cumulative_curriculum.py`, `tests/test_cumulative_curriculum.py`.

**Interface:** `export_cumulative_curriculum(candidates: tuple[BoundaryCandidate, ...], reviews: tuple[GenerationReview, ...], *, policy: CurationPolicy, as_of: str, authenticate: HostCheck | None = None, enabled: bool = False) -> JsonObject`.

Consume Task 1 records and existing generated `CausalPair` objects. Do not implement another
generator: examples in tests/guide call `adapt_relation_pair`, `make_debate_ablation`, or
`generate_controlled_laundering_families` and pass their actual pairs into this export function.

1. Add failing `test_balanced_bundle_requires_independent_reviews`: four reviewed candidates
   cover `invariance`, `sensitivity`, `benign_assistance`, `legitimate_refusal`; output coverage is
   `complete`. Missing any required tag gives `incomplete` and exact missing tags. Coverage tags
   are claims authenticated by review, never inferred from prompt wording.
2. Add `test_actor_prompts_exclude_labels`, `test_compound_edits_remain_compound`,
   `test_decisive_case_change_keeps_each_arm_identity`, `test_hidden_parent_excluded`,
   `test_generated_failure_never_becomes_target_truth`, and serialization restriction tests.
   Use a sentinel evaluator-only answer and assert it is absent from the actor-view serialization.
3. Run `python -m pytest tests/test_cumulative_curriculum.py -q`; confirm RED.
4. Implement source selection for `causal-diagnostics-v1` (`rows`, `opportunity_id`) and
   `sensitive-debate-report-v1` (`metric_rows`, `opportunity_id`). Validate report digest, unique
   row selector and source schema. Retain complete source and pair. A source diagnostic status
   only identifies a boundary; independent generation review supplies label validity.
5. Require checks `expected_actions`, `declared_changes`, `case_identity`, `source_lineage`,
   `coverage_tags`, `public_prompt_separation`, each pass before inclusion. Authenticate a detached
   payload with `purpose="generation_review"`, complete candidate/review/policy/as-of. The host
   checks independence from generator/lesson author. Missing or disputed review excludes the
   candidate with reasons. The authenticated payload includes the complete retained source,
   generator identity/configuration and review evidence references, binding the host's independence
   check to that generator. Unknown rows and forged digest/schema reject structurally.
6. Export `cumulative-curriculum-v1` with complete inputs, `examples`, `excluded`, `coverage`,
   restrictions and digest. `examples` have actor prompts from `render_causal_variant` only;
   evaluator labels, changes and provenance occupy a separate evaluator object. Report nested
   restrictions, never drop them to make a trainable payload. All input IDs appear exactly once
   across examples and excluded; input order does not change ID-sorted output.
7. Run Tasks 1-4 tests plus existing causal/debate/pluralistic generator tests. Commit
   `feat(curriculum): export independently reviewed boundary examples`.

## Task 5: Seven-arm manifests, exposures and PR-6 integration

**Files:** Create `evaluation/curriculum_comparison.py`, `tests/test_curriculum_comparison.py`.

**Interface:** Produce `audit_curriculum_comparison(arms: tuple[AblationArm, ...], protocol: ImprovementProtocol, manifest: DatasetManifest, journal: AttemptJournal, evidence: tuple[DiagnosticEvidence, ...], *, as_of: str, curation_exposures: tuple[CurationExposure, ...] = (), exposures: tuple[ExposureRecord, ...] = (), final_authorizations: tuple[FinalTestAuthorization, ...] = (), authenticate_manifest: HostCheck | None = None, authenticate_audit: Callable[[AuthenticationRequest], bool] | None = None, enabled: bool = False) -> JsonObject`.

1. Add failing `test_seven_arms_have_explicit_controls`,
   `test_planned_match_does_not_imply_actual_match`, `test_unintended_component_change`,
   `test_knowledge_bank_ancestor_exposes_holdout`, `test_knowledge_arm_requires_fresh_authentication`,
   `test_control_result_matches_protocol_baseline`, and
   `test_cancelled_attempt_and_severe_event_survive`. A hand-authored changed reward digest
   prevents an isolated-curriculum claim despite identical advertised arm names.
2. Run `python -m pytest tests/test_curriculum_comparison.py -q`; confirm RED.
3. Require exactly seven unique arm IDs: `baseline`, `synthetic`, `causal`, `causal_laundering`,
   `causal_debate`, `causal_pluralistic`, `knowledge`. Candidate IDs resolve against the protocol.
   Controls: baseline=null, synthetic=baseline, causal=synthetic, the next three=causal; knowledge
   explicitly names one preceding frozen arm or reports `control_missing` (no inferred control).
   Distinct arms require distinct candidate IDs. For each controlled arm, its protocol candidate's
   `baseline` must equal the control candidate's resulting `candidate` SystemConfig; a mismatch
   prevents isolation. This comparison is distinct from equality of the arms' starting systems.
4. `components_json` is an exact map with keys `synthetic`, `causal_pairs`, `laundering`,
   `sensitive_debate`, `pluralistic`, `knowledge`. Each value is a configuration digest or null.
   Baseline has all null; synthetic adds synthetic; causal adds causal_pairs; arms 4-6 each add
   only their named component; knowledge adds knowledge to its control. Shared component digests,
   starting system, optimizer/reward digests and seed policy must match for an isolation claim.
   Changed component datasets are allowed, but unchanged component configurations must bind
   unchanged source rosters through the independently authenticated complete manifest.
5. Budget JSON is exactly `{"train": {"compute": number, "unit": str, "examples": int, "tokens": int}, "evaluation": {same four fields}}`; nonnegative finite values, built-in integers for counts.
   Planned equality and actual equality are separate exact canonical comparisons. Actual absent
   means unknown. Unequal units, counts or measured compute prevent actual matched claims; no
   unit conversion or tolerance is silently applied. Cost stays in PR-6's existing journal.
6. Authenticate complete arms/protocol/manifest/journal/evidence/exposures/authorizations in one
   detached `curriculum_manifest` payload including `as_of`. This verifies actual component isolation, source
   completeness and budget measurements; hashes alone do not. Knowledge readiness also requires
   the complete `lesson-curation-v1` report with current bank/source bindings and curated entries;
   the host must freshly attest those bindings and their validity at the current `as_of`.
   Recheck embedded review expiry locally at that time. Serialized curated flags alone are insufficient.
7. Resolve every curation-exposed case and all ancestors in the manifest, including bank ancestors.
   Convert each affected partition to an existing `ExposureRecord` with `use="proposal"`, exact
   `partition_digest`, ordinal and recipients. Retain original `CurationExposure` records as audit
   evidence. Join these with supplied exposures and call unchanged `audit_improvement` with its
   separate `authenticate_audit` callback. Never bypass final-test authorization or unsupported
   diagnostic-source checks. Use `validate_manifest` for partition/family/content independence.
   The manifest callback also attests that curation exposures cover all inspected lesson/bank
   sources and actual consuming candidates/rounds. Missing source correspondence withholds
   knowledge readiness; an empty exposure list never establishes that inspection was absent.
8. Return `curriculum-comparison-v1`: complete inputs, each arm's planned/actual budget match,
   isolation and readiness with reasons, derived exposure records, the entire PR-6 report, and
   DEVELOPMENT/no-authority/deployment-not-assessed fields. Unauthenticated manifests cannot
   endorse matched comparisons; retain PR-6 evidence and severe events without promoting them.
9. Run Task 5 plus existing improvement audit/statistics/integration tests. Commit
   `feat(curriculum): audit controlled experiment arms and curation exposure`.

## Task 6: Integration guide, quality gate and final review

**Files:** Create `docs/cumulative_curriculum.md`, `tests/test_cumulative_curriculum_integration.py`;
modify `docs/peo_curriculum.md`, `docs/adr/0021-causal-alignment-diagnostic-extension.md`,
`docs/recommendations/RESEARCH_TRACEABILITY.md`, `docs/recommendations/references.yaml`,
`pyproject.toml` docs package data, and `tests/test_documentation_links.py` DOCUMENTS.

**Interfaces:** Consume all preceding public functions; no additional production entry point.

1. Add an executable guide example test showing a supported lesson, an expired review withheld,
   a utility increase with no truth authorization, a completed step without progress, a balanced
   offline export and seven planned arms with unknown actual budgets. Run the new integration
   file RED before writing the executable example; then make it pass through the real public APIs.
2. Add end-to-end rejection of the actual exported metadata by `require_training_eligible`.
   Wrap public prompts and full source metadata into existing `RolloutRequest`/`CurriculumUnit`
   fixtures for `PEOCurriculumDataset`: collect/evaluate work, train/resume reject. Populate all
   existing required pools without weakening quotas or anchor retention. No production trainer wiring.
3. Document exact schemas, callbacks, statuses, exclusions, time bounds, utility denominators,
   experiment controls and observed-budget limitations. Cite the three existing research entries;
   mark proposed adaptations/not-reproduced evidence accurately. Preserve earlier ADR statuses and
   record only approvals actually received. Link the guide and include it in wheel package data.
4. Run all validation commands below and record commands, exit codes, baseline/head identities,
   test counts and limitations under `../../outputs/PR7-verification-2026-10-11/`. Do not claim a
   withheld-scenario gain from authored software fixtures. Resolve documentation-precision BLOCKs.
5. Build the wheel with `python -m build --wheel --no-isolation`, install without dependencies
   into a new output directory, and run the exact guide example outside the checkout with only
   that installed target on PYTHONPATH. Verify the four module paths, packaged guide presence,
   and full parsed JSON equality with the source example. Do not install new dependencies.
6. Commit `docs(curriculum): document and validate offline curation contracts`. Under the selected
   execution workflow, perform its independent final review, verify findings against current code,
   fix still-valid issues minimally and validate affected behavior. Do not auto-merge or run a
   live study. Publication of the new PR-7 branch is a separate integration decision.

## Validation commands

Use the existing interpreter from the repository root. `python -m ...` in task steps means
`& $pr7Python -m ...`. No model/network access is needed for these validations.

```powershell
$pr7Python = 'C:/Users/evanh/Documents/Codex/2026-10-07/files-pasted-by-the-user-you/work/venv/Scripts/python.exe'
$env:PYTHONPATH="$PWD;$PWD/src;$PWD/modules;$PWD/reasoning-generalization-tracer/src"
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$pr7Source = @('synthetic_data/curriculum_records.py', 'synthetic_data/lesson_curation.py', 'synthetic_data/cumulative_curriculum.py', 'evaluation/curriculum_comparison.py')
$pr7Tests = @('tests/test_curriculum_records.py', 'tests/test_lesson_curation.py', 'tests/test_curriculum_milestones.py', 'tests/test_cumulative_curriculum.py', 'tests/test_curriculum_comparison.py', 'tests/test_cumulative_curriculum_integration.py')
```

Before product edits, save a baseline from these existing tests; repeat after integration:

```powershell
& $pr7Python -m pytest -q tests/test_causal_records.py tests/test_causal_interventions.py tests/test_debate_interventions.py tests/test_pluralistic_curriculum.py tests/test_peo_curriculum.py tests/test_training_eligibility.py tests/test_learning_surfaces.py tests/test_improvement_records.py tests/test_improvement_audit.py tests/test_improvement_statistics.py tests/test_improvement_integration.py tests/test_causal_reward.py tests/test_causal_reward_integration.py
```

Final checks (each requires exit 0; inspect and record each result separately):

```powershell
& $pr7Python -m pytest -q @pr7Tests
& $pr7Python -m pytest -q tests/test_documentation_links.py tests/test_recommendation_documentation_consistency.py tests/test_v5_documentation_consistency.py
& $pr7Python -m ruff check .
& $pr7Python -m black --check --line-length 100 .
& $pr7Python -m mypy --follow-imports=silent @pr7Source
& $pr7Python -m pytest -q
git diff --check
```

Also run the existing CI mypy targets from `.github/workflows/ci.yml`, unchanged; on Windows use
`--config-file=NUL` for its isolated logging-schema check. Python source length <=100 is enforced
by Ruff E501 and an explicit changed-file scan. Format only new/modified Python, not the repository.
Tests outside the new module remain required even when the focused tests pass. Record any failure
by name; do not erase or relabel it as pre-existing without evidence.

## Self-review and execution handoff

Coverage: exact bounded records (Task 1), completed lessons/contradictions/freshness/replay and
utility (Task 2), separate progress facts (Task 3), controlled examples and generation validation
(Task 4), seven arms/actual budgets/source exposure/PR-6 reuse (Task 5), unchanged admission,
packaging and documentation (Task 6). Every review-focus item has a named owning test.

Planning checks verify source APIs, schema/signature consistency, reference paths and spec
coverage. Proposed feature tests do not exist yet and have not been reported as passing.
The maintainer approved this plan and native execution on 2026-10-12. The six tasks share
contracts and are followed by one fresh whole-branch reviewer. Live execution and publication
remain separate decisions.
