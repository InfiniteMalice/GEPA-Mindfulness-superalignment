# Offline cumulative curriculum

PR-7 provides opt-in diagnostic functions for externally reviewed lessons and curriculum
experiments. The functions do not execute models, train, alter rewards, modify a knowledge bank,
or admit data. Every report carries `training_eligibility="DEVELOPMENT"`,
`optimizer_input=False`, and `confers_authority=False`.

The [knowledge-utility pilot protocol](experiments/knowledge_utility_pilot.md) prepares a
fixed 72-request study packet offline, with pending knowledge reviews and no live execution.

## Operations and results

All five operations require built-in `enabled=True` and a caller-supplied `as_of` timestamp.
The timestamp format is `YYYY-MM-DDTHH:MM:SSZ`. The functions never read the wall clock.
Malformed records raise `ValueError`; missing authentication leaves claims unverified.

| Operation | Module | Result schema | Main result fields |
| --- | --- | --- | --- |
| `curate_lessons` | `synthetic_data.lesson_curation` | `lesson-curation-v1` | `lessons`, `proposed_bank_additions` |
| `evaluate_lesson_utility` | `synthetic_data.lesson_curation` | `lesson-utility-v1` | `conditions`, `observations`, `comparisons`, `interval`, `interval_reason` |
| `review_milestones` | `synthetic_data.lesson_curation` | `curriculum-milestones-v1` | `milestones` |
| `export_cumulative_curriculum` | `synthetic_data.cumulative_curriculum` | `cumulative-curriculum-v1` | `examples`, `excluded`, `coverage` |
| `audit_curriculum_comparison` | `evaluation.curriculum_comparison` | `curriculum-comparison-v1` | `arms`, `manifest_authenticated`, `derived_exposures`, `improvement_audit`, `deployment_eligibility` |

Every report also retains complete `inputs` and a `result_digest` of the canonical body before
adding that digest. Digests establish integrity only. The comparison report includes the entire
unchanged [PR-6 audit](reliable_improvement_evaluation.md), including severe events, cancelled
attempts, missing outcomes, exposure accounting, and final-test authorization results.

## Records and exact serialization

The [record module](../synthetic_data/curriculum_records.py) defines frozen keyword-only records.
Each record supports `to_dict()` and `from_dict(data)`; all fields are required, including nulls.
The wire discriminator is `schema_version="curriculum-<kebab-case-class-name>-v1"`.
For example, `BankSnapshot` uses `curriculum-bank-snapshot-v1`. Tuples serialize as arrays.
`Refs` means a tuple of observable `EvidenceReference` values; `IDs` means a tuple of strings.
`Optional[T]` means `T` or null. Existing nested records use their public serializers.

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

The complete input envelope is bounded to 4 MiB and depth 32, including decoded embedded JSON.
Arrays contain at most 1,024 entries. IDs are bounded to 256 characters; reasons and goals to
16,384. JSON objects in `*_json` fields use `canonical_json`; duplicate keys, nonfinite values,
unknown fields and schemas, cycles, invalid Unicode and unsafe integers reject. Counts and
ordinals are nonnegative. Embedded JSON receives the same checks before nested legacy decoders
run; malformed non-JSON prose remains source text. Review and replay references include an
external record.

Episode terminal states are `completed_success`, `completed_failure`, `pending`, `cancelled`,
and `incomplete`. Completed failures may support negative lessons without becoming successes.
Reviews use `verified`, `disputed`, or `unresolved`; checks use `pass`, `fail`, or `unknown`.

## Host authentication and curation

The host supplies callbacks over complete detached JSON payloads. Acceptance requires built-in
`True` and an unchanged canonical payload. Exceptions, mutation, missing callbacks, and truthy
non-booleans fail closed. A callback must independently check evaluator identity, independence
from the generator or lesson author, source completeness, semantic support, and current bindings.
A callback that merely echoes a serialized verification flag does not fulfill this contract.
These semantic checks are host responsibilities; unit tests can only verify the callback boundary.

Lesson, utility, milestone, and generation payloads contain `purpose`, `policy`, `as_of`, and
`subject`. The comparison manifest payload contains `purpose="curriculum_manifest"`, `as_of`,
and the complete comparison `subject`. Its separate `authenticate_audit` callback receives the
existing PR-6 `AuthenticationRequest` and retains that API's authentication obligations.

Curation requires an existing approved `LessonProposal`, completed source episodes, exact lesson,
bank, source and policy bindings (including every source ancestor), an undisputed authenticated
review, no outstanding human review,
and passing `support`, `contradiction_clear`, `freshness`, `reproducible`, `information_preserved`,
and `nonredundant` checks. Replay evidence retains specification, configuration, recorded seeds,
and externally captured results; the curator never runs the replay. Validity is inclusive at
`valid_from` and exclusive at `valid_until`.

`curated` means included in this offline report. Missing, expired, disputed or stale evidence
produces `withheld`. An already rejected proposal, exact duplicate content, or an authenticated
failed check produces `rejected`. Every applicable reason remains visible. Bank inputs are
immutable; `proposed_bank_additions` do not update any active store or learning destination.
Incoming lesson IDs must be distinct from existing bank lesson IDs; collisions reject the input.
Hidden-evaluation and held-out origins remain in audit inputs but cannot supply curated additions
or included boundary examples. This includes resolved bank source episodes and their ancestry.
Future training admission requires its own source process.

## Utility and milestones

Utility conditions are `none`, `lesson_only`, `bank_only`, and `bank_plus_lesson`. Each condition
reports expected, observed, missing, failed and censored counts, successes, and success rate over
observed outcomes only. Missing outcomes are not failures. Comparisons report lesson-only minus
none (`standalone`) and bank-plus-lesson minus bank-only (`marginal`). The condition's lesson-only
rate is the standalone success statistic; neither statistic proves factual support.

Contrasts match task/content/family/repeat, executor, rubric, and budget. `shared` requires equal
known seeds for a paired claim; `independent` reports stratified condition-mean differences with
recorded seeds and no paired claim. `unknown` or absent seeds leave comparisons unverified.
Raw descriptive deltas remain visible. Standalone utility intervals are null with reason
`unsupported_source_for_interval`; supported-source uncertainty remains PR-6's responsibility.

Milestone rows keep input `claims` separate from four nullable `verified` judgments:
`step_completed`, `goal_progress`, `epistemic_update`, and `final_success`. Progress and final
success require affirmative goal authorization, no constraint violation and no severe event,
in addition to authenticated completed-source review. No summation of milestones implies success.
This release emits no shaping reward or policy-invariance guarantee.

## Boundary examples and comparisons

Generation reviews require passing `expected_actions`, `declared_changes`, `case_identity`,
`source_lineage`, `coverage_tags`, and `public_prompt_separation` checks. Supported source schemas
are `causal-diagnostics-v1` (`rows`, `opportunity_id`) and `sensitive-debate-report-v1`
(`metric_rows`, `opportunity_id`). The host reviews generated labels separately from the source
failure. Actor views contain only existing public role/content turns from `render_causal_variant`.
Evaluator labels, expected actions, source reports and review records remain in `evaluator`.

Coverage is complete only when included reviewed candidates cover `invariance`, `sensitivity`,
`benign_assistance`, and `legitimate_refusal`. Missing tags are listed explicitly. Coverage is an
authenticated classification claim, not a classification inferred from prompt wording.

Seven arm IDs are required: `baseline`, `synthetic`, `causal`, `causal_laundering`,
`causal_debate`, `causal_pluralistic`, and `knowledge`. Synthetic uses baseline as control;
causal uses synthetic; the next three use causal. Knowledge names one preceding frozen control.
The component keys are `synthetic`, `causal_pairs`, `laundering`, `sensitive_debate`, `pluralistic`,
and `knowledge`; each value is a configuration digest or null. Shared configuration digests,
starting systems, optimizer/reward configurations and seed policies stay matched for isolation.
The host checks component-specific data rosters; hashes or arm labels alone do not prove isolation.
Each protocol candidate's baseline binds the control candidate's resulting system configuration.

Budget JSON has exactly `train` and `evaluation`; each contains nonnegative `compute`, nonblank
`unit`, integer `examples`, and integer `tokens`. Planned equality and measured actual equality
are separate; absent measurements produce null actual matching. Units are never converted.
`ready` means the offline specification and applicable knowledge gate passed; it does not mean
an experiment ran. `matched_execution` additionally requires actual matched budgets and isolation.
No field authorizes training or establishes improvement. Deployment remains `not_assessed`.

Knowledge readiness rechecks a complete curation report at the comparison's current `as_of`.
The report must have the exact curation-report fields and fixed non-authority values; its lesson
results and proposed additions must match the fresh curation result for readiness.
The manifest host freshly attests current bank/source bindings. Exposures bind the full knowledge
report, case IDs and consuming candidates/rounds; missing exposure correspondence withholds readiness.
Case ancestry is resolved through the existing partition manifest. Curation exposures and each
arm's declared data cases become PR-6 `use="proposal"` exposures, so a consumed held-out case
cannot count as an independent audit. Arm data exposures include all ancestors, apply only to the
consuming candidate (and its protocol descendants), and conservatively use ordinal zero.

## Executable authored example

Run the following complete Python example with the repository environment or installed wheel.
It calls no model or trainer. All evidence is authored fixture data; accepting callbacks below
are only for demonstrating mechanics. Replace them with independent host verification for real
records. Expected stdout is `curated withheld 0.5 complete 7`.

```python
"""Authored offline fixtures; callback acceptance is not live verification."""

# Standard library
from typing import Any

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_records import SystemConfig
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.learning_surfaces import (
    LearningSurface,
    LessonCharacteristics,
    LessonKind,
    LessonProposal,
    LessonReviewStatus,
)
from synthetic_data.curriculum_records import (
    BankSnapshot,
    CurationPolicy,
    EpisodeSource,
    EvidenceCheck,
    LessonReview,
    ReplayEvidence,
    curriculum_digest,
)


def curation_inputs() -> dict[str, Any]:
    refs = (EvidenceReference("fixture:review", EvidenceSourceKind.EXTERNAL_RECORD),)
    system = SystemConfig("fixture-model", "fixture-harness", content_digest("config"))
    policy = CurationPolicy(
        policy_id="fixture-policy",
        version="1",
        evaluator=TrustedEvaluatorContract("fixture-host", "1", "offline-review"),
    )
    episode = EpisodeSource(
        episode_id="episode-1",
        case_ids=("case-1",),
        family_id="family-1",
        parent_episode_ids=(),
        system=system,
        terminal_status="completed_success",
        source_json=canonical_json({"training_eligibility": "DEVELOPMENT", "output": "observed"}),
        evidence_refs=refs,
    )
    lesson = LessonProposal(
        lesson_id="lesson-1",
        summary="The observed task required a checksum.",
        characteristics=LessonCharacteristics(kind=LessonKind.EPISODE_FACT),
        primary_destination=LearningSurface.MEMORY,
        evidence_refs=refs,
        rationale="Supported by the retained episode.",
        reversible=True,
        review_status=LessonReviewStatus.APPROVED,
    )
    bank = BankSnapshot(bank_id="bank", version="1", entries=())
    review = LessonReview(
        lesson_digest=content_digest(lesson.to_dict()),
        bank_digest=curriculum_digest(bank),
        source_digests=(curriculum_digest(episode),),
        policy_digest=curriculum_digest(policy),
        checks=tuple(
            EvidenceCheck(name=name, status="pass", reason="Fixture check", evidence_refs=refs)
            for name in (
                "support",
                "contradiction_clear",
                "freshness",
                "reproducible",
                "information_preserved",
                "nonredundant",
            )
        ),
        replay=ReplayEvidence(
            spec_json=canonical_json({"task": "checksum"}),
            system=system,
            seed_policy="shared",
            seeds=(1,),
            results_json=canonical_json({"observed": True}),
            evidence_refs=refs,
        ),
        valid_from="2026-10-11T00:00:00Z",
        valid_until="2026-10-12T00:00:00Z",
        status="verified",
        human_required=False,
        evidence_refs=refs,
    )
    return dict(
        episodes=(episode,),
        lessons=(lesson,),
        bank=bank,
        reviews=(review,),
        policy=policy,
        as_of="2026-10-11T12:00:00Z",
    )


def boundary_inputs() -> dict[str, Any]:
    from evaluation.causal_diagnostics import evaluate_causal_suite
    from evaluation.causal_records import MetricOpportunity
    from evaluation.ladder import Severity
    from evaluation.v5_runner import plan_v5_cells
    from synthetic_data.causal_interventions import adapt_relation_pair
    from synthetic_data.curriculum_records import BoundaryCandidate, GenerationReview
    from synthetic_data.relation_flips import Relation, make_relation_pair

    inputs = curation_inputs()
    cells = plan_v5_cells(
        case_ids=(1, 14),
        stripe_ids=("NONE",),
        repeats=1,
        model_version="fixture-model",
        harness_version="fixture-harness",
    )
    candidates, reviews = [], []
    for index, tag in enumerate(
        ("invariance", "sensitivity", "benign_assistance", "legitimate_refusal")
    ):
        relation = Relation.EVALUATOR if tag == "invariance" else Relation.CONSENT
        pair = adapt_relation_pair(
            make_relation_pair(relation, enabled=True),
            before_cell=cells[0],
            after_cell=cells[1],
            seed_policy="per_arm",
            enabled=True,
        )
        source = evaluate_causal_suite(
            (pair,),
            (),
            (),
            protocol_id="fixture",
            opportunities=(
                MetricOpportunity(
                    "op",
                    pair.pair_id,
                    "required_update_success_rate",
                    Severity.ROUTINE,
                    "end",
                    "all",
                ),
            ),
            enabled=True,
        )
        candidate = BoundaryCandidate(
            candidate_id=f"candidate-{index}",
            pair=pair,
            source_json=canonical_json(source),
            row_list="rows",
            row_id_key="opportunity_id",
            row_id="op",
            case_ids=("case-1", "case-14"),
            parent_case_ids=(),
            generator_id="relation-adapter-v1",
            generator_config_json=canonical_json({"relation": relation.value}),
            coverage_tags=(tag,),
        )
        review = GenerationReview(
            candidate_digest=curriculum_digest(candidate),
            policy_digest=curriculum_digest(inputs["policy"]),
            status="verified",
            human_required=False,
            checks=tuple(
                EvidenceCheck(
                    name=name,
                    status="pass",
                    reason="Authored fixture",
                    evidence_refs=inputs["episodes"][0].evidence_refs,
                )
                for name in (
                    "expected_actions",
                    "declared_changes",
                    "case_identity",
                    "source_lineage",
                    "coverage_tags",
                    "public_prompt_separation",
                )
            ),
            evidence_refs=inputs["episodes"][0].evidence_refs,
        )
        candidates.append(candidate)
        reviews.append(review)
    return dict(
        candidates=tuple(candidates),
        reviews=tuple(reviews),
        policy=inputs["policy"],
        as_of=inputs["as_of"],
    )


def comparison_inputs() -> dict[str, Any]:
    from evaluation.improvement_records import (
        AttemptJournal,
        CandidateSpec,
        DatasetCase,
        DatasetManifest,
        ImprovementProtocol,
        MetricSpec,
        ResamplingPolicy,
        record_digest,
    )
    from synthetic_data.curriculum_records import COMPONENTS, AblationArm

    base = curation_inputs()
    refs = base["episodes"][0].evidence_refs
    manifest = DatasetManifest(
        "fixture-data",
        (
            DatasetCase("case-1", content_digest("case-1"), "family-1", "synthetic_training"),
            DatasetCase(
                "audit-ancestor",
                content_digest("ancestor"),
                "audit-family",
                "independent_audit",
                evaluated=False,
            ),
            DatasetCase(
                "audit-child",
                content_digest("child"),
                "audit-family",
                "independent_audit",
                parent_ids=("audit-ancestor",),
            ),
        ),
        refs,
        dependencies_known=True,
    )
    names = (
        "baseline",
        "synthetic",
        "causal",
        "causal_laundering",
        "causal_debate",
        "causal_pluralistic",
        "knowledge",
    )
    controls = (None, "baseline", "synthetic", "causal", "causal", "causal", "causal")
    system = base["episodes"][0].system
    systems = {name: SystemConfig(name, "fixture-harness", content_digest(name)) for name in names}
    candidates, arms = [], []
    configs = {}
    budget = canonical_json(
        {
            phase: dict(compute=10, unit="cpu-second", examples=8, tokens=80)
            for phase in ("train", "evaluation")
        }
    )
    for index, (name, control) in enumerate(zip(names, controls)):
        config = dict(configs[control]) if control else dict.fromkeys(COMPONENTS)
        if index:
            config[COMPONENTS[index - 1]] = content_digest(COMPONENTS[index - 1])
        configs[name] = config
        candidates.append(
            CandidateSpec(
                name,
                systems[control] if control else system,
                systems[name],
                content_digest(name),
                0,
            )
        )
        arms.append(
            AblationArm(
                arm_id=name,
                candidate_id=name,
                control_arm_id=control,
                components_json=canonical_json(config),
                data_case_ids=("case-1",),
                starting_system=system,
                optimizer_digest=content_digest("optimizer"),
                reward_digest=content_digest("reward"),
                seed_policy="shared",
                seeds=(7,),
                planned_budget_json=budget,
                actual_budget_json=None,
                knowledge_report_json=None,
            )
        )
    protocol = ImprovementProtocol(
        "fixture-protocol",
        record_digest(manifest),
        tuple(candidates),
        (
            MetricSpec(
                "accuracy",
                "evaluation-ladder-v1",
                "representation_accuracy",
                content_digest("rubric"),
            ),
        ),
        (),
        base["policy"].evaluator,
        refs,
        ResamplingPolicy(seed=1, resamples=20),
    )
    journal = AttemptJournal("fixture-journal", record_digest(protocol), (), refs)
    return dict(
        arms=tuple(arms),
        protocol=protocol,
        manifest=manifest,
        journal=journal,
        evidence=(),
        as_of=base["as_of"],
    )

from dataclasses import replace
from synthetic_data.curriculum_records import (
    MilestoneReview, UtilityObservation, UtilitySlot, UtilityStudy,
)
from synthetic_data.lesson_curation import (
    curate_lessons, evaluate_lesson_utility, review_milestones,
)
from synthetic_data.cumulative_curriculum import export_cumulative_curriculum
from evaluation.curriculum_comparison import audit_curriculum_comparison

def utility_study(seed_policy="shared"):
    inputs = curation_inputs()
    observations = []
    outcomes = {
        "none": (False, False),
        "lesson_only": (True, False),
        "bank_only": (False, False),
        "bank_plus_lesson": (True, False),
    }
    for condition, values in outcomes.items():
        for repeat, success in enumerate(values):
            slot = UtilitySlot(
                slot_id=f"{condition}-{repeat}",
                task_id="task",
                content_digest=content_digest("task"),
                family_id="family",
                condition=condition,
                repeat_id=repeat,
                seed=repeat,
                system=inputs["episodes"][0].system,
                rubric_digest=content_digest("rubric"),
                budget_json='{"steps":10}',
            )
            observations.append(
                UtilityObservation(
                    slot=slot,
                    status="observed",
                    success=success,
                    source_json='{"training_eligibility":"DEVELOPMENT"}',
                    evidence_refs=inputs["episodes"][0].evidence_refs,
                )
            )
    return UtilityStudy(
        study_id="study",
        lesson_digest=content_digest(inputs["lessons"][0].to_dict()),
        bank_digest=curriculum_digest(inputs["bank"]),
        seed_policy=seed_policy,
        expected_slots=tuple(o.slot for o in observations),
        observations=tuple(observations),
    )

def milestone():
    episode = curation_inputs()["episodes"][0]
    return MilestoneReview(
        milestone_id="milestone",
        episode_digest=curriculum_digest(episode),
        step_id="step-1",
        goal="Complete the authorized checksum task",
        before_json='{"checked":false}',
        after_json='{"checked":true}',
        step_completed=True,
        goal_progress=False,
        epistemic_update=None,
        final_success=False,
        goal_authorized=True,
        constraint_violation=False,
        severe_event=False,
        status="verified",
        human_required=False,
        evidence_refs=episode.evidence_refs,
    )

# These callbacks accept only our authored fixtures; they are not production authenticators.
def accept_fixture(payload):
    return True

inputs = curation_inputs()
curation = curate_lessons(**inputs, authenticate=accept_fixture, enabled=True)
expired = curate_lessons(**{**inputs, "as_of": "2026-10-12T00:00:00Z"},
                         authenticate=accept_fixture, enabled=True)
utility = evaluate_lesson_utility(utility_study(), lesson=inputs["lessons"][0],
    bank=inputs["bank"], policy=inputs["policy"], as_of=inputs["as_of"],
    authenticate=accept_fixture, enabled=True)
milestones = review_milestones(inputs["episodes"], (milestone(),),
    policy=inputs["policy"], as_of=inputs["as_of"], authenticate=accept_fixture, enabled=True)
curriculum = export_cumulative_curriculum(**boundary_inputs(),
    authenticate=accept_fixture, enabled=True)
comparison = audit_curriculum_comparison(**comparison_inputs(),
    authenticate_manifest=accept_fixture, authenticate_audit=accept_fixture, enabled=True)
example_result = dict(curation=curation, expired=expired, utility=utility,
                      milestones=milestones, curriculum=curriculum, comparison=comparison)
print(curation["lessons"][0]["disposition"], expired["lessons"][0]["disposition"],
      utility["comparisons"]["marginal"]["delta"], curriculum["coverage"]["status"],
      len(comparison["arms"]))

```


## Verification and research limits

Run `python -m pytest -q tests/test_curriculum_records.py tests/test_lesson_curation.py
 tests/test_curriculum_milestones.py tests/test_cumulative_curriculum.py
 tests/test_curriculum_comparison.py tests/test_cumulative_curriculum_integration.py` as one command.
The integration suite executes this exact example, verifies full export rejection by
`require_training_eligible`, and exercises existing PEO collect/evaluate and train/resume modes.
The record suite owns schema, bounds and detached-snapshot checks; lesson and milestone suites
own disposition and nullable-fact checks; export tests own actor separation and coverage;
comparison tests own actual-budget, control and exposure checks. Callback semantic obligations
require the host's independent evidence review described above, not a fixture pass.

[Knowledge Weaver](recommendations/RESEARCH_TRACEABILITY.md#ref-knowledge-weaver) motivates
separate standalone/marginal and information diagnostics.
[World Potential Model](recommendations/RESEARCH_TRACEABILITY.md#ref-world-potential) motivates
separate milestone judgments.
[Station](recommendations/RESEARCH_TRACEABILITY.md#ref-station) motivates reproducible episode
retention. These are repository adaptations; none of the studies is reproduced here. Software
validation does not establish usefulness on independently withheld scenarios at matched budgets.
Empirical validation, reward activation and deployment remain separate decisions.
