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
