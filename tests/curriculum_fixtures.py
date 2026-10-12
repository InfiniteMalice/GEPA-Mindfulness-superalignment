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
