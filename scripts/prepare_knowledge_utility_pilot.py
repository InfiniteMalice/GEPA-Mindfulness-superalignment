"""Prepare a fixed, offline knowledge-utility pilot without granting execution authority."""

# Standard library
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_audit import validate_manifest
from evaluation.improvement_records import DatasetCase, DatasetManifest, SystemConfig
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
    BankEntry,
    BankSnapshot,
    CurationPolicy,
    EpisodeSource,
    UtilitySlot,
    UtilityStudy,
    curriculum_digest,
)
from synthetic_data.lesson_curation import CONDITIONS, curate_lessons, evaluate_lesson_utility

AS_OF = "2026-10-12T00:00:00Z"
BASE_INSTRUCTION = (
    "Solve the task using its explicit facts and rules. Return only JSON with one key, "
    '"answer", whose value is a string. Do not add explanation or other keys.'
)


def _cases() -> list[dict[str, str]]:
    """Keep authored public prompts beside evaluator-only answers for offline review."""
    definitions = (
        ("inventory", "Start with 11 items, receive 4, then ship 9. How many remain?", "6"),
        ("precedence", "A is before B and B is before C. Which is earliest: A, B, or C?", "A"),
        (
            "route_cost",
            "DIRECT costs 9. VIA costs 3 plus 4. Which route costs less: DIRECT or VIA?",
            "VIA",
        ),
        (
            "intervals",
            "Intervals include their lower bound and exclude their upper bound. "
            "Do [2,5) and [5,8) overlap? Answer OVERLAP or DISJOINT.",
            "DISJOINT",
        ),
        (
            "quorum",
            "A proposal passes with at least 4 yes votes. Of 5 votes, 3 are yes. "
            "Answer PASS or FAIL.",
            "FAIL",
        ),
        (
            "units",
            "Is 3 minutes shorter, equal to, or longer than 170 seconds? "
            "Answer SHORTER, EQUAL, or LONGER.",
            "LONGER",
        ),
    )
    return [
        dict(case_id=f"pilot-{family}", family_id=family, prompt=prompt, expected_answer=answer)
        for family, prompt, answer in definitions
    ]


def _knowledge() -> tuple[tuple[EpisodeSource, ...], LessonProposal, BankSnapshot]:
    """Retain authored demonstrations and pending hypotheses without faking model captures."""
    demonstrations = (
        (
            "source_rectangle",
            "A rectangle has sides 4 and 7. What is its perimeter?",
            "22",
            "Compute required intermediate quantities from the stated facts before selecting "
            "an answer; incidental wording cannot replace the task's explicit rule.",
        ),
        (
            "source_access",
            "Entry requires both a badge and a reservation. Sam has a badge only. "
            "Is entry permitted? Answer YES or NO.",
            "NO",
            "Check every explicit condition and boundary in the current task's rule; "
            "do not infer an exception from a partially satisfied rule.",
        ),
    )
    system = SystemConfig(
        "authored-demonstration", "pilot-source-v1", content_digest(demonstrations)
    )
    episodes, lessons = [], []
    for family, prompt, answer, summary in demonstrations:
        refs = (
            EvidenceReference(f"authored:{family}:output", EvidenceSourceKind.OBSERVABLE_OUTPUT),
        )
        source = dict(
            origin="authored_demonstration",
            training_eligibility="DEVELOPMENT",
            prompt=prompt,
            output={"answer": answer},
        )
        episodes.append(
            EpisodeSource(
                episode_id=f"episode-{family}",
                case_ids=(f"case-{family}",),
                family_id=family,
                parent_episode_ids=(),
                system=system,
                terminal_status="completed_success",
                source_json=canonical_json(source),
                evidence_refs=refs,
            )
        )
        lessons.append(
            LessonProposal(
                lesson_id=f"lesson-{family}",
                summary=summary,
                characteristics=LessonCharacteristics(kind=LessonKind.EPISODE_FACT),
                primary_destination=LearningSurface.MEMORY,
                evidence_refs=refs,
                rationale="Proposed generalization from an authored demonstration; support, "
                "scope and nonredundancy require independent review.",
                reversible=True,
                review_status=LessonReviewStatus.PENDING,
            )
        )
    entry = BankEntry(
        lesson=lessons[1],
        version="1",
        source_episode_ids=(episodes[1].episode_id,),
        source_json=canonical_json(
            {"training_eligibility": "DEVELOPMENT", "episodes": [episodes[1].to_dict()]}
        ),
    )
    return (
        tuple(episodes),
        lessons[0],
        BankSnapshot(bank_id="proposed-pilot-bank", version="1", entries=(entry,)),
    )


def prepare_pilot() -> dict[str, Any]:
    """Return deterministic review material, empty observations and withheld diagnostics."""
    episodes, lesson, bank = _knowledge()
    cases = _cases()
    limits = dict(
        generation_requests=72,
        automatic_retries=0,
        max_input_tokens=2048,
        max_output_tokens=256,
        generation_cost_ceiling_usd=5,
        tools=[],
    )
    model: dict[str, Any] = dict(
        model="gpt-5.4-mini-2026-03-17",
        reasoning_effort="none",
        temperature=1,
        generation_seed=None,
        availability_checked=False,
    )
    rubric = dict(
        version="pilot-exact-answer-v1",
        rule="Parse exactly one JSON object with exactly the key answer and a string value; "
        "success requires exact equality with expected_answer. Reject duplicate keys, "
        "trailing non-whitespace text, markdown fences and additional fields. "
        "Do not trim or normalize the answer string.",
        outcome_policy={
            "observed": "Completed answer; success is exact-match truth, including False "
            "for invalid format or wrong answer.",
            "failed": "Transport or provider error; success is null.",
            "censored": "Refusal or truncation; success is null, even if partial text parses.",
            "missing": "Unsent request or absent capture; success is null.",
        },
    )
    system = SystemConfig(
        model["model"],
        "knowledge-utility-pilot-preparation-v1",
        content_digest(dict(model=model, limits=limits, base_instruction=BASE_INSTRUCTION)),
    )
    policy = CurationPolicy(
        policy_id="proposed-pilot-review",
        version="1",
        evaluator=TrustedEvaluatorContract("unassigned-reviewer", "1", "offline-pilot-review"),
    )
    contexts = {
        "none": "",
        "lesson_only": lesson.summary,
        "bank_only": bank.entries[0].lesson.summary,
        "bank_plus_lesson": bank.entries[0].lesson.summary + "\n" + lesson.summary,
    }
    slots, requests = [], []
    # Rotate condition order to avoid always running the same condition first.
    for repeat in range(3):
        for index, case in enumerate(cases):
            offset = (repeat + index) % len(CONDITIONS)
            order = CONDITIONS[offset:] + CONDITIONS[:offset]
            for condition in order:
                slot = UtilitySlot(
                    slot_id=f"{case['case_id']}:{repeat}:{condition}",
                    task_id=case["case_id"],
                    content_digest=content_digest(case["prompt"]),
                    family_id=case["family_id"],
                    condition=condition,
                    repeat_id=repeat,
                    seed=None,
                    system=system,
                    rubric_digest=content_digest(
                        dict(rubric=rubric, expected_answer=case["expected_answer"])
                    ),
                    budget_json=canonical_json(limits),
                )
                slots.append(slot)
                instruction = BASE_INSTRUCTION
                if contexts[condition]:
                    instruction += "\nKnowledge context:\n" + contexts[condition]
                requests.append(
                    dict(
                        slot_id=slot.slot_id,
                        case_id=case["case_id"],
                        condition=condition,
                        messages=[
                            dict(role="system", content=instruction),
                            dict(role="user", content=case["prompt"]),
                        ],
                    )
                )
    study = UtilityStudy(
        study_id="knowledge-utility-pilot-v1",
        lesson_digest=content_digest(lesson.to_dict()),
        bank_digest=curriculum_digest(bank),
        seed_policy="unknown",
        expected_slots=tuple(slots),
        observations=(),
    )
    manifest = DatasetManifest(
        dataset_id="knowledge-utility-pilot-v1",
        cases=tuple(
            DatasetCase(
                e.case_ids[0],
                content_digest(e.source_json),
                e.family_id,
                "synthetic_training",
                evaluated=False,
            )
            for e in episodes
        )
        + tuple(
            DatasetCase(
                c["case_id"],
                content_digest(c["prompt"]),
                c["family_id"],
                "independent_audit",
                evaluated=True,
            )
            for c in cases
        ),
        dependencies_known=False,
    )
    validate_manifest(manifest)
    packet: dict[str, Any] = dict(
        schema_version="knowledge-utility-pilot-preparation-v1",
        as_of=AS_OF,
        execution_authorized=False,
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        model=model,
        limits=limits,
        rubric=rubric,
        source_episodes=[e.to_dict() for e in episodes],
        lesson=lesson.to_dict(),
        bank=bank.to_dict(),
        policy=policy.to_dict(),
        manifest=manifest.to_dict(),
        evaluator_cases=cases,
        actor_requests=requests,
        utility_study=study.to_dict(),
        curation_preview=curate_lessons(
            episodes=episodes,
            lessons=(lesson,),
            bank=bank,
            reviews=(),
            policy=policy,
            as_of=AS_OF,
            enabled=True,
        ),
        utility_preview=evaluate_lesson_utility(
            study, lesson=lesson, bank=bank, policy=policy, as_of=AS_OF, enabled=True
        ),
    )
    packet["packet_digest"] = content_digest(packet)
    return packet


def main(argv: list[str] | None = None) -> None:
    """Write a new UTF-8 packet; refuse to overwrite any existing path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON output path; the parent directory must already exist.",
    )
    args = parser.parse_args(argv)
    text = canonical_json(prepare_pilot()) + "\n"
    try:
        stream = args.output.open("x", encoding="utf-8")
    except FileExistsError:
        raise
    except OSError as error:
        parser.error(f"cannot write {args.output}: {error}")
    try:
        with stream:
            stream.write(text)
    except OSError as error:
        # Only clean up after this invocation successfully created the output.
        try:
            args.output.unlink(missing_ok=True)
        except OSError as cleanup_error:
            parser.error(f"cannot write {args.output}: {error}; cleanup failed: {cleanup_error}")
        parser.error(f"cannot write {args.output}: {error}")


if __name__ == "__main__":
    main()
