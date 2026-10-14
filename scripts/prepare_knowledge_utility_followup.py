"""Prepare a pending offline study; this command grants no capture or review authority."""

# Standard library
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import permutations
from typing import Any

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from scripts.knowledge_utility_followup_tasks import AuthoredTask, authored_tasks, validate_tasks
from scripts.prepare_knowledge_utility_pilot import prepare_pilot

SCHEDULE_KEY = "knowledge-utility-followup-order-v1"
ORIGIN_DIGEST = "637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1"
MAX_PACKET_BYTES = 4 * 1024 * 1024
CONDITIONS = ("bank_only", "bank_plus_lesson", "lesson_only", "none")
METRICS = ("answer_correct", "format_valid", "joint_success")
BASE_INSTRUCTION = (
    "Solve the task using its explicit facts and rules. Return only JSON with one key, "
    '"answer", whose value is exactly "A", "B", "C", or "D". '
    "Do not add explanation or other keys."
)


def _hash_order(suffix: str) -> str:
    """Make the declared schedule reproducible without claiming a generation seed."""
    return sha256((SCHEDULE_KEY + "|" + suffix).encode("utf-8")).hexdigest()


def _case(task: AuthoredTask, index: int) -> dict[str, Any]:
    """Move each distractor's explanation with its text when assigning option labels."""
    correct_label = "ABCD"[index % 4]
    alternatives = iter(task.distractors)
    options, errors = {}, {}
    for label in "ABCD":
        if label == correct_label:
            options[label] = task.correct_text
        else:
            alternative = next(alternatives)
            options[label] = alternative.text
            errors[label] = alternative.error
    return dict(
        task_id=task.task_id,
        mechanism_id=task.mechanism_id,
        variant=task.variant,
        prompt=task.prompt,
        options=options,
        expected_option=correct_label,
        derivation=list(task.derivation),
        distractor_errors=errors,
        variant_rationale=task.variant_rationale,
        task_digest=content_digest(dict(prompt=task.prompt, options=options)),
        dependency_group=None,
        dependencies_known=False,
        review_status="pending",
    )


def _rubric() -> dict[str, Any]:
    """Declare the proposed separate outcomes without evaluating any response."""
    return dict(
        version="followup-separate-outcomes-v1",
        allowed_options=list("ABCD"),
        format_rule="single_json_answer_option",
        duplicate_keys="reject",
        noncanonical_rule="two_blinded_reviews_agree",
        unresolved_rule="withhold_semantic_rate",
        noncompleted_rule="null_metrics",
        contrasts=dict(
            standalone=["lesson_only", "none"], marginal=["bank_plus_lesson", "bank_only"]
        ),
        metrics=list(METRICS),
    )


def _review_templates(cases: list[dict[str, Any]], rubric: dict[str, Any]) -> dict[str, Any]:
    """Leave reviewer identity and judgments unfilled rather than inventing evidence."""
    checks = ("answer", "ambiguity", "variant_structure", "source_overlap", "dependency_group")
    return dict(
        task_reviews=[
            dict(
                task_id=case["task_id"],
                task_digest=case["task_digest"],
                status="pending",
                reviewer_id=None,
                checks=dict.fromkeys(checks),
                evidence_refs=[],
            )
            for case in cases
        ],
        response_review=dict(
            review_id=None,
            task_digest=None,
            response_digest=None,
            rubric_digest=content_digest(rubric),
            reviewer_id=None,
            selected_option=None,
            status="pending",
            evidence_refs=[],
        ),
    )


def _report_template() -> dict[str, Any]:
    """Keep unexecuted outcomes null, including counts that could imply an attempted run."""
    counts = (
        "completed",
        "failed",
        "censored",
        "missing",
        "correct",
        "incorrect",
        "unresolved",
        "format_valid",
        "joint_success",
    )
    return dict(
        execution_state="not_started",
        expected_slots=384,
        conditions={
            condition: dict(expected=96, **dict.fromkeys(counts), rates=dict.fromkeys(METRICS))
            for condition in CONDITIONS
        },
        comparisons={name: dict.fromkeys(METRICS) for name in ("standalone", "marginal")},
        observations=[],
        verified=False,
        paired=False,
        interval=None,
        reasons=["not_executed", "seed_policy_unknown", "study_unauthenticated"],
    )


def _requests(
    cases: list[dict[str, Any]], knowledge: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project only public text into messages, keeping schedule identities outside them."""
    lesson = knowledge["lesson"]["summary"]
    bank = knowledge["bank"]["entries"][0]["lesson"]["summary"]
    contexts = {
        "none": "",
        "lesson_only": lesson,
        "bank_only": bank,
        "bank_plus_lesson": bank + "\n" + lesson,
    }
    by_id = {case["task_id"]: case for case in cases}
    blocks = [(case["task_id"], repeat) for case in cases for repeat in (0, 1)]
    blocks.sort(key=lambda block: (_hash_order(f"block|{block[0]}|{block[1]}"), *block))
    orders = tuple(permutations(CONDITIONS))
    slots: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for index, (task_id, repeat) in enumerate(blocks):
        case = by_id[task_id]
        public = (
            case["prompt"]
            + "\nOptions:\n"
            + "\n".join(f"{label}: {text}" for label, text in case["options"].items())
        )
        for condition in orders[index % len(orders)]:
            slot = dict(
                sequence=len(slots),
                slot_id=f"{task_id}:{repeat}:{condition}",
                task_id=task_id,
                repeat=repeat,
                condition=condition,
                generation_seed=None,
            )
            instruction = BASE_INSTRUCTION
            if contexts[condition]:
                instruction += "\nKnowledge context:\n" + contexts[condition]
            slots.append(slot)
            requests.append(
                dict(
                    **slot,
                    messages=[
                        dict(role="system", content=instruction),
                        dict(role="user", content=public),
                    ],
                )
            )
    return slots, requests


def prepare_followup() -> dict[str, Any]:
    """Return fresh deterministic review material, without freezing or executing the study."""
    tasks = authored_tasks()
    validate_tasks(tasks)
    ordered = sorted(tasks, key=lambda task: (_hash_order("task|" + task.task_id), task.task_id))
    cases = [_case(task, index) for index, task in enumerate(ordered)]
    prior = prepare_pilot()
    digest = prior.pop("packet_digest")
    if digest != ORIGIN_DIGEST or content_digest(prior) != ORIGIN_DIGEST:
        raise ValueError("original pilot content changed")
    knowledge = dict(
        origin_packet_digest=digest,
        **{key: deepcopy(prior[key]) for key in ("source_episodes", "lesson", "bank")},
    )
    slots, requests = _requests(cases, knowledge)
    rubric = _rubric()
    packet = dict(
        schema_version="knowledge-utility-followup-packet-v1",
        study_id="knowledge-utility-followup-v1",
        as_of="2026-10-13T00:00:00Z",
        status="pending_independent_review",
        execution_authorized=False,
        packet_frozen=False,
        optimizer_input=False,
        confers_authority=False,
        training_eligibility="DEVELOPMENT",
        model=dict(
            model="gpt-5.4-mini-2026-03-17",
            reasoning_effort="none",
            temperature=1,
            generation_seed=None,
            availability_checked=False,
        ),
        limits=dict(
            generation_requests=384,
            automatic_retries=0,
            max_input_tokens=2048,
            max_output_tokens=256,
            tools=[],
            approved_generation_budget_usd=None,
        ),
        knowledge=knowledge,
        rubric=rubric,
        schedule=dict(
            key=SCHEDULE_KEY,
            assignment_policy="balanced_deterministic",
            generation_seed_policy="unknown",
            block_count=96,
            slots=slots,
        ),
        evaluator_cases=cases,
        actor_requests=requests,
        review_templates=_review_templates(cases, rubric),
        report_template=_report_template(),
    )
    packet["packet_digest"] = content_digest(packet)
    if len(canonical_json(packet).encode("utf-8")) > MAX_PACKET_BYTES:
        raise ValueError("follow-up packet exceeds byte limit")
    return packet
