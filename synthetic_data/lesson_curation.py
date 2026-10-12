"""Pure offline lesson, utility and milestone diagnostics with explicit host trust."""

# Standard library
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.learning_surfaces import LessonProposal, LessonReviewStatus

from .curriculum_records import (
    BankEntry,
    BankSnapshot,
    CurationPolicy,
    EpisodeSource,
    JsonObject,
    LessonReview,
    MilestoneReview,
    UtilityStudy,
    curriculum_digest,
    unique,
    utc_time,
    validate_curriculum_envelope,
    validate_episode_roster,
)

HostCheck = Callable[[JsonObject], bool]
LESSON_CHECKS = (
    "support",
    "contradiction_clear",
    "freshness",
    "reproducible",
    "information_preserved",
    "nonredundant",
)
CONDITIONS = ("none", "lesson_only", "bank_only", "bank_plus_lesson")


def require_enabled(enabled: bool) -> None:
    """Reject implicit opt-in before inspecting inputs or invoking a callback."""
    if enabled is not True:
        raise ValueError("offline curriculum operation requires enabled=True")


def snapshot(value: Any, kind: type) -> Any:
    """Revalidate and detach an exact repository record through its public schema."""
    if type(value) is not kind:
        raise ValueError(f"expected exact {kind.__name__}")
    return kind.from_dict(cast(Any, value).to_dict())


def snapshots(values: tuple[Any, ...], kind: type) -> tuple[Any, ...]:
    """Bound a typed collection before any caller-owned record is consumed."""
    if type(values) is not tuple or len(values) > 1024:
        raise ValueError("records require a bounded tuple")
    return tuple(snapshot(value, kind) for value in values)


def authenticate_payload(payload: JsonObject, authenticate: HostCheck | None) -> bool:
    """Require exact True over an unchanged detached complete payload; never persist trust."""
    validate_curriculum_envelope(payload)
    original = canonical_json(payload)
    detached = json.loads(original)
    if authenticate is None:
        return False
    try:
        accepted = authenticate(detached)
        validate_curriculum_envelope(detached)
        return accepted is True and canonical_json(detached) == original
    except Exception:
        return False


def restricted_origin(payload: object) -> bool:
    """Find held-out source labels through nested objects and encoded source envelopes."""
    validate_curriculum_envelope(payload)
    pending = [payload]
    while pending:
        value = pending.pop()
        if type(value) is dict:
            for key, item in value.items():
                if key == "training_eligibility" and item == "HIDDEN_EVAL":
                    return True
                if key == "purpose" and item in (
                    "independent_audit",
                    "final_test",
                    "ood_combinations",
                ):
                    return True
                if key == "holdout_status" and item not in (None, False, "none", "not_held_out"):
                    return True
                pending.append(item)
        elif type(value) is list:
            pending.extend(value)
        elif type(value) is str and value.lstrip().startswith(("{", "[", '"')):
            try:
                decoded = json.loads(value)
            except ValueError:
                continue
            pending.append(decoded)
    return False


def report_document(schema: str, inputs: JsonObject, **results: Any) -> JsonObject:
    """Seal a diagnostic report without adding a receipt or an optimizer interface."""
    report = dict(
        schema_version=schema,
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        inputs=inputs,
        **results,
    )
    report["result_digest"] = content_digest(report)
    return report


def _payload(purpose: str, policy: CurationPolicy, as_of: str, subject: JsonObject) -> JsonObject:
    return dict(purpose=purpose, policy=policy.to_dict(), as_of=as_of, subject=subject)


def _content(lesson: LessonProposal) -> str:
    return content_digest([lesson.summary, lesson.characteristics.to_dict()])


def curate_lessons(
    *,
    episodes: tuple[EpisodeSource, ...],
    lessons: tuple[LessonProposal, ...],
    bank: BankSnapshot,
    reviews: tuple[LessonReview, ...],
    policy: CurationPolicy,
    as_of: str,
    authenticate: HostCheck | None = None,
    enabled: bool = False,
) -> JsonObject:
    """Return supported offline additions; keep all exclusions and original review statuses."""
    require_enabled(enabled)
    now = utc_time(as_of)
    episodes = snapshots(episodes, EpisodeSource)
    lessons = snapshots(lessons, LessonProposal)
    reviews = snapshots(reviews, LessonReview)
    bank, policy = snapshot(bank, BankSnapshot), snapshot(policy, CurationPolicy)
    validate_episode_roster(episodes)
    unique(lessons, "lesson_id")
    unique(reviews, "lesson_digest")
    episode_ids = {e.episode_id: e for e in episodes}
    if any(
        source not in episode_ids for entry in bank.entries for source in entry.source_episode_ids
    ):
        raise ValueError("unknown bank source episode")
    inputs = dict(
        episodes=[e.to_dict() for e in episodes],
        lessons=[item.to_dict() for item in lessons],
        bank=bank.to_dict(),
        reviews=[r.to_dict() for r in reviews],
        policy=policy.to_dict(),
        as_of=as_of,
    )
    validate_curriculum_envelope(inputs)
    by_digest = {curriculum_digest(e): e for e in episodes}
    by_review = {r.lesson_digest: r for r in reviews}
    counts = Counter(_content(item) for item in lessons)
    counts.update(_content(e.lesson) for e in bank.entries)
    rows, additions = [], []
    for lesson in sorted(lessons, key=lambda item: item.lesson_id):
        review = by_review.get(content_digest(lesson.to_dict()))
        reasons: list[str] = []
        authenticated = False
        failed = lesson.review_status is LessonReviewStatus.REJECTED
        if lesson.review_status is not LessonReviewStatus.APPROVED:
            reasons.append("proposal_" + lesson.review_status.value)
        if counts[_content(lesson)] > 1:
            reasons.append("duplicate_content")
            failed = True
        selected: dict[str, EpisodeSource] = {}
        checks: list[JsonObject] = []
        if review is None:
            reasons.append("review_missing")
        else:
            authenticated = authenticate_payload(
                _payload(
                    "lesson_review",
                    policy,
                    as_of,
                    {**inputs, "lesson": lesson.to_dict(), "review": review.to_dict()},
                ),
                authenticate,
            )
            if not authenticated:
                reasons.append("review_unauthenticated")
            if review.bank_digest != curriculum_digest(bank):
                reasons.append("bank_binding_stale")
            if review.policy_digest != curriculum_digest(policy):
                reasons.append("policy_binding_stale")
            if review.status != "verified":
                reasons.append("review_" + review.status)
            if review.human_required:
                reasons.append("human_review_required")
            if not utc_time(review.valid_from) <= now < utc_time(review.valid_until):
                reasons.append("review_outside_validity")
            if not review.source_digests or any(d not in by_digest for d in review.source_digests):
                reasons.append("source_binding_missing")
            selected = {
                by_digest[d].episode_id: by_digest[d]
                for d in review.source_digests
                if d in by_digest
            }
            pending = list(selected.values())
            while pending:
                for parent in pending.pop().parent_episode_ids:
                    if parent not in selected:
                        selected[parent] = episode_ids[parent]
                        pending.append(episode_ids[parent])
            if any(
                e.terminal_status not in ("completed_success", "completed_failure")
                for e in selected.values()
            ):
                reasons.append("episode_incomplete")
            if restricted_origin([e.to_dict() for e in selected.values()] + [bank.to_dict()]):
                reasons.append("held_out_source")
            if review.replay is None:
                reasons.append("replay_missing")
            elif (
                review.replay.seed_policy == "unknown"
                or not review.replay.seeds
                or None in review.replay.seeds
            ):
                reasons.append("replay_seed_unknown")
            checks = [c.to_dict() for c in review.checks]
            mapping = {c.name: c for c in review.checks}
            if set(mapping) != set(LESSON_CHECKS):
                reasons.append("check_roster_mismatch")
            for check in review.checks:
                if check.status != "pass":
                    reasons.append(f"check_{check.name}_{check.status}")
                failed |= authenticated and check.status == "fail"
        disposition = "rejected" if failed else "withheld" if reasons else "curated"
        rows.append(
            dict(
                lesson_id=lesson.lesson_id,
                disposition=disposition,
                reasons=sorted(set(reasons)),
                authentication=authenticated,
                checks=checks,
            )
        )
        if disposition == "curated":
            assert review is not None
            additions.append(
                BankEntry(
                    lesson=lesson,
                    version="1",
                    source_episode_ids=tuple(sorted(selected)),
                    source_json=canonical_json(
                        {
                            "training_eligibility": "DEVELOPMENT",
                            "review": review.to_dict(),
                            "episodes": [selected[k].to_dict() for k in sorted(selected)],
                        }
                    ),
                ).to_dict()
            )
    return report_document(
        "lesson-curation-v1", inputs, lessons=rows, proposed_bank_additions=additions
    )


def evaluate_lesson_utility(
    study: UtilityStudy,
    *,
    lesson: LessonProposal,
    bank: BankSnapshot,
    policy: CurationPolicy,
    as_of: str,
    authenticate: HostCheck | None = None,
    enabled: bool = False,
) -> JsonObject:
    """Report measured condition rates and matched contrasts separately from factual support."""
    require_enabled(enabled)
    utc_time(as_of)
    study, lesson = snapshot(study, UtilityStudy), snapshot(lesson, LessonProposal)
    bank, policy = snapshot(bank, BankSnapshot), snapshot(policy, CurationPolicy)
    inputs = dict(
        study=study.to_dict(),
        lesson=lesson.to_dict(),
        bank=bank.to_dict(),
        policy=policy.to_dict(),
        as_of=as_of,
    )
    valid = authenticate_payload(_payload("utility_study", policy, as_of, inputs), authenticate)
    reasons = []
    if not valid:
        reasons.append("study_unauthenticated")
    if study.lesson_digest != content_digest(lesson.to_dict()):
        reasons.append("lesson_binding_stale")
    if study.bank_digest != curriculum_digest(bank):
        reasons.append("bank_binding_stale")
    if study.seed_policy == "unknown":
        reasons.append("seed_policy_unknown")
    observed = {o.slot.slot_id: o for o in study.observations}
    conditions: JsonObject = {}
    matched: dict[str, dict[str, Any]] = {}
    rows = []
    for condition in CONDITIONS:
        slots = [s for s in study.expected_slots if s.condition == condition]
        counts: Counter[str] = Counter()
        successes = 0
        strata: dict[str, Any] = {}
        for slot in slots:
            obs = observed.get(slot.slot_id)
            status = obs.status if obs else "missing"
            counts[status] += 1
            rows.append(
                dict(slot=slot.to_dict(), status=status, success=obs.success if obs else None)
            )
            data = slot.to_dict()
            for field in ("schema_version", "slot_id", "condition", "seed"):
                data.pop(field)
            key = canonical_json(data)
            if key in strata:
                raise ValueError("duplicate utility stratum in a condition")
            strata[key] = obs
            if status == "observed":
                successes += int(obs is not None and obs.success is True)
        conditions[condition] = dict(
            expected=len(slots),
            successes=successes,
            rate=successes / counts["observed"] if counts["observed"] else None,
            **{name: counts[name] for name in ("observed", "missing", "failed", "censored")},
        )
        matched[condition] = strata
    comparisons = {}
    for name, left, right in (
        ("standalone", "none", "lesson_only"),
        ("marginal", "bank_only", "bank_plus_lesson"),
    ):
        first, second = matched[left], matched[right]
        issues = list(reasons)
        if not first or set(first) != set(second):
            issues.append("strata_mismatch")
        deltas = []
        for key in sorted(set(first) & set(second)):
            a, b = first[key], second[key]
            if a is None or b is None or a.status != "observed" or b.status != "observed":
                issues.append("incomplete_observations")
                continue
            deltas.append(int(b.success) - int(a.success))
            if a.slot.seed is None or b.slot.seed is None:
                issues.append("seed_missing")
            elif study.seed_policy == "shared" and a.slot.seed != b.slot.seed:
                issues.append("seed_mismatch")
        comparisons[name] = dict(
            delta=sum(deltas) / len(deltas) if deltas else None,
            matched_count=len(deltas),
            verified=bool(deltas) and not issues,
            paired=study.seed_policy == "shared" and not issues,
            reasons=sorted(set(issues)),
        )
    return report_document(
        "lesson-utility-v1",
        inputs,
        conditions=conditions,
        observations=rows,
        comparisons=comparisons,
        interval=None,
        interval_reason="unsupported_source_for_interval",
    )


def review_milestones(
    episodes: tuple[EpisodeSource, ...],
    milestones: tuple[MilestoneReview, ...],
    *,
    policy: CurationPolicy,
    as_of: str,
    authenticate: HostCheck | None = None,
    enabled: bool = False,
) -> JsonObject:
    """Keep step completion, legitimate progress, updating and final success independent."""
    require_enabled(enabled)
    utc_time(as_of)
    episodes = snapshots(episodes, EpisodeSource)
    milestones = snapshots(milestones, MilestoneReview)
    policy = snapshot(policy, CurationPolicy)
    validate_episode_roster(episodes)
    unique(milestones, "milestone_id")
    inputs = dict(
        episodes=[e.to_dict() for e in episodes],
        milestones=[m.to_dict() for m in milestones],
        policy=policy.to_dict(),
        as_of=as_of,
    )
    validate_curriculum_envelope(inputs)
    sources = {curriculum_digest(e): e for e in episodes}
    dimensions = ("step_completed", "goal_progress", "epistemic_update", "final_success")
    rows = []
    for milestone in sorted(milestones, key=lambda item: item.milestone_id):
        claims = milestone.to_dict()
        authenticated = authenticate_payload(
            _payload("milestone_review", policy, as_of, {**inputs, "milestone": claims}),
            authenticate,
        )
        reasons = []
        if not authenticated:
            reasons.append("review_unauthenticated")
        episode = sources.get(milestone.episode_digest)
        if episode is None:
            reasons.append("source_binding_missing")
        elif episode.terminal_status not in ("completed_success", "completed_failure"):
            reasons.append("episode_incomplete")
        if milestone.status != "verified":
            reasons.append("review_" + milestone.status)
        if milestone.human_required:
            reasons.append("human_review_required")
        verified = {name: getattr(milestone, name) if not reasons else None for name in dimensions}
        if (
            milestone.goal_authorized is not True
            or milestone.constraint_violation is not False
            or milestone.severe_event is not False
        ):
            verified["goal_progress"] = verified["final_success"] = None
            reasons.append("legitimate_goal_or_constraints_unverified")
        rows.append(
            dict(
                milestone_id=milestone.milestone_id,
                claims=claims,
                verified=verified,
                authentication=authenticated,
                reasons=reasons,
            )
        )
    return report_document("curriculum-milestones-v1", inputs, milestones=rows)
