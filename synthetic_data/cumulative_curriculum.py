"""Export reviewed offline boundary examples with evaluator provenance kept separate."""

# Standard library
from __future__ import annotations

# Third-party
# Local
from evaluation.causal_records import content_digest

from .causal_interventions import render_causal_variant
from .curriculum_records import (
    BoundaryCandidate,
    CurationPolicy,
    GenerationReview,
    JsonObject,
    curriculum_digest,
    read_curriculum_object,
    unique,
    utc_time,
    validate_curriculum_envelope,
)
from .lesson_curation import (
    HostCheck,
    authenticate_payload,
    report_document,
    require_enabled,
    restricted_origin,
    snapshot,
    snapshots,
)

GENERATION_CHECKS = (
    "expected_actions",
    "declared_changes",
    "case_identity",
    "source_lineage",
    "coverage_tags",
    "public_prompt_separation",
)
COVERAGE_TAGS = ("invariance", "sensitivity", "benign_assistance", "legitimate_refusal")


def _source(candidate: BoundaryCandidate) -> tuple[JsonObject, JsonObject]:
    source = read_curriculum_object(candidate.source_json)
    selectors = {
        "causal-diagnostics-v1": ("rows", "opportunity_id"),
        "sensitive-debate-report-v1": ("metric_rows", "opportunity_id"),
    }
    if selectors.get(source.get("schema_version", "")) != (
        candidate.row_list,
        candidate.row_id_key,
    ):
        raise ValueError("unsupported source schema or selector")
    if source.get("result_digest") != content_digest(
        {k: v for k, v in source.items() if k != "result_digest"}
    ):
        raise ValueError("source result digest mismatch")
    if source.get("confers_authority") is not False or source.get("training_eligibility") not in (
        "DEVELOPMENT",
        "REGRESSION",
        "HIDDEN_EVAL",
    ):
        raise ValueError("source must retain diagnostic restrictions")
    rows = source.get(candidate.row_list)
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError("source requires a row list")
    selected = [row for row in rows if row.get(candidate.row_id_key) == candidate.row_id]
    if len(selected) != 1:
        raise ValueError("source row must resolve uniquely")
    return source, selected[0]


def export_cumulative_curriculum(
    candidates: tuple[BoundaryCandidate, ...],
    reviews: tuple[GenerationReview, ...],
    *,
    policy: CurationPolicy,
    as_of: str,
    authenticate: HostCheck | None = None,
    enabled: bool = False,
) -> JsonObject:
    """Export only independently reviewed candidates; incomplete coverage stays explicit."""
    require_enabled(enabled)
    utc_time(as_of)
    candidates = snapshots(candidates, BoundaryCandidate)
    reviews = snapshots(reviews, GenerationReview)
    policy = snapshot(policy, CurationPolicy)
    unique(candidates, "candidate_id")
    unique(reviews, "candidate_digest")
    inputs = dict(
        candidates=[c.to_dict() for c in sorted(candidates, key=lambda c: c.candidate_id)],
        reviews=[r.to_dict() for r in sorted(reviews, key=lambda r: r.candidate_digest)],
        policy=policy.to_dict(),
        as_of=as_of,
    )
    validate_curriculum_envelope(inputs)
    review_map = {r.candidate_digest: r for r in reviews}
    examples, excluded = [], []
    coverage: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        source, row = _source(candidate)
        review = review_map.get(curriculum_digest(candidate))
        reasons = []
        if restricted_origin(candidate.to_dict()):
            reasons.append("held_out_source")
        if review is None:
            reasons.append("review_missing")
        else:
            authenticated = authenticate_payload(
                dict(
                    purpose="generation_review",
                    policy=policy.to_dict(),
                    as_of=as_of,
                    subject=dict(candidate=candidate.to_dict(), review=review.to_dict()),
                ),
                authenticate,
            )
            if not authenticated:
                reasons.append("review_unauthenticated")
            if review.policy_digest != curriculum_digest(policy):
                reasons.append("policy_binding_stale")
            if review.status != "verified" or review.human_required:
                reasons.append("review_unresolved")
            if {c.name for c in review.checks} != set(GENERATION_CHECKS):
                reasons.append("check_roster_mismatch")
            reasons.extend(
                f"check_{c.name}_{c.status}" for c in review.checks if c.status != "pass"
            )
        if reasons:
            excluded.append(dict(candidate_id=candidate.candidate_id, reasons=sorted(set(reasons))))
            continue
        assert review is not None
        actor = {
            arm: list(
                render_causal_variant(
                    candidate.pair, getattr(candidate.pair, arm).variant_id, enabled=True
                )
            )
            for arm in ("before", "after")
        }
        examples.append(
            dict(
                candidate_id=candidate.candidate_id,
                actor=actor,
                evaluator=dict(
                    pair=candidate.pair.to_dict(),
                    source=source,
                    source_row=row,
                    generation=candidate.to_dict(),
                    review=review.to_dict(),
                ),
                training_eligibility="DEVELOPMENT",
                optimizer_input=False,
                confers_authority=False,
            )
        )
        coverage.update(candidate.coverage_tags)
    missing = sorted(set(COVERAGE_TAGS) - coverage)
    return report_document(
        "cumulative-curriculum-v1",
        inputs,
        examples=examples,
        excluded=excluded,
        coverage=dict(status="incomplete" if missing else "complete", missing=missing),
    )
