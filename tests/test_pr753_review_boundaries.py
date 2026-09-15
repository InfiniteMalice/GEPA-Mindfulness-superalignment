"""Regressions for verified CodeRabbit and Codex review findings on PR 753."""

from dataclasses import replace

import pytest

from cognitive_pairwise_training import build_pairwise_examples
from evaluation.failure_atlas import FailureAtlas
from evaluation.schema import EvalCase
from evaluation.suites.common import evaluate_matched_error
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.factuality_observability.routing import RoutingContext, choose_routing_action
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from synthetic_data.generators import GenerationMetadata
from synthetic_data.generators.cooperation_under_uncertainty_generator import (
    generate_cooperation_cpt_candidates,
)
from synthetic_data.generators.proxy_vs_purpose_generator import generate_proxy_vs_purpose_cases
from test_v5_failure_atlas import failed, failure_events, regression_events, repaired_record


def test_hidden_regression_cannot_close_public_failure() -> None:
    atlas = FailureAtlas().observe("public", failed(), failure_events(), "2026-09-15T12:00:00Z")
    record = repaired_record()
    hidden = replace(
        record, assessment=replace(record.assessment, training_eligibility="HIDDEN_EVAL")
    )
    with pytest.raises(ValueError, match="hidden"):
        atlas.repair(
            "public", "repair", hidden, regression_events(), ("test:1",), "2026-09-15T12:01:00Z"
        )
    assert atlas.entries[0].status == "NEW"


def test_repair_cannot_predate_later_family_observation() -> None:
    atlas = FailureAtlas().observe("first", failed(), failure_events(), "2026-09-15T12:00:00Z")
    atlas = atlas.observe("later", failed(), failure_events(), "2026-09-15T12:02:00Z")
    with pytest.raises(ValueError, match="family last_seen"):
        atlas.repair(
            "first",
            "repair",
            repaired_record(),
            regression_events(),
            ("test:1",),
            "2026-09-15T12:01:00Z",
        )


def test_matched_errors_reject_whitespace_only_differences() -> None:
    cell = plan_v5_cells(
        case_ids=[1],
        stripe_ids=["DISTRACTOR"],
        repeats=1,
        subtypes={"DISTRACTOR": ["FABRICATED_FACT"]},
        model_version="m",
        harness_version="h",
    )[0]
    with pytest.raises(ValueError, match="distinct"):
        evaluate_matched_error(
            EvalCase("row", "local", "factuality", "2+2?", "4"),
            "4",
            " 4 ",
            cell=cell,
            defect="FABRICATED_FACT",
        )


def test_direct_routing_defaults_to_verification() -> None:
    context = RoutingContext(1, 0.99, 0.1, 0.1, 10, True, False, True, 0.0)
    assert choose_routing_action(context).recommended_action is RecommendedAction.ROUTE_EXTERNAL
    context.verification_required = False
    assert choose_routing_action(context).recommended_action is RecommendedAction.ACCEPT


def test_cpt_rejects_development_candidates() -> None:
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        build_pairwise_examples(generate_cooperation_cpt_candidates())


def test_generated_train_requires_review_authorization() -> None:
    with pytest.raises(ValueError, match="review|cell"):
        GenerationMetadata(training_eligibility=TrainingEligibility.TRAIN)


def test_training_ingress_rechecks_generated_review() -> None:
    record = generate_proxy_vs_purpose_cases()[0]
    record["metadata"]["training_eligibility"] = "TRAIN"
    with pytest.raises(ValueError, match="review|cell"):
        require_training_eligible(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("review_completed", False),
        ("reviewed_by", ""),
        ("review_authorization", None),
        ("canonical_case_id", 0),
        ("canonical_case_key", "wrong"),
        ("stripe", "UNKNOWN"),
    ],
)
def test_reviewed_generation_is_revalidated_at_ingress(field: str, value: object) -> None:
    cell = plan_v5_cells(
        case_ids=[1], stripe_ids=["NONE"], repeats=1, model_version="m", harness_version="h"
    )[0]
    approved = GenerationMetadata(
        cell=cell,
        training_eligibility=TrainingEligibility.TRAIN,
        review_completed=True,
        reviewed_by="reviewer:human",
        review_authorization=EvidenceReference("review:1", EvidenceSourceKind.EXTERNAL_RECORD),
    )
    source = generate_proxy_vs_purpose_cases()[0]["case_id"]
    record = generate_proxy_vs_purpose_cases(cell_metadata={source: approved})[0]
    require_training_eligible(record)
    record["metadata"][field] = value
    with pytest.raises(ValueError):
        require_training_eligible(record)


def test_only_latest_family_observation_can_close_and_regress() -> None:
    atlas = FailureAtlas().observe("first", failed(), failure_events(), "2026-09-15T12:00:00Z")
    atlas = atlas.observe("later", failed(), failure_events(), "2026-09-15T12:02:00Z")
    with pytest.raises(ValueError, match="latest family"):
        atlas.repair(
            "first",
            "repair",
            repaired_record(),
            regression_events(),
            ("test:1",),
            "2026-09-15T12:03:00Z",
        )
    atlas = atlas.repair(
        "later",
        "repair",
        repaired_record(),
        regression_events(),
        ("test:1",),
        "2026-09-15T12:03:00Z",
    )
    atlas = atlas.observe("regressed", failed(), failure_events(), "2026-09-15T12:04:00Z")
    assert atlas.entries[-1].status == "REGRESSION"


def test_candidate_selection_never_falls_back_to_an_older_family_entry() -> None:
    atlas = FailureAtlas().observe("first", failed(), failure_events(), "2026-09-15T12:00:00Z")
    record = failed()
    regression_only = replace(
        record, assessment=replace(record.assessment, training_eligibility="REGRESSION")
    )
    atlas = atlas.observe("latest", regression_only, failure_events(), "2026-09-15T12:01:00Z")
    assert atlas.repair_candidates() == ()
