"""PEO data scheduling preserves anchors, source restrictions, and head phases."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.participatory_agency.training.curriculum import (
    DEFAULT_CURRICULUM,
    PEOStage,
    get_peo_curriculum,
)
from gepa_mindfulness.training.peo_curriculum import (
    AnchorEvaluation,
    AnchorResult,
    CurriculumBucket,
    CurriculumMixture,
    CurriculumUnit,
    PEOCurriculumDataset,
)
from gepa_mindfulness.training.trajectory import RolloutRequest
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
from synthetic_data.worlds import generate_world


def unit(identifier, bucket, stage=PEOStage.CAUSAL, metadata=None, size=1):
    return CurriculumUnit(
        identifier,
        stage,
        bucket,
        tuple(
            RolloutRequest(
                prompt=f"Public prompt {identifier}:{index}",
                case_id=f"{identifier}:{index}",
                metadata=metadata if metadata is not None else {"training_eligibility": "TRAIN"},
            )
            for index in range(size)
        ),
        dimensions={"U": 0.4, "T": 0.1},
    )


def dataset(**changes):
    units = [unit(f"anchor-{i}", CurriculumBucket.ANCHOR) for i in range(3)]
    for stage in PEOStage:
        for bucket in list(CurriculumBucket)[1:]:
            units.extend(unit(f"{stage.value}:{bucket.value}:{i}", bucket, stage) for i in range(3))
    values = dict(
        units=tuple(units),
        phase=DEFAULT_CURRICULUM[0],
        stage=PEOStage.CAUSAL,
        seed=42,
        enabled=True,
    )
    values.update(changes)
    return PEOCurriculumDataset(**values)


def evaluation(plan, **changes):
    results = tuple(
        AnchorResult(
            item.unit_id,
            True,
            EvidenceReference(f"assessment:{item.unit_id}", EvidenceSourceKind.EXTERNAL_RECORD),
        )
        for item in plan.units
        if item.bucket is CurriculumBucket.ANCHOR
    )
    values = dict(
        plan_digest=plan.digest,
        system=EvaluatedSystemVersion("trained-v2", "harness"),
        results=results,
    )
    values.update(changes)
    return AnchorEvaluation(**values)


def test_stage_catalog_and_existing_heads_remain_separate() -> None:
    with pytest.raises(ValueError, match="enabled"):
        get_peo_curriculum()
    stages = get_peo_curriculum(enabled=True)
    assert [spec.stage for spec in stages] == list(PEOStage)
    assert len(stages) == 7
    assert "supersession" in stages[-1].temporal_features
    assert "source_reliability_change" in stages[-1].temporal_features
    assert len(DEFAULT_CURRICULUM) == 5
    assert DEFAULT_CURRICULUM[0].active_heads == ("epistemic",)
    assert DEFAULT_CURRICULUM[-1].loss_weights["epistemic"] == 0.25


@pytest.mark.parametrize("stage", list(PEOStage))
def test_each_stage_keeps_anchors_and_all_four_pools(stage) -> None:
    plan = dataset(stage=stage)
    requests = plan.materialize("train")
    reports = [request.metadata["peo_curriculum"] for request in requests]
    assert Counter(report["bucket"] for report in reports) == {
        "anchor": 4,
        "weakness": 2,
        "frontier": 1,
        "ood": 1,
    }
    assert all(report["stage"] == stage.value for report in reports)
    assert all(request.prompt.startswith("Public prompt") for request in requests)
    assert all(request.metadata["training_eligibility"] == "TRAIN" for request in requests)
    assert all(report["phase"] == DEFAULT_CURRICULUM[0].name for report in reports)


def test_determinism_input_order_independence_and_round_rotation() -> None:
    plan = dataset(mixture=CurriculumMixture(1, 1, 1, 1))
    assert plan.materialize("evaluate") == plan.materialize("evaluate")
    reversed_plan = replace(plan, units=tuple(reversed(plan.units)))
    assert plan.digest == reversed_plan.digest
    assert plan.materialize("evaluate") == reversed_plan.materialize("evaluate")
    seen = set()
    for round_id in range(3):
        for request in replace(plan, round_id=round_id).materialize("evaluate"):
            report = request.metadata["peo_curriculum"]
            if report["bucket"] == "anchor":
                seen.add(report["unit_id"])
    assert seen == {"anchor-0", "anchor-1", "anchor-2"}


def test_grouped_requests_stay_contiguous_and_source_is_snapshotted() -> None:
    metadata = {"training_eligibility": "TRAIN", "source_record": {"annotation": ["kept"]}}
    paired = unit("paired", CurriculumBucket.FRONTIER, metadata=metadata, size=2)
    plan = dataset()
    plan = replace(
        plan,
        units=tuple(
            item
            for item in plan.units
            if not (item.stage is PEOStage.CAUSAL and item.bucket is CurriculumBucket.FRONTIER)
        )
        + (paired,),
    )
    metadata["source_record"]["annotation"][0] = "mutated"
    requests = plan.materialize("train")
    positions = [
        index
        for index, request in enumerate(requests)
        if request.metadata["peo_curriculum"]["unit_id"] == "paired"
    ]
    assert positions == list(range(positions[0], positions[0] + 2))
    assert [requests[i].case_id for i in positions] == ["paired:0", "paired:1"]
    assert requests[positions[0]].metadata["source_record"]["annotation"] == ["kept"]
    requests[positions[0]].metadata["source_record"]["annotation"][0] = "changed output"
    assert plan.materialize("train")[positions[0]].metadata["source_record"]["annotation"] == [
        "kept"
    ]


@pytest.mark.parametrize("mode", ["train", "resume"])
@pytest.mark.parametrize("restriction", ["DEVELOPMENT", "REGRESSION", "HIDDEN_EVAL"])
def test_nested_restrictions_and_unselected_candidates_reject(mode, restriction) -> None:
    plan = dataset(mixture=CurriculumMixture(1, 1, 1, 1))
    poisoned = replace(
        plan.units[-1],
        stage=PEOStage.CAUSAL,
        requests=(
            RolloutRequest(
                "never optimize",
                metadata={
                    "training_eligibility": "TRAIN",
                    "source_record": {"training_eligibility": restriction},
                },
            ),
        ),
    )
    plan = replace(plan, units=plan.units[:-1] + (poisoned,))
    with pytest.raises(ValueError, match="training_eligibility"):
        plan.materialize(mode)
    assert len(plan.materialize("evaluate")) == 4


def test_pr9_worlds_keep_their_training_exclusion() -> None:
    source = json.loads(json.dumps(generate_world(seed=1, enabled=True).to_dict()))
    plan = dataset()
    world_unit = unit(
        "world",
        CurriculumBucket.ANCHOR,
        metadata={"training_eligibility": "TRAIN", "source_record": source},
    )
    plan = replace(plan, units=plan.units + (world_unit,))
    with pytest.raises(ValueError, match="training_eligibility"):
        plan.materialize("train")
    assert plan.materialize("collect")


def test_adaptation_preserves_anchors_and_targets_declared_weaknesses() -> None:
    plan = dataset()
    target = "representation_invariance:weakness:1"
    adapted = plan.adapt(
        evaluation(plan),
        stage=PEOStage.INVARIANCE,
        weakness_ids=(target,),
        mixture=CurriculumMixture(3, 3, 1, 1),
    )
    assert adapted.round_id == plan.round_id + 1
    assert adapted.units == plan.units
    assert adapted.phase == plan.phase
    assert adapted.last_evaluation.plan_digest == plan.digest
    requests = adapted.materialize("train")
    targets = [
        r.metadata["peo_curriculum"]["unit_id"]
        for r in requests
        if r.metadata["peo_curriculum"]["bucket"] == "weakness"
    ]
    assert targets == [target] * 3
    assert plan.stage is PEOStage.CAUSAL


@pytest.mark.parametrize("change", ["failed", "missing", "duplicate", "stale", "unknown"])
def test_adaptation_rejects_incomplete_or_regressed_anchors(change) -> None:
    plan = dataset()
    assessment = evaluation(plan)
    with pytest.raises(ValueError):
        if change == "failed":
            assessment = replace(
                assessment,
                results=(replace(assessment.results[0], passed=False),) + assessment.results[1:],
            )
        elif change == "missing":
            assessment = replace(assessment, results=assessment.results[1:])
        elif change == "duplicate":
            assessment = replace(assessment, results=assessment.results + assessment.results[:1])
        elif change == "stale":
            assessment = replace(assessment, plan_digest=dataset(seed=43).digest)
        else:
            assessment = replace(
                assessment,
                results=(replace(assessment.results[0], unit_id="other"),) + assessment.results[1:],
            )
        plan.adapt(assessment, stage=PEOStage.INVARIANCE)


@pytest.mark.parametrize(
    "counts", [(0, 1, 1, 1), (1, 8, 1, 1), (True, 1, 1, 1), (1, -1, 1, 1), (1, 1, 1.5, 1)]
)
def test_invalid_quotas_cannot_remove_anchors(counts) -> None:
    with pytest.raises(ValueError):
        CurriculumMixture(*counts)


def test_other_fail_closed_boundaries() -> None:
    with pytest.raises(ValueError, match="enabled"):
        dataset(enabled=False)
    plan = dataset()
    for action in (
        lambda: plan.materialize("invented"),
        lambda: plan.adapt(evaluation(plan), stage=PEOStage.TEMPORAL),
        lambda: plan.adapt(evaluation(plan), weakness_ids=("anchor-0",)),
        lambda: replace(plan, units=plan.units + plan.units[:1]),
        lambda: replace(
            plan, units=tuple(u for u in plan.units if u.bucket is not CurriculumBucket.OOD)
        ),
        lambda: replace(plan.units[0], dimensions={"unknown": 0.5}),
        lambda: replace(plan.units[0], dimensions={"U": float("nan")}),
        lambda: replace(plan.units[0], temporal_features=("invented",)),
        lambda: replace(
            evaluation(plan).results[0],
            evidence=EvidenceReference("private", EvidenceSourceKind.PRIVATE_REASONING),
        ),
    ):
        with pytest.raises(ValueError):
            action()


def test_explicit_train_declaration_is_required_only_in_new_provider() -> None:
    plan = dataset()
    untagged = unit("legacy", CurriculumBucket.ANCHOR, metadata={})
    plan = replace(plan, units=plan.units + (untagged,))
    with pytest.raises(ValueError, match="TRAIN"):
        plan.materialize("train")
    assert plan.materialize("evaluate")


def test_future_stage_restrictions_cannot_enter_training_plan_identity() -> None:
    plan = dataset()
    future = unit(
        "future-holdout",
        CurriculumBucket.OOD,
        PEOStage.TEMPORAL,
        metadata={"training_eligibility": "HIDDEN_EVAL"},
    )
    plan = replace(plan, units=plan.units + (future,))
    with pytest.raises(ValueError, match="training_eligibility"):
        plan.materialize("train")


def test_each_adapted_stage_retains_anchor_bank_and_resets_old_targets() -> None:
    plan = dataset(weakness_ids=("causal_epistemic:weakness:1",))
    anchors = tuple(u for u in plan.units if u.bucket is CurriculumBucket.ANCHOR)
    for stage in list(PEOStage)[1:]:
        plan = plan.adapt(evaluation(plan), stage=stage)
        assert tuple(u for u in plan.units if u.bucket is CurriculumBucket.ANCHOR) == anchors
        assert plan.weakness_ids == ()
        assert len(plan.materialize("train")) == 8
    same_stage = plan.adapt(evaluation(plan))
    assert same_stage.stage is PEOStage.TEMPORAL
    assert same_stage.round_id == 7


@pytest.mark.parametrize(
    "metadata",
    [
        {"training_eligibility": "TRAIN", "peo_curriculum": {}},
        {"training_eligibility": "TRAIN", "source": {"private": object()}},
    ],
)
def test_untrusted_metadata_cannot_overwrite_report_or_escape_json_snapshot(metadata) -> None:
    with pytest.raises((ValueError, TypeError)):
        unit("bad-source", CurriculumBucket.ANCHOR, metadata=metadata)
