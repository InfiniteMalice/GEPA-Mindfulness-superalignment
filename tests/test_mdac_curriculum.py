"""Budget, uncertainty, and export invariants for the MDAC planner."""

import json
from dataclasses import replace

import pytest

from synthetic_data.mdac_curriculum import (
    CHANNELS,
    LESSONS,
    AllocationPolicy,
    plan_curriculum,
)


def test_no_diagnostics_spends_only_on_baseline():
    result = plan_curriculum((), policy=AllocationPolicy(budget=7), enabled=True)
    assert len(result["slots"]) == 7
    assert result["channel_counts"]["baseline_capability"] == 7
    assert set(result["channel_counts"]) == set(CHANNELS)
    assert all(slot["mechanism"] is None for slot in result["slots"])
    assert result["optimizer_input"] is False
    assert set(LESSONS) == {f"M{i:02d}" for i in range(1, 17)}


def test_disabled_and_invalid_budgets():
    with pytest.raises(ValueError, match="enabled"):
        plan_curriculum(())
    for invalid in (True, 1.5, -1, 1025):
        with pytest.raises(ValueError, match="budget"):
            replace(AllocationPolicy(), budget=invalid)
    assert not plan_curriculum((), policy=AllocationPolicy(budget=0), enabled=True)["slots"]


def observed(
    mechanism="M01",
    diagnosis="d1",
    *,
    origin="model",
    state="supported",
    confidence=0.75,
    failures=1,
    opportunities=10,
    current=True,
    previous=None,
    group=None,
):
    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
    from gepa_mindfulness.verification.mdac import MDACRecord, MechanismAssessment
    from gepa_mindfulness.verification.mdac_metrics import measure
    from synthetic_data.mdac_curriculum import CurriculumCriterion

    refs = (EvidenceReference("host:run", EvidenceSourceKind.EXTERNAL_RECORD),)
    assessment = MechanismAssessment(mechanism, state, confidence, supporting=refs)
    record = MDACRecord(
        diagnosis,
        "An evaluation opportunity.",
        dataset_version="development:1",
        evaluator_version="rubric:1",
        assessments=(assessment,),
        severity=0.9,
        occurrences=failures,
        opportunities=opportunities,
        provenance=refs,
    )

    def metric(ok):
        if mechanism == "M01":
            row = dict(
                id="i",
                present=True,
                retrieved=True,
                used=True,
                correct=ok,
                contradicted=False,
                transform="summary",
            )
            return measure(mechanism, [row], enabled=True)["retention"]
        row = dict(id="i", relevant=True, retrieved=ok, provenance_valid=True, stale=False)
        return measure(mechanism, [row], enabled=True)["recall"]

    return CurriculumCriterion(
        record=record,
        mechanism=mechanism,
        origin=origin,
        partition="diagnostic_development",
        current=metric(current),
        previous=None if previous is None else metric(previous),
        previous_record=(
            None if previous is None else replace(record, diagnostic_id=diagnosis + "-p")
        ),
        group=group,
    )


def test_priorities_do_not_pool_score_scales_or_impute_missing_counts():
    criterion = observed(current=False, previous=True)
    unknown = observed("M02", "d2", current=False, failures=0, opportunities=0)
    result = plan_curriculum((criterion, unknown), enabled=True)
    assert result["criteria"][0]["priority"] == pytest.approx(0.9 * 0.75 * 0.1 * 2)
    assert result["criteria"][1]["priority"] is None
    assert {s["channel"] for s in result["slots"] if s["mechanism"] == "M02"} == {
        "diagnostic_exploration"
    }
    assert result["inputs"]["criteria"][0]["record"]["provenance"]


def test_six_channels_reserves_correlations_and_deterministic_order():
    observations = (
        observed(current=False, previous=True, group="related"),
        observed("M02", "d2", current=False, group="related"),
        observed("M01", "mastered", failures=0),
        observed("M01", "uncertain", state="inconclusive", current=False),
    )
    policy = AllocationPolicy(budget=30, group_cap=5)
    result = plan_curriculum(observations, policy=policy, enabled=True)
    assert plan_curriculum(observations[::-1], policy=policy, enabled=True) == result
    assert sum(result["channel_counts"].values()) == 30
    assert all(result["channel_counts"].values())
    assert sum(s["group"] == "related" for s in result["slots"]) == 5
    assert any(s["diagnostic_id"] == "mastered" for s in result["slots"])
    assert any(s["diagnostic_id"] == "uncertain" for s in result["slots"])


@pytest.mark.parametrize("origin,route", [("runtime", "engineering"), ("governance", "governance")])
def test_external_failures_skip_training_even_with_high_severity(origin, route):
    criterion = observed(origin=origin, current=False)
    result = plan_curriculum((criterion,), enabled=True)
    assert result["routes"][0]["destination"] == route
    assert result["channel_counts"]["baseline_capability"] == len(result["slots"])
    assert all(s["diagnostic_id"] is None for s in result["slots"])


def test_duplicate_or_incomparable_inputs_fail_closed():
    criterion = observed(current=False, previous=True)
    with pytest.raises(ValueError, match="duplicate"):
        plan_curriculum((criterion, criterion), enabled=True)
    wrong_scope = replace(criterion.previous_record, evaluator_version="new-rubric")
    with pytest.raises(ValueError, match="comparable"):
        plan_curriculum((replace(criterion, previous_record=wrong_scope),), enabled=True)
    with pytest.raises(ValueError, match="mechanism"):
        plan_curriculum((replace(criterion, current=observed("M02").current),), enabled=True)


def test_cpu_generation_and_curation_preserve_eligibility_and_counts():
    from curriculum_fixtures import boundary_inputs
    from evaluation.causal_records import canonical_json
    from gepa_mindfulness.training.eligibility import require_training_eligible
    from synthetic_data.curriculum_records import curriculum_digest
    from synthetic_data.mdac_curriculum import generate_curriculum

    inputs = boundary_inputs()
    seen = []

    def generate(slot):
        seen.append(slot)
        i = slot["index"]
        base = inputs["candidates"][i]
        config = json.loads(base.generator_config_json) | {"mdac_request": slot}
        candidate = replace(base, generator_config_json=canonical_json(config))
        review = replace(inputs["reviews"][i], candidate_digest=curriculum_digest(candidate))
        return candidate, review

    options = dict(
        generate=generate,
        curation=inputs["policy"],
        as_of=inputs["as_of"],
        policy=AllocationPolicy(budget=2),
        enabled=True,
    )
    result = generate_curriculum((observed(current=False),), authenticate=lambda _: True, **options)
    assert len(seen) == 2
    assert result["generated_counts"] == result["plan"]["channel_counts"]
    assert result["exported_counts"] == result["generated_counts"]
    assert len(result["export"]["examples"]) == 2
    assert all("mechanism" not in str(e["actor"]) for e in result["export"]["examples"])
    with pytest.raises(ValueError):
        require_training_eligible({"training_eligibility": "TRAIN", "source": result})
    rejected = generate_curriculum((), authenticate=lambda _: False, **options)
    assert sum(rejected["generated_counts"].values()) == 2
    assert sum(rejected["exported_counts"].values()) == 0
    assert len(rejected["export"]["excluded"]) == 2


def test_generated_candidate_for_another_plan_is_rejected():
    from curriculum_fixtures import boundary_inputs
    from synthetic_data.mdac_curriculum import generate_curriculum

    inputs = boundary_inputs()
    with pytest.raises(ValueError, match="allocation request"):
        generate_curriculum(
            (),
            policy=AllocationPolicy(budget=1),
            curation=inputs["policy"],
            as_of=inputs["as_of"],
            enabled=True,
            generate=lambda _: (inputs["candidates"][0], inputs["reviews"][0]),
        )


def test_held_out_diagnostics_and_unmeasured_numbers_cannot_steer_planning():
    signal = observed()
    with pytest.raises(ValueError, match="partition"):
        replace(signal, partition="final_test")
    with pytest.raises(ValueError, match="measured opportunities"):
        replace(signal, current=replace(signal.current, denominator=0, normalized=0))


def test_no_generator_supply_is_explicit():
    from curriculum_fixtures import curation_inputs
    from synthetic_data.mdac_curriculum import generate_curriculum

    fixture = curation_inputs()
    result = generate_curriculum(
        (),
        policy=AllocationPolicy(budget=3),
        generate=lambda _: None,
        curation=fixture["policy"],
        as_of=fixture["as_of"],
        enabled=True,
    )
    assert sum(result["plan"]["channel_counts"].values()) == 3
    assert sum(result["generated_counts"].values()) == sum(result["exported_counts"].values()) == 0
