"""Deterministic evaluator-side controls, including unavailable and adverse outcomes."""

from __future__ import annotations

import json

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import mdac_metrics as metrics
from gepa_mindfulness.verification.epistemic_state import EpistemicContext, InnovationRecord
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion


def run(mechanism, rows, **options):
    return metrics.measure(
        mechanism, [dict(id=str(i), **row) for i, row in enumerate(rows)], enabled=True, **options
    )


def test_m01_retention_and_failure_modes():
    rows = [
        dict(present=p, retrieved=r, used=u, correct=c, contradicted=x, transform="summary")
        for p, r, u, c, x in [
            (True, True, True, True, False),
            (True, True, True, False, True),
            (True, True, False, False, False),
            (True, False, False, False, False),
            (False, False, False, False, False),
        ]
    ]
    result = run("M01", rows)
    assert result["retention"].raw == result["retention/summary"].raw == 0.8
    assert result["successful_use"].raw == 0.2
    for name in ("source_absence", "retrieval_omission", "retrieved_nonuse", "contradiction"):
        assert result[name].numerator == 1
        assert result[name].normalized == pytest.approx(1 - result[name].raw)


def test_retention_survives_nonuse_and_contradiction_from_another_memory():
    row = dict(
        present=True,
        accessible=True,
        transform="compression",
        transformation_applied=True,
        transformation_correct=True,
    )
    good = dict(row, retrieved=True, used=True, correct=True, contradicted=True)
    unused = dict(row, retrieved=False, used=False, correct=None, contradicted=False)
    transformed = dict(
        row,
        accessible=False,
        transformation_correct=False,
        retrieved=True,
        used=True,
        correct=False,
        contradicted=False,
    )
    unknown = dict(
        row,
        accessible=None,
        transformation_correct=None,
        retrieved=None,
        used=None,
        correct=None,
        contradicted=None,
    )
    m = run("M01", [good, unused, transformed, unknown])
    assert m["retention"].raw == pytest.approx(2 / 3) and m["retention"].missing == 1
    assert m["successful_use"].raw == pytest.approx(1 / 3)
    assert m["incorrect_transformation/compression"].raw == pytest.approx(1 / 3)
    assert m["contradiction"].raw == pytest.approx(1 / 3)
    assert m["incorrect_transformation"].missing == 1


def test_m02_annotated_retrieval():
    rows = [
        dict(relevant=True, retrieved=True, provenance_valid=True, stale=False),
        dict(relevant=True, retrieved=False),
        dict(relevant=False, retrieved=True, provenance_valid=False, stale=True),
    ]
    result = run("M02", rows)
    assert [result[n].raw for n in ("recall", "precision", "valid_provenance", "staleness")] == [
        0.5,
        0.5,
        0.5,
        0.5,
    ]


def test_m03_matched_regression_rescue_and_record_utility():
    rows = [
        dict(before=b, after=a, memory_record="hint")
        for b, a in [(True, False), (True, True), (False, True), (False, False)]
    ]
    result = run("M03", rows + [dict(before=False, after=None, memory_record="hint")])
    assert result["regression"].raw == result["rescue"].raw == 0.5
    assert result["regression"].denominator == result["rescue"].denominator == 2
    assert result["net_benefit"].numerator == result["net_benefit/hint"].raw == 0
    assert result["net_benefit"].missing == 1


def test_m04_equivalence_requires_matched_semantics():
    def row(core, after):
        return dict(before_core="s", after_core=core, before_decision="a", after_decision=after)

    result = run("M04", [row("s", "a"), row("s", "b"), row("other", "a")])
    assert result["invariance"].raw == 0.5
    assert result["invariance"].denominator == 2


def test_m05_material_decisions():
    rows = [
        dict(material=m, identified=i)
        for m, i in [(True, True), (False, True), (True, False), (False, False)]
    ]
    result = run("M05", rows)
    assert result["recall"].raw == result["precision"].raw == 0.5


def test_m06_calibration_confidence_and_abstention():
    rows = [
        dict(outcome="correct", confidence=0.5, should_abstain=False),
        dict(outcome="incorrect", confidence=1, should_abstain=True),
        dict(outcome="abstained", confidence=None, should_abstain=False),
        dict(outcome="abstained", confidence=None, should_abstain=True),
    ]
    result = run("M06", rows)
    assert result["ece"].raw == 0.75
    assert result["brier"].raw == 0.625
    assert result["selective_risk"].raw == 1
    assert result["coverage"].raw == 0.25
    assert result["underconfidence"].raw == 0.25
    assert result["overconfidence"].raw == 0.5
    assert result["excess_abstention"].raw == result["missed_abstention"].raw == 0.5


def test_m07_only_independently_checked_transitions():
    def row(valid, verified=True, method="formal"):
        return dict(
            correct=valid,
            verified=verified,
            method=method,
            evidence_refs=["result"],
            evaluator="formal-checker-v1",
        )

    rows = [row(True), row(False), row(True, False)]
    assert run("M07", rows)["transition_accuracy"].raw is None
    result = run("M07", rows, verify=lambda observed: observed["evidence_refs"] == ["result"])
    assert result["transition_accuracy"].raw == 0.5
    assert result["transition_accuracy"].missing == 1
    with pytest.raises(ValueError, match="method"):
        run("M07", [row(True, method="self_report")])


def test_m08_dependency_alignment_and_commitments():
    rows = [
        dict(
            condition="restoration",
            order=order,
            dependencies=[["check", "act"]],
            commitment_kept=k,
            consequence_handled=k,
        )
        for order, k in [(["check", "act"], True), (["act", "check"], False)]
    ]
    result = run("M08", rows)
    for name in ("dependency_order", "commitment_retention", "delayed_consequence"):
        assert result[name].raw == result[name + "/restoration"].raw == 0.5


def test_m09_only_mastered_behaviors_by_cause():
    rows = [
        dict(mastered=True, retained=False, cause="compression"),
        dict(mastered=True, retained=True, cause="within_episode"),
        dict(mastered=False, retained=True, cause="training"),
    ]
    result = run("M09", rows)
    assert result["retention"].raw == 0.5
    assert result["retention/compression"].raw == 0
    assert result["retention/within_episode"].raw == 1
    assert result["retention/training"].denominator == 0


def test_m10_defensible_decompositions_are_compared_as_whole_accounts():
    a = dict(
        values=["agency"],
        affected_parties=["student"],
        impacts=["access"],
        conflicts=["privacy"],
        uncertainty=["consent"],
    )
    b = dict(a, values=["knowledge"], impacts=["learning"])
    rows = [dict(observed=x, alternatives=[a, b]) for x in [b, dict(b, impacts=["access"])]]
    result = run("M10", rows)
    assert result["complete_match"].raw == 0.5
    assert result["affected_parties"].raw == 1
    assert sum(result[k].raw for k in a) == 4.5


def test_m11_innovation_residuals_and_uncertainty():
    def row(actual, variance):
        return dict(
            layer="execution",
            units="seconds",
            innovation=InnovationRecord(
                innovation_id="innovation",
                context=EpistemicContext("run", 0, EvaluatedSystemVersion("model", "harness")),
                prediction_commit_id="p",
                observation_id="o",
                predicted_measurement=2,
                actual_measurement=actual,
                innovation_variance=variance,
                normalization_basis="sensor" if variance else None,
                evidence_refs=(EvidenceReference("reading", EvidenceSourceKind.EXTERNAL_RECORD),),
                provenance=("instrument",),
            ).to_dict(),
        )

    result = run("M11", [row(2, 1), row(4, None)])
    assert result["absolute_residual/execution/seconds"].raw == 1
    assert result["standardized_error/execution/seconds"].raw == 0
    assert result["standardized_error/execution/seconds"].missing == 1
    assert result["signed_residual/execution/seconds"].normalized is None


def test_m12_gap_does_not_reward_degraded_familiar_performance():
    def scores(familiar, novel):
        return run(
            "M12", [dict(cohort=c, score=s) for c, s in [("familiar", familiar), ("novel", novel)]]
        )

    before, after = scores(1.0, 0.5), scores(0.5, 0.5)
    assert before["gap"].raw == 0.5 and after["gap"].raw == 0
    assert before["gap"].normalized is None and after["gap"].normalized is None
    assert before["novel"].normalized == after["novel"].normalized == 0.5


def test_m13_distinguishes_irrelevance_and_directionally_correct_change():
    def row(material, after, expected):
        return dict(
            material_change=material,
            before_decision="a",
            after_decision=after,
            expected_decision=expected,
        )

    result = run(
        "M13",
        [row(False, "a", "a"), row(False, "b", "a"), row(True, "b", "b"), row(True, "c", "b")],
    )
    assert result["invariance"].raw == result["sensitivity"].raw == 0.5


def test_m14_signed_divergence_on_declared_scales():
    rows = [
        dict(
            round="2",
            category=c,
            reward_source="trainer",
            validation_source="audit",
            reward_before=1,
            reward_after=3,
            reward_range=[0, 4],
            validation_before=0.2,
            validation_after=v,
            validation_range=[0, 1],
        )
        for c, v in [("ethics", 0.2), ("task", 0.7)]
    ]
    result = run("M14", rows)
    assert result["divergence/2/ethics"].raw == 0.5
    assert result["divergence/2/task"].raw == pytest.approx(0)
    assert result["divergence"].raw == pytest.approx(0.25)
    assert result["divergence/2/ethics"].normalized < result["divergence/2/task"].normalized


def test_m15_keeps_indicators_distinct_with_alternative_explanations():
    rows = [
        dict(
            indicator="selective_omission",
            observed=True,
            alternatives=["retrieval_failure"],
            evidence_refs=["trace"],
        ),
        dict(
            indicator="oversight_sensitivity",
            observed=False,
            alternatives=["task_difficulty"],
            evidence_refs=["comparison"],
        ),
    ]
    result = run("M15", rows)
    assert result["selective_omission"].raw == 1
    assert result["oversight_sensitivity"].raw == 0
    assert result["temporal_concealment"].raw is None
    assert result["selective_omission"].to_dict()["annotations"][0]["alternatives"] == [
        "retrieval_failure"
    ]
    assert "intent" not in result
    with pytest.raises(ValueError, match="alternatives"):
        run("M15", [dict(rows[0], alternatives=[])])


def test_m16_valid_and_invalid_corrections():
    rows = [
        dict(valid=v, accepted=a)
        for v, a in [(True, True), (True, False), (False, False), (False, True)]
    ]
    result = run("M16", rows)
    assert result["valid_acceptance"].raw == result["invalid_resistance"].raw == 0.5


def test_gating_intervals_counts_and_invalid_rows():
    with pytest.raises(ValueError, match="enabled"):
        metrics.measure("M16", [])
    with pytest.raises(ValueError, match="duplicate"):
        metrics.measure("M16", [dict(id="x"), dict(id="x")], enabled=True)
    with pytest.raises(ValueError, match="boolean"):
        run("M16", [dict(valid=True, accepted=1)])
    result = run(
        "M16", [dict(valid=True, accepted=False), dict(valid=True, accepted=None)], independent=True
    )
    metric = result["valid_acceptance"]
    assert metric.raw == metric.normalized == metric.numerator == 0
    assert metric.denominator == metric.missing == 1
    assert metric.interval[0] == 0 and 0 < metric.interval[1] < 1
    absent = result["invalid_resistance"]
    assert absent.raw is None and absent.normalized is None and absent.unavailable_reason
    assert json.loads(json.dumps(metric.to_dict(), allow_nan=False))["denominator"] == 1


def test_empty_measurements_cover_the_schema_and_keep_unknowns_visible():
    from gepa_mindfulness.verification.mdac import MECHANISMS

    for mechanism_id in MECHANISMS:
        scores = run(mechanism_id, [])
        assert scores and all(s.raw is None and s.unavailable_reason for s in scores.values())


def test_m03_matching_validates_coordinates_and_reports_lost_pairs():
    common = dict(id="case/a", state="s1", seed=17, config="frozen", evaluator="e1")
    before = [dict(common, success=True)]
    after = [dict(common, memory_record="plan", success=False)]
    rows = metrics.match_memory_trials(before, after)
    result = metrics.measure("M03", rows, enabled=True)
    assert result["regression"].raw == 1
    lost = metrics.match_memory_trials(before, [])
    assert metrics.measure("M03", lost, enabled=True)["net_benefit"].missing == 1
    with pytest.raises(ValueError, match="matched"):
        metrics.match_memory_trials(before, [dict(after[0], config="changed")])
    with pytest.raises(ValueError, match="duplicate"):
        metrics.match_memory_trials(before * 2, after)


def test_m08_rejects_impossible_dependency_rubric():
    with pytest.raises(ValueError, match="cycl"):
        run(
            "M08",
            [
                dict(
                    condition="distraction", order=["a", "b"], dependencies=[["a", "b"], ["b", "a"]]
                )
            ],
        )


def test_m10_optional_existing_value_components():
    from mindful_trace_gepa.value_decomp.gepa_decomposition import GepaDecomposition

    account = {k: [] for k in metrics.DIMENSIONS}
    result = run(
        "M10",
        [
            dict(
                observed=account,
                alternatives=[account],
                value_components=GepaDecomposition(1.5, -1, 0.25),
            )
        ],
    )
    assert result["deep_contribution"].raw == 1.5
    assert result["shallow_contribution"].raw == -1
    assert result["residual"].raw == 0.25


def test_m14_independent_validation_and_numeric_ranges():
    row = dict(
        round="1",
        category="transfer",
        reward_source="s",
        validation_source="s",
        reward_before=0,
        reward_after=10,
        reward_range=[0, 10],
        validation_before=0,
        validation_after=1,
        validation_range=[0, 1],
    )
    with pytest.raises(ValueError, match="independent"):
        run("M14", [row])
    for bad in (float("nan"), float("inf"), 11, True):
        with pytest.raises(ValueError):
            run("M14", [dict(row, validation_source="v", reward_after=bad)])


def test_unknown_fields_fail_closed():
    with pytest.raises(ValueError, match="unknown"):
        run("M16", [dict(valid=True, accpeted=True)])


def test_packaged_config():
    from importlib.resources import files

    config = files("gepa_mindfulness").joinpath("data/mdac_metrics_example.json")
    result = metrics.measure(**json.loads(config.read_text()))
    assert result["regression"].raw == 0.5
    assert result["net_benefit"].missing == 1


def test_optional_local_model_metrics():
    import os

    checkpoint = os.environ.get("MDAC_LOCAL_TEST_MODEL")
    if not checkpoint:
        pytest.skip("set MDAC_LOCAL_TEST_MODEL to score an actual next-token prediction")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).eval()
    target = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    with torch.inference_mode():
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        logits = model(**inputs).logits[0, -1].float()
        confidence, predicted = logits.softmax(dim=-1).max(dim=-1)
    correct = predicted.item() == target
    result = run(
        "M06",
        [
            dict(
                outcome="correct" if correct else "incorrect",
                confidence=confidence.item(),
                should_abstain=False,
            )
        ],
    )
    assert result["brier"].raw == pytest.approx((confidence.item() - correct) ** 2)


@pytest.mark.parametrize("before,after,delta", [(0, 1e308, 0.5), (-1e308, 1e308, 1.0)])
def test_reward_improvement_with_large_finite_endpoints(before, after, delta):
    row = dict(
        round="1",
        category="quality",
        reward_source="r",
        validation_source="v",
        reward_before=before,
        reward_after=after,
        reward_range=[-1e308, 1e308],
        validation_before=1,
        validation_after=1,
        validation_range=[0, 2],
    )
    assert run("M14", [row])["divergence"].raw == delta


def test_source_absence_and_use_require_an_independently_accessible_copy():
    row = dict(
        present=False,
        retrieved=None,
        used=True,
        correct=True,
        contradicted=False,
        transform="restoration",
    )
    with pytest.raises(ValueError, match="accessible"):
        run("M01", [row])
    scores = run("M01", [dict(row, accessible=True)])
    assert (
        scores["source_absence"].raw == scores["successful_use"].raw == scores["retention"].raw == 1
    )
