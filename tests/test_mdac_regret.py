from dataclasses import replace

import pytest

from evaluation.improvement_records import MetricSpec, ResamplingPolicy
from evaluation.improvement_statistics import PairedObservation
from evaluation.mdac_regret import CriterionGate, compare_candidates, evaluate_regret

DIGEST = "a" * 64
METRIC = MetricSpec("accuracy", "v5", "observed", DIGEST)
GATE = CriterionGate(METRIC, "factuality", 0.75, 1, 1, 1, 1)


def report(name, scores, *, gate=GATE, expected=None):
    observations = tuple(
        PairedObservation(str(i), str(i), "clean", 0, str(i), DIGEST, gate.metric, old, new)
        for i, (old, new) in enumerate(scores)
    )
    return evaluate_regret(
        name,
        observations,
        expected_pair_ids=expected or tuple(o.pair_id for o in observations),
        criterion=gate,
        policy=ResamplingPolicy(seed=1),
        dependencies_known=True,
        enabled=True,
    )


def test_binary_matrix_and_priority():
    result = report("model-A", [(0, 0), (0, 1), (1, 1), (1, 0)])
    assert list(result["transitions"].values()) == [1, 1, 1, 1]
    assert result["rows"][-1]["hurt"] == 1
    assert result["rows"][-1]["priority"] == 2
    assert result["rows"][0]["priority"] == 1
    assert result["paired_estimate"]["interval"] is None


def test_lower_is_better_and_missingness():
    gate = replace(GATE, metric=replace(METRIC, direction="lower"), pass_threshold=0.1, target=0)
    result = report("model-B", [(0, 0.2)], gate=gate, expected=("0", "1"))
    assert result["rows"][0]["transition"] == "introduced_regression"
    assert result["rows"][0]["priority"] == pytest.approx(0.4)
    assert result["missing_pair_ids"] == ["1"]


def test_candidate_contrast_requires_same_roster_and_incumbent():
    left = report("A", [(0, 1), (1, 1)])
    right = report("B", [(0, 0), (1, 0)])
    assert compare_candidates((left, right), enabled=True)["criterion_contrast"] == 1
    with pytest.raises(ValueError):
        compare_candidates((left, report("B", [(1, 1), (1, 1)])), enabled=True)
    with pytest.raises(ValueError):
        compare_candidates((left, report("B", [(0, 1)], expected=("0", "1"))), enabled=True)


def test_recurring_rejection_ledger_validates_protocol():
    from test_improvement_records import candidate, protocol

    from evaluation.improvement_records import AttemptEvent, AttemptJournal, record_digest
    from evaluation.mdac_regret import rejection_history

    spec, _ = protocol(candidates=(candidate("A"), candidate("B")))
    events = []
    for cid in ("A", "B"):
        events.extend(
            (
                AttemptEvent(cid, len(events), cid, cid, "proposal"),
                AttemptEvent(
                    cid + "-d", len(events) + 1, cid, cid, "decision", terminal_status="rejected"
                ),
            )
        )
    journal = AttemptJournal("journal", record_digest(spec), tuple(events))
    history = rejection_history(spec, journal, enabled=True)
    assert history["recurring_proposals"] == {DIGEST: ["A", "B"]}
    with pytest.raises(ValueError):
        rejection_history(spec, replace(journal, protocol_digest="b" * 64), enabled=True)


def test_gepa_binding_preserves_original_results_and_missing_scores():
    from evaluation.gepa_alignment_scoring import pair_gepa_results
    from evaluation.schema import EvalResult

    incumbent = EvalResult("case", "accuracy", "factuality", "question", "yes", "yes", "correct")
    candidate = replace(incumbent, outcome="incorrect")
    options = dict(
        metric=replace(METRIC, source_schema="EvalResult", source_metric="gepa_score"),
        pair_id="p",
        repeat_id=0,
        condition_id="unperturbed",
        cluster_id="family",
        stratum_digest=DIGEST,
        enabled=True,
    )
    result = pair_gepa_results(incumbent, candidate, **options)
    assert (result.baseline_value, result.candidate_value) == (4, 1)
    assert incumbent.gepa_score is None
    assert (
        pair_gepa_results(incumbent, replace(candidate, outcome="needs_manual_review"), **options)
        is None
    )
    with pytest.raises(ValueError):
        pair_gepa_results(incumbent, replace(candidate, prompt="different"), **options)
