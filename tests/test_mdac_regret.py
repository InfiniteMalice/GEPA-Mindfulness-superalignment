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
