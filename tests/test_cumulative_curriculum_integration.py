"""The executable offline guide must preserve every existing admission boundary."""

# Standard library
import re
from pathlib import Path

# Third-party
import pytest

# Local
from gepa_mindfulness.participatory_agency.training.curriculum import DEFAULT_CURRICULUM, PEOStage
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.peo_curriculum import (
    CurriculumBucket,
    CurriculumMixture,
    CurriculumUnit,
    PEOCurriculumDataset,
)
from gepa_mindfulness.training.trajectory import RolloutRequest


@pytest.fixture(scope="module")
def guide_result():
    path = Path(__file__).resolve().parents[1] / "docs" / "cumulative_curriculum.md"
    assert path.is_file(), "The offline guide must exist"
    blocks = re.findall(r"```python\n(.*?)\n```", path.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 1
    namespace = {}
    exec(compile(blocks[0], str(path), "exec"), namespace)
    return namespace["example_result"]


def test_executable_guide(guide_result):
    assert guide_result["curation"]["lessons"][0]["disposition"] == "curated"
    assert guide_result["expired"]["lessons"][0]["disposition"] == "withheld"
    assert guide_result["utility"]["comparisons"]["marginal"]["delta"] == 0.5
    milestone = guide_result["milestones"]["milestones"][0]["verified"]
    assert milestone["step_completed"] is True and milestone["goal_progress"] is False
    assert guide_result["curriculum"]["coverage"]["status"] == "complete"
    comparison = guide_result["comparison"]
    assert len(comparison["arms"]) == 7
    assert all(arm["actual_budget_match"] is None for arm in comparison["arms"])


def test_all_actual_exports_reject_training(guide_result):
    for report in guide_result.values():
        assert report["optimizer_input"] is False
        with pytest.raises(ValueError):
            require_training_eligible({"training_eligibility": "TRAIN", "source": report})


def test_peo_collect_evaluate_preserve_source_and_train_resume_reject(guide_result):
    export = guide_result["curriculum"]
    prompt = export["examples"][0]["actor"]["before"][0]["content"]
    units = tuple(
        CurriculumUnit(
            bucket.value,
            PEOStage.CAUSAL,
            bucket,
            (RolloutRequest(prompt=prompt, case_id=bucket.value, metadata={"source": export}),),
        )
        for bucket in CurriculumBucket
    )
    dataset = PEOCurriculumDataset(
        units=units,
        phase=DEFAULT_CURRICULUM[0],
        mixture=CurriculumMixture(1, 1, 1, 1),
        enabled=True,
    )
    for mode in ("collect", "evaluate"):
        requests = dataset.materialize(mode)
        assert len(requests) == 4
        assert all(request.metadata["source"] == export for request in requests)
    for mode in ("train", "resume"):
        with pytest.raises(ValueError):
            dataset.materialize(mode)
