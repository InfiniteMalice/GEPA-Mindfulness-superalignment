"""Contamination regressions at the public training boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gepa_mindfulness.training.dataloader import DatasetBatch
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.engine import RLTrainingEngine
from gepa_mindfulness.training.runtime_config import DatasetConfig, RLRunConfig
from gepa_mindfulness.training.trajectory import RolloutRequest
from mindful_trace_gepa.dspy_modules import compile as compiler_module


@pytest.mark.parametrize("eligibility", ["DEVELOPMENT", "REGRESSION", "HIDDEN_EVAL", None, "train"])
def test_legacy_rl_loader_rejects_nontraining_provenance(
    tmp_path: Path, eligibility: object
) -> None:
    path = tmp_path / "input.jsonl"
    record = {
        "prompt": "secret evaluation prompt",
        "training_eligibility": "TRAIN",
        "metadata": {"source_record": {"training_eligibility": eligibility}},
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="training_eligibility"):
        DatasetBatch.from_path(path)


@pytest.mark.parametrize("split", ["trainset", "valset"])
def test_compiler_rejects_holdout_before_optimizer_visibility(
    monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    # The optimizer has external model side effects; rejection must precede construction.
    monkeypatch.setattr(compiler_module, "dspy", SimpleNamespace())

    def forbidden_optimizer(**kwargs: object) -> None:
        pytest.fail("contaminated input reached optimizer construction")

    monkeypatch.setattr(compiler_module, "BootstrapFewShot", forbidden_optimizer)
    compiler = compiler_module.GEPACompiler(
        {"enabled": True, "allow_optimizations": True}, lambda *args: 0.0, []
    )
    example = {"metadata": {"training_eligibility": "HIDDEN_EVAL"}}
    inputs = {"trainset": [], "valset": []}
    inputs[split] = [example]
    with pytest.raises(ValueError, match="training_eligibility"):
        compiler.compile(object(), **inputs)


def test_untagged_legacy_loader_remains_compatible(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"prompt": "legacy prompt"}), encoding="utf-8")
    assert DatasetBatch.from_path(path).items[0].prompt == "legacy prompt"


@pytest.mark.parametrize("mode", ["train", "resume"])
def test_canonical_rl_rejects_before_backend_creation(tmp_path: Path, mode: str) -> None:
    dataset = tmp_path / "unused.txt"
    dataset.write_text("unused", encoding="utf-8")
    request = RolloutRequest(
        prompt="held-out prompt",
        metadata={"source_record": {"training_eligibility": "HIDDEN_EVAL"}},
    )

    def forbidden_backend(config: object) -> None:
        pytest.fail("hidden evaluation reached a model backend")

    engine = RLTrainingEngine(
        RLRunConfig(dataset=DatasetConfig(train_path=str(dataset))),
        backend_factory=forbidden_backend,
        dataset_factory=lambda config: SimpleNamespace(materialize=lambda mode: [request]),
    )
    with pytest.raises(ValueError, match="training_eligibility"):
        if mode == "train":
            engine.train(max_steps=1)
        else:
            engine.resume(tmp_path / "checkpoint", max_steps=1)


@pytest.mark.parametrize(
    "metadata",
    [
        {"metadata": None},
        {"source_record": "stripped"},
        {"lineage": [{"training_eligibility": "HIDDEN_EVAL"}]},
        {"holdout_status": "held_out"},
    ],
)
def test_malformed_or_nested_holdout_provenance_rejects(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="training_eligibility"):
        require_training_eligible(metadata)


@pytest.mark.parametrize("passed", [False, None, "true", 1])
def test_failed_or_malformed_v5_observation_cannot_be_reinforced(passed: object) -> None:
    metadata = {
        "training_eligibility": "TRAIN",
        "source_record": {
            "case": {"case_version": "17case-v5", "case_id": 1},
            "system": {"repeat_id": 0},
            "outcome": {"passed": passed},
        },
    }
    with pytest.raises(ValueError, match="V5.*outcome"):
        require_training_eligible(metadata)


def test_failed_text_does_not_create_an_inferred_training_restriction() -> None:
    require_training_eligible({"prompt": "Discuss a failed outcome", "outcome": {"passed": False}})


def test_unrelated_legacy_structured_metadata_is_not_treated_as_v5() -> None:
    require_training_eligible(
        {
            "case": {"domain": "support-ticket"},
            "system": {"name": "customer-support"},
            "outcome": {"passed": False},
        }
    )


@pytest.mark.parametrize(
    "record",
    [
        {"case": {}, "system": {}, "outcome": {"passed": False}},
        {"case": {}, "system": {}, "outcome": {"passed": "true"}},
        {"case": {}, "outcome": {"passed": True}},
        {"case": {}, "system": None, "outcome": {"passed": True}},
        "not-a-record",
    ],
)
def test_explicit_v5_metadata_rejects_failed_or_malformed_records(record: object) -> None:
    with pytest.raises(ValueError, match="V5"):
        require_training_eligible({"v5_record": record})


def test_failed_v5_record_rejected_at_rl_loader_ingress(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Do not reinforce failed evaluation",
                "v5_record": {"case": {}, "system": {}, "outcome": {"passed": False}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="V5.*outcome"):
        DatasetBatch.from_path(path)
