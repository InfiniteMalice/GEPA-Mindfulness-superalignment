"""Tests for the canonical portable PyTorch RL runtime configuration."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from gepa_mindfulness.training.runtime_config import (
    RLRunConfig,
    load_rl_config,
    translate_legacy_config,
)


def canonical_payload() -> dict[str, object]:
    """Return a complete canonical configuration suitable for CPU smoke runs."""
    return {
        "runtime": {"backend": "pytorch", "device": "cpu"},
        "policy": {"model_name": "demo-model", "max_new_tokens": 32},
        "algorithm": {
            "name": "ppo",
            "learning_rate": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
        },
        "reward": {"weights": {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2}},
        "dataset": {"train_path": "prompts.txt", "format": "text"},
        "checkpoint": {"output_dir": "runs/cpu"},
        "logging": {"log_dir": "runs/cpu/logs", "level": "INFO"},
        "seed": 42,
    }


def test_canonical_mapping_produces_frozen_nested_sections() -> None:
    config = RLRunConfig.from_mapping(canonical_payload())

    assert config.runtime.backend == "pytorch"
    assert config.policy.max_new_tokens == 32
    assert config.algorithm.name == "ppo"
    with pytest.raises(FrozenInstanceError):
        config.runtime.device = "cuda"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"runtime": {"backend": "pytorch", "device": "mps"}}, "device"),
        ({"runtime": {"backend": "pytorch", "device": 1}}, "runtime.device"),
        ({"policy": {"unknown": "value"}}, "policy"),
        ({"seed": "42"}, "seed"),
    ],
)
def test_canonical_mapping_rejects_invalid_values_without_coercion(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RLRunConfig.from_mapping(payload)


def test_legacy_grpo_config_translates_to_canonical() -> None:
    with pytest.warns(DeprecationWarning):
        config = translate_legacy_config({"grpo": {"group_size": 4}, "device": "cpu"})

    assert config.runtime.backend == "pytorch"
    assert config.algorithm.name == "grpo"
    assert config.algorithm.group_size == 4


def test_canonical_file_load_has_no_deprecation_warning(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "canonical.yaml"
    path.write_text(yaml.safe_dump(canonical_payload()), encoding="utf-8")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        config = load_rl_config(path)

    assert config.runtime.device == "cpu"
    assert not [warning for warning in captured if warning.category is DeprecationWarning]


def test_legacy_file_load_warns_and_translates(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump({"trainer_type": "ppo", "device": "cpu"}), encoding="utf-8")

    with pytest.warns(DeprecationWarning):
        config = load_rl_config(path)

    assert config.algorithm.name == "ppo"


@pytest.mark.parametrize("name", ["pytorch_cpu_ppo.yaml", "pytorch_cpu_grpo.yaml"])
def test_cpu_runtime_examples_load(name: str) -> None:
    path = Path(__file__).parents[1] / "configs" / "rl" / name

    config = load_rl_config(path)

    assert config.runtime.device == "cpu"
    assert config.algorithm.name in {"ppo", "grpo"}


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": 7},
        {"dataset": {"train_path": "prompts.jsonl"}},
    ],
)
def test_partial_canonical_file_load_is_warning_free(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "partial.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        config = load_rl_config(path)

    assert config.seed == payload.get("seed", 42)
    expected_path = payload.get("dataset", {}).get("train_path", "")
    assert config.dataset.train_path == expected_path
    assert not [warning for warning in captured if warning.category is DeprecationWarning]


def test_real_legacy_dataset_path_translates_to_train_path() -> None:
    path = Path(__file__).parents[1] / "configs" / "training" / "phi3_dual_path.yml"

    with pytest.warns(DeprecationWarning):
        config = load_rl_config(path)

    assert config.dataset.train_path == "datasets/dual_path/data.jsonl"


@pytest.mark.parametrize(
    ("payload", "invalid"),
    [
        ({"algorithm": {"learning_rate": None}}, float("nan")),
        ({"algorithm": {"kl_coef": None}}, float("inf")),
        ({"algorithm": {"clip_range": None}}, float("-inf")),
        ({"algorithm": {"value_coef": None}}, float("nan")),
        ({"reward": {"weights": {"alpha": None}}}, float("nan")),
        ({"reward": {"weights": {"beta": None}}}, float("inf")),
        ({"reward": {"weights": {"gamma": None}}}, float("-inf")),
        ({"reward": {"weights": {"delta": None}}}, float("nan")),
    ],
)
def test_canonical_float_values_must_be_finite(
    payload: dict[str, object],
    invalid: float,
) -> None:
    section = next(iter(payload.values()))
    assert isinstance(section, dict)
    target = section.get("weights", section)
    assert isinstance(target, dict)
    key = next(key for key, value in target.items() if value is None)
    target[key] = invalid

    with pytest.raises(ValueError, match="finite"):
        RLRunConfig.from_mapping(payload)


def test_direct_legacy_translation_warning_points_to_caller() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1
        translate_legacy_config({"trainer_type": "ppo"})

    assert len(captured) == 1
    warning = captured[0]
    assert Path(warning.filename).resolve() == Path(__file__).resolve()
    assert warning.lineno == expected_line


def test_legacy_file_warning_points_to_loader_caller(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump({"trainer_type": "ppo"}), encoding="utf-8")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1
        load_rl_config(path)

    assert len(captured) == 1
    warning = captured[0]
    assert Path(warning.filename).resolve() == Path(__file__).resolve()
    assert warning.lineno == expected_line


@pytest.mark.parametrize(
    ("marker", "value"),
    [
        ("device", "cuda"),
        ("grpo", {"group_size": 4}),
        ("model", {"name": "legacy-model"}),
        ("model_name", "legacy-model"),
        ("output", {"checkpoint_dir": "runs/legacy"}),
        ("output_dir", "runs/legacy"),
        ("ppo", {"batch_size": 2}),
        ("reward_weights", {"alpha": 1.0}),
        ("trainer_type", "ppo"),
        ("training", {"max_steps": 2}),
    ],
)
def test_canonical_section_rejects_mixed_legacy_marker(
    tmp_path: Path,
    marker: str,
    value: object,
) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "mixed.yaml"
    payload = {"runtime": {"backend": "pytorch", "device": "cpu"}, marker: value}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys"):
        load_rl_config(path)


@pytest.mark.parametrize(
    "payload",
    [
        {"device": "cpu"},
        {"trainer_type": "grpo", "grpo": {"group_size": 4}},
        {"dataset": {"path": "legacy.jsonl"}},
    ],
)
def test_pure_legacy_markers_still_translate(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.warns(DeprecationWarning):
        config = load_rl_config(path)

    assert isinstance(config, RLRunConfig)


@pytest.mark.parametrize(
    "dataset",
    [
        {"train_path": "canonical.jsonl", "path": "legacy.jsonl"},
        {"unknown": "value"},
    ],
)
def test_dataset_with_canonical_or_unknown_subkeys_uses_strict_parser(
    tmp_path: Path,
    dataset: dict[str, object],
) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump({"dataset": dataset}), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset contains unknown keys"):
        load_rl_config(path)
