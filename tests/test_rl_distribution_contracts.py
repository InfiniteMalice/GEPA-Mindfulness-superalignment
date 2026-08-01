"""Distribution contracts for shipped portable RL presets and development dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from gepa_mindfulness.training.adapters import FlatJSONLAdapter
from gepa_mindfulness.training.runtime_config import load_rl_config

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PRESET_NAMES = ("pytorch_cpu_ppo.yaml", "pytorch_cpu_grpo.yaml")
_LOCAL_MODEL_PATH_TEMPLATE = "/absolute/path/to/local-transformers-model"


def _preset_path(name: str) -> Path:
    return _REPOSITORY_ROOT / "configs" / "rl" / name


def _preset_payload(name: str) -> dict[str, object]:
    payload = yaml.safe_load(_preset_path(name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize("name", _PRESET_NAMES)
def test_shipped_presets_ingest_the_bundled_strict_pair_dataset(name: str) -> None:
    config = load_rl_config(_preset_path(name))

    assert config.dataset.format == "jsonl"
    dataset_path = _REPOSITORY_ROOT / config.dataset.train_path
    requests = list(FlatJSONLAdapter(dataset_path).iter_requests())
    assert requests
    assert all(
        request.metadata["schema_version"] == "reward-integrity-rl-pairs-v1" for request in requests
    )


@pytest.mark.parametrize("name", _PRESET_NAMES)
def test_shipped_presets_require_an_explicit_local_model_path(name: str) -> None:
    config = load_rl_config(_preset_path(name))

    assert config.policy.model_name == _LOCAL_MODEL_PATH_TEMPLATE


@pytest.mark.parametrize("name", _PRESET_NAMES)
def test_shipped_presets_explicitly_enable_gradient_clipping(name: str) -> None:
    payload = _preset_payload(name)
    algorithm = payload["algorithm"]
    assert isinstance(algorithm, dict)
    assert "max_grad_norm" in algorithm

    config = load_rl_config(_preset_path(name))
    assert config.algorithm.max_grad_norm is not None
    assert config.algorithm.max_grad_norm > 0.0


def test_shipped_grpo_preset_explicitly_enables_stochastic_sampling() -> None:
    name = "pytorch_cpu_grpo.yaml"
    payload = _preset_payload(name)
    policy = payload["policy"]
    assert isinstance(policy, dict)
    assert {"do_sample", "temperature", "top_p"}.issubset(policy)

    config = load_rl_config(_preset_path(name))
    assert config.policy.do_sample is True
    assert config.policy.temperature > 0.0
    assert 0.0 < config.policy.top_p <= 1.0


def test_development_extras_support_pytest_8_and_9_but_not_10() -> None:
    pyproject = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    for extra in ("dev", "rl-dev"):
        pytest_requirements = [item for item in optional[extra] if item.startswith("pytest")]
        assert pytest_requirements == ["pytest>=8.0,<10"]
    assert not [item for item in optional["all"] if item.startswith("pytest")]


def test_training_guides_keep_model_weight_and_compatibility_maturity_distinct() -> None:
    guide_paths = (
        _REPOSITORY_ROOT / "gepa_mindfulness" / "training" / "README.md",
        _REPOSITORY_ROOT / "docs" / "NEWCOMER_GUIDE.md",
        _REPOSITORY_ROOT / "docs" / "execution_flow.md",
    )
    guides = [path.read_text(encoding="utf-8") for path in guide_paths]

    for guide in guides:
        lowered = guide.lower()
        assert "gepa rl" in guide
        assert "sole canonical" in lowered
        assert "compatibility" in lowered
        assert "optimizer step" in lowered
        assert "unchanged" in lowered

    training_readme = guides[0]
    assert "frozen dataclasses" in training_readme
    pydantic_index = training_readme.index("Pydantic")
    assert "not evidence of model-weight training" in training_readme[pydantic_index - 180 :]
