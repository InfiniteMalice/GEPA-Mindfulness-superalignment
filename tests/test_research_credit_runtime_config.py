"""Optional credit configuration must bind resume identity and host setup."""

from dataclasses import replace

import pytest
from test_research_credit_runtime import policy

from gepa_mindfulness.training.engine import _config_hash
from gepa_mindfulness.training.runtime_config import (
    RLRunConfig,
    RuntimeConfig,
    translate_legacy_config,
)


def test_enabled_credit_policy_is_in_checkpoint_identity_and_disabled_is_compatible():
    baseline = RLRunConfig()
    assert baseline.research_credit is None
    assert _config_hash(baseline) == (
        "9f0306ad79e6ed93e0bc40b3918270e12538960e4061456be7f91d6fb1f6f1b2"
    )
    inactive = replace(baseline, research_credit=policy(enabled=False))
    assert _config_hash(inactive) == _config_hash(baseline)
    enabled = replace(baseline, research_credit=policy())
    assert _config_hash(enabled) != _config_hash(baseline)
    changed = replace(enabled, research_credit=policy(coefficients=(("knowledge", 0.25),)))
    assert _config_hash(changed) != _config_hash(enabled)
    settings = policy().to_dict()
    settings.pop("schema_version")
    assert RLRunConfig.from_mapping({"research_credit": settings}) == enabled


def test_legacy_translation_preserves_explicit_credit_enable():
    settings = policy().to_dict()
    settings.pop("schema_version")
    with pytest.warns(DeprecationWarning):
        result = translate_legacy_config({"model_name": "local", "research_credit": settings})
    assert result.research_credit == policy()


def test_declared_policy_rejects_unknown_settings_and_unsupported_runtime():
    settings = policy().to_dict()
    settings.pop("schema_version")
    with pytest.raises(ValueError):
        RLRunConfig.from_mapping({"research_credit": {**settings, "skip_review": True}})
    with pytest.raises(ValueError):
        replace(
            RLRunConfig(),
            runtime=RuntimeConfig(backend="cuda", device="cuda:0"),
            research_credit=policy(),
        )
