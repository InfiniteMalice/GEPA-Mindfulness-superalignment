from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime
from typing import Any

import pytest

from evaluation import experimental_overlays, plan_v5_cells
from evaluation.cases.registry import load_case_manifest
from gepa_mindfulness.core.rewards import (
    GEPARewardCalculator,
    HallucinationConfig,
    RewardWeights,
)
from gepa_mindfulness.verification import (
    ActionAuthorityPolicy,
    AuthorityGrantRegistry,
    RuntimeCapability,
    RuntimeRole,
    action_record_digest,
    authorize_action,
)
from mindful_trace_gepa import ActionRecord


@dataclass
class _FixedClock:
    current: datetime

    def now(self) -> datetime:
        return self.current


def test_default_config_enables_nothing_and_changes_no_canonical_consumers() -> None:
    config = experimental_overlays.ExperimentalOverlayConfig()
    cases_before = load_case_manifest()
    cells_before = _cells()
    reward_before = _reward_total()
    authority_before = _denied_authority_result()

    assert experimental_overlays.enabled_overlays(config) == ()

    assert load_case_manifest() == cases_before
    assert _cells() == cells_before
    assert _reward_total() == reward_before
    assert _denied_authority_result() == authority_before


def test_explicit_flags_expose_only_selected_diagnostic_declarations() -> None:
    config = experimental_overlays.ExperimentalOverlayConfig(
        competing_hypotheses=True,
        declarative_orchestration_scope=True,
    )

    enabled = experimental_overlays.enabled_overlays(config)

    assert tuple(item.id for item in enabled) == (
        "competing_hypotheses",
        "declarative_orchestration_scope",
    )
    assert tuple(item.allowed_outputs for item in enabled) == (
        ("hypothesis_set",),
        ("orchestration_scope_declaration",),
    )
    assert all("canonical_case_creation" in item.prohibited_effects for item in enabled)
    assert all("direct_optimizer_reward" in item.prohibited_effects for item in enabled)


def test_external_mapping_rejects_unknown_keys_and_non_boolean_flags() -> None:
    with pytest.raises(ValueError, match="unknown overlay configuration keys"):
        experimental_overlays.ExperimentalOverlayConfig.from_mapping({"execute": True})

    with pytest.raises(ValueError, match="built-in bool"):
        experimental_overlays.ExperimentalOverlayConfig.from_mapping({"competing_hypotheses": 1})


def test_external_mapping_defaults_omitted_flags_and_config_is_frozen() -> None:
    config = experimental_overlays.ExperimentalOverlayConfig.from_mapping(
        {"mechanistic_circuit_audit": True}
    )

    assert tuple(item.id for item in experimental_overlays.enabled_overlays(config)) == (
        "mechanistic_circuit_audit",
    )
    with pytest.raises(FrozenInstanceError):
        config.mechanistic_circuit_audit = False  # type: ignore[misc]


def test_enabled_overlays_rejects_config_subclasses_and_mutated_fields() -> None:
    class DerivedConfig(experimental_overlays.ExperimentalOverlayConfig):
        pass

    with pytest.raises(ValueError, match="exact ExperimentalOverlayConfig"):
        experimental_overlays.enabled_overlays(DerivedConfig())

    config = experimental_overlays.ExperimentalOverlayConfig()
    object.__setattr__(config, "competing_hypotheses", "yes")
    with pytest.raises(ValueError, match="built-in bool"):
        experimental_overlays.enabled_overlays(config)


def _cells() -> tuple[object, ...]:
    return plan_v5_cells(
        case_ids=(1, 14, 17),
        stripe_ids=("NONE", "TOOL_ERROR"),
        repeats=2,
        base_seed=17,
        model_version="model-v5",
        harness_version="harness-v5",
    )


def _reward_total() -> float:
    calculator = GEPARewardCalculator(
        weights=RewardWeights(0.4, 0.3, 0.2, 0.1),
        hallucination=HallucinationConfig(0.8, -1.0, -0.25, 0.5, -0.5),
    )
    return calculator.compute_reward(
        response="verified answer",
        reference_answers=("verified answer",),
        gepa_scores={"alignment": 0.5},
        imperatives=None,
        confidence=0.9,
        trace_summary={},
    ).total


def _denied_authority_result() -> tuple[Any, ...]:
    action = ActionRecord("action-overlay", "read", True, "repo:docs", "prediction-1")
    policy = ActionAuthorityPolicy(
        "policy-overlay",
        action.action_id,
        action_record_digest(action),
        action.authorization_scope,
        RuntimeCapability.READ,
    )
    registry = AuthorityGrantRegistry.enroll((), (policy,))
    decision = authorize_action(
        action,
        principal_id="auditor-1",
        role=RuntimeRole.AUDITOR,
        capability=RuntimeCapability.READ,
        policy_id=policy.policy_id,
        grant_registry=registry,
        grant_ids=(),
        clock=_FixedClock(datetime.fromisoformat("2026-09-10T12:00:00+00:00")),
    )
    return (
        decision.authorized,
        decision.grant_id,
        decision.reason,
        decision.capability,
        decision.authorization_scope,
    )
