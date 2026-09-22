"""Software contracts for optional latent-to-language diagnostics."""

# Standard library
import json
from dataclasses import FrozenInstanceError, replace
from importlib import import_module, util

# Third-party
import pytest

# Local
from semantic_intent_robustness.internal_state_trajectory import (
    MeasurementStatus,
    SoTStateSnapshot,
)


@pytest.fixture
def api():
    module = "semantic_intent_robustness.latent_language_transition"
    assert util.find_spec(module) is not None, "REC-019 transition diagnostic is missing"
    return import_module(module)


def test_optional_diagnostic_api_is_available():
    assert util.find_spec("semantic_intent_robustness.latent_language_transition") is not None


def state(value=0.0, *, turn=0, **changes):
    snapshot = SoTStateSnapshot(
        snapshot_id=f"state-{turn}",
        conversation_id="conversation-1",
        turn_index=turn,
        adapter_name="test-telemetry",
        source_model_id="test-model",
        backend_id="test-backend",
        layer_sources=("layer-1",),
        feature_schema="four-unit-features-v1",
        provenance=(f"telemetry-{turn}",),
        measurement_status=MeasurementStatus.MEASURED_INTERNAL,
        evidence_status="observed",
        source_kind="internal",
        local_organization=value,
        progress_magnitude=value,
        directional_consistency=value,
        predictive_uncertainty=value,
    )
    return replace(snapshot, **changes)


def delta(api, value, **changes):
    arguments = dict(
        value=value,
        comparable=True,
        origin=api.PublicMeasurementOrigin.OBSERVED,
        metric_id="public-difference-v1",
        normalization="unit-interval-v1",
        endpoint_refs=("public-before", "public-after"),
        provenance=("public-audit",),
    )
    arguments.update(changes)
    return api.PublicDelta(**arguments)


def audit(api, **changes):
    arguments = dict(
        assessment_id="transition-1",
        before=state(),
        after=state(0.8, turn=1),
        output=delta(api, 0.4),
        action=delta(api, 1.0),
        provenance=("paired-observation",),
        enabled=True,
    )
    arguments.update(changes)
    return api.audit_latent_language_transition(**arguments)


def test_agreeing_changes_keep_output_and_action_separate(api):
    result = audit(api)
    assert result.status is api.TransitionStatus.CO_CHANGE_OBSERVED
    assert result.latent_delta == pytest.approx(0.8)
    assert result.output_delta == 0.4
    assert result.action_delta == 1.0
    assert result.output_transfer_ratio == pytest.approx(0.5)
    assert result.action_transfer_ratio == pytest.approx(1.25)
    assert result.latent_comparable is True
    assert result.measurement_origins == (
        MeasurementStatus.MEASURED_INTERNAL,
        MeasurementStatus.MEASURED_INTERNAL,
    )


def test_latent_change_without_language_change_is_decoupling(api):
    result = audit(api, output=delta(api, 0.0), action=delta(api, 0.0))
    assert result.status is api.TransitionStatus.LATENT_LANGUAGE_DECOUPLING
    assert result.output_transfer_ratio == 0.0


def test_action_change_does_not_mask_language_decoupling(api):
    result = audit(api, output=delta(api, 0.0))
    assert result.status is api.TransitionStatus.LATENT_LANGUAGE_DECOUPLING
    assert result.action_delta == 1.0


def test_language_change_without_measured_latent_change(api):
    result = audit(api, after=state(0.0, turn=1))
    assert result.status is api.TransitionStatus.LANGUAGE_CHANGE_WITHOUT_MATCHED_LATENT_SIGNAL
    assert result.output_transfer_ratio is None
    assert result.action_transfer_ratio is None


@pytest.mark.parametrize("value,status", [(0.01, "NO_SUBSTANTIAL_CHANGE"), (0.1, "INDETERMINATE")])
def test_threshold_gap_is_not_silently_classified(api, value, status):
    result = audit(api, after=state(value, turn=1), output=delta(api, value))
    assert result.status is getattr(api.TransitionStatus, status)


@pytest.mark.parametrize("changes", [dict(feature_schema="other"), dict(backend_id="other")])
def test_incomparable_latent_spaces_fail_closed(api, changes):
    result = audit(api, after=state(0.8, turn=1, **changes))
    assert result.status is api.TransitionStatus.INCOMPARABLE
    assert result.latent_comparable is False
    assert result.latent_delta is None
    assert result.output_transfer_ratio is None
    assert result.output_delta == 0.4


@pytest.mark.parametrize("explicit", [False, True])
def test_unavailable_internal_state_keeps_public_behavior_available(api, explicit):
    after = None
    if explicit:
        after = state(
            turn=1,
            measurement_status=MeasurementStatus.UNAVAILABLE,
            evidence_status="unavailable",
            source_kind="unavailable",
            local_organization=None,
            progress_magnitude=None,
            directional_consistency=None,
            predictive_uncertainty=None,
        )
    result = audit(api, after=after)
    assert result.status is api.TransitionStatus.UNAVAILABLE
    assert result.latent_delta is None
    assert result.action_delta == 1.0


def test_transcript_proxy_preserves_origin_without_measured_anomaly(api):
    changes = dict(
        measurement_status=MeasurementStatus.TRANSCRIPT_PROXY,
        source_kind="transcript",
        layer_sources=(),
    )
    result = audit(
        api, before=state(**changes), after=state(0.8, turn=1, **changes), output=delta(api, 0.0)
    )
    assert result.status is api.TransitionStatus.PROXY_ONLY
    assert result.latent_delta == pytest.approx(0.8)
    assert result.measurement_origins == (MeasurementStatus.TRANSCRIPT_PROXY,) * 2
    assert result.output_transfer_ratio is None


@pytest.mark.parametrize("origin", ["HEURISTIC", "UNAVAILABLE"])
def test_public_proxy_or_unavailable_does_not_support_measured_anomaly(api, origin):
    measurement = delta(
        api,
        None if origin == "UNAVAILABLE" else 0.0,
        comparable=origin != "UNAVAILABLE",
        origin=getattr(api.PublicMeasurementOrigin, origin),
    )
    result = audit(api, output=measurement)
    assert result.status is getattr(
        api.TransitionStatus, "UNAVAILABLE" if origin == "UNAVAILABLE" else "PROXY_ONLY"
    )
    assert result.output_transfer_ratio is None
    assert result.action_transfer_ratio == pytest.approx(1.25)


def test_incomparable_public_measurement_is_not_zero(api):
    result = audit(api, output=delta(api, None, comparable=False))
    assert result.status is api.TransitionStatus.INCOMPARABLE
    assert result.output_delta is None


def test_disabled_default_emits_no_computed_diagnostic(api):
    result = api.audit_latent_language_transition(
        assessment_id="disabled",
        before=state(),
        after=state(1.0, turn=1),
        output=delta(api, 0.0),
        action=delta(api, 0.0),
        provenance=("pair",),
    )
    assert result.status is api.TransitionStatus.DISABLED
    assert result.latent_delta is None
    assert result.output_transfer_ratio is None


def test_result_is_immutable_and_serializes_provenance_without_authority(api):
    result = audit(api)
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["before"]["provenance"] == ["telemetry-0"]
    assert payload["output"]["provenance"] == ["public-audit"]
    assert payload["provenance"] == ["paired-observation"]
    forbidden = {"passed", "reward", "runtime_authority", "training_eligible", "deploy"}
    assert forbidden.isdisjoint(payload)
    with pytest.raises(FrozenInstanceError):
        result.status = api.TransitionStatus.DISABLED


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 1.1, True, "0.2"])
def test_invalid_public_delta_is_rejected(api, value):
    with pytest.raises(ValueError):
        delta(api, value)


@pytest.mark.parametrize(
    "changes",
    [
        dict(provenance=[]),
        dict(provenance=()),
        dict(provenance=({"mutable": "value"},)),
        dict(endpoint_refs=("one",)),
        dict(endpoint_refs=("same", "same")),
        dict(metric_id=""),
        dict(normalization=""),
        dict(comparable="true"),
        dict(origin="observed"),
        dict(comparable=False),
    ],
)
def test_malformed_public_metadata_is_rejected(api, changes):
    with pytest.raises(ValueError):
        delta(api, 0.2, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(enabled=1),
        dict(provenance=()),
        dict(provenance=["mutable"]),
        dict(substantial_threshold=float("nan")),
        dict(substantial_threshold=0.0),
        dict(negligible_threshold=0.3),
        dict(before="state"),
        dict(output=0.2),
    ],
)
def test_malformed_assessment_inputs_are_rejected(api, changes):
    with pytest.raises(ValueError):
        audit(api, **changes)


@pytest.mark.parametrize(
    "changes", [dict(conversation_id="unrelated"), dict(turn_index=0), dict(snapshot_id="state-0")]
)
def test_snapshot_pair_must_have_bound_ordered_identity(api, changes):
    with pytest.raises(ValueError):
        audit(api, after=state(0.8, turn=1, **changes))


def test_tiny_nonzero_denominator_never_emits_infinite_ratio(api):
    result = audit(api, after=state(1e-320, turn=1), negligible_threshold=0.0)
    assert result.output_transfer_ratio is None
    assert result.action_transfer_ratio is None
    json.dumps(result.to_dict(), allow_nan=False)
