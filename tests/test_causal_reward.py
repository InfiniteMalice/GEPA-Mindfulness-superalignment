"""Public adapter compatibility, callback trust and explicit optimizer admission."""

# Standard library
import json
import math

# Third-party
import pytest

# Local
from causal_reward_fixtures import make_accounting, make_context, make_policy, make_verdict
from gepa_mindfulness.training.causal_reward import CausalRewardAdapter, CausalRewardDecision
from gepa_mindfulness.training.causal_reward_records import CausalRewardContext
from gepa_mindfulness.training.reward_pipeline import RewardPipeline, RewardResult


class RecordingPipeline:
    """Return an exact sentinel, recording every request and observation."""

    overlay_enabled = False

    def __init__(self, total=-0.0):
        self.result = RewardResult(total, total, total)
        self.calls = []

    def score(self, request, *, observation=None):
        """Record the one target scoring call."""
        self.calls.append((request, observation))
        return self.result


@pytest.mark.parametrize("enabled,weight,cap", [(False, 0.25, 0.1), (True, 0, 0)])
def test_disabled_returns_same_result_without_optional_validation(enabled, weight, cap):
    """No-op policies ignore invalid optional data, preserving identity and signed zero."""
    pipeline, request, calls = RecordingPipeline(), make_context().target, []
    policy = make_policy(enabled=enabled, weight=weight, max_adjustment=cap)
    adapter = CausalRewardAdapter(
        pipeline,
        policy=policy,
        authenticate_verdict=lambda p: calls.append(p),
        authorize_use=lambda p: calls.append(p),
    )
    observation = object()
    result = adapter.score(
        request, observation=observation, context="bad", verdict=1, accounting=[]
    )
    assert result is pipeline.result
    assert pipeline.calls == [(request, observation)]
    assert calls == []
    assert math.copysign(1, result.total) == -1


def inputs(total=0.8, **facts):
    """Return complete authored arguments for enabled scoring."""
    context = make_context()
    policy = make_policy(enabled=True, weight=0.25, max_adjustment=0.1)
    verdict = make_verdict(context, policy, **facts)
    accounting = make_accounting(context, policy, verdict, total)
    return policy, dict(context=context, verdict=verdict, accounting=accounting)


def run_score(*, total=0.8, auth=lambda p: True, use=lambda p: True, **facts):
    """Score authored evidence through a recording pipeline."""
    policy, values = inputs(total, **facts)
    adapter = CausalRewardAdapter(
        RecordingPipeline(total), policy=policy, authenticate_verdict=auth, authorize_use=use
    )
    return adapter.score(values["context"].target, **values)


@pytest.mark.parametrize("which", ["auth", "use"])
@pytest.mark.parametrize("behavior", ["one", "truthy", "false", "none", "raise", "mutate"])
def test_enabled_authentication_is_detached_and_exact(which, behavior):
    """Only unchanged payloads with a built-in True result cross the host boundary."""
    policy, values = inputs()
    original = values["context"].to_dict()

    def callback(payload):
        if behavior == "raise":
            raise RuntimeError("sensitive callback exception")
        if behavior == "mutate":
            payload["context"]["sources"][0][1]["training_eligibility"] = "HIDDEN_EVAL"
            return True
        return {"one": 1, "truthy": object(), "false": False, "none": None}[behavior]

    kwargs = {"authenticate_verdict": lambda p: True, "authorize_use": lambda p: True}
    kwargs["authenticate_verdict" if which == "auth" else "authorize_use"] = callback
    result = CausalRewardAdapter(RecordingPipeline(0.8), policy=policy, **kwargs).score(
        values["context"].target, **values
    )
    assert result.disposition == "withheld"
    assert result.deduction == 0
    assert result.diagnostic_total == 0.8
    assert values["context"].to_dict() == original
    assert "sensitive callback exception" not in result.audit
    with pytest.raises(ValueError):
        result.optimizer_reward()


@pytest.mark.parametrize("missing", ["context", "verdict", "accounting"])
def test_missing_inputs_withhold_without_incomplete_callbacks(missing):
    """A callback is never asked to authorize an invented incomplete payload."""
    policy, values = inputs()
    target = values["context"].target
    calls = []
    del values[missing]
    adapter = CausalRewardAdapter(
        RecordingPipeline(0.8),
        policy=policy,
        authenticate_verdict=lambda p: calls.append("auth") or True,
        authorize_use=lambda p: calls.append("use") or True,
    )
    result = adapter.score(target, **values)
    assert result.disposition == "withheld"
    assert "use" not in calls
    assert ("auth" in calls) is (missing == "accounting")


def test_provider_cannot_mutate_detached_request():
    """A mutating provider is rejected without altering the caller's nested request."""
    policy, values = inputs()
    target = values["context"].target

    class MutatingProvider:
        def score(self, request):
            request.trajectory.sampling_parameters["metadata"][
                "training_eligibility"
            ] = "HIDDEN_EVAL"
            return 0.8

    adapter = CausalRewardAdapter(RewardPipeline(MutatingProvider()), policy=policy)
    with pytest.raises(ValueError, match="mutat"):
        adapter.score(target, **values)
    assert target.trajectory.sampling_parameters["metadata"]["training_eligibility"] == "TRAIN"


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_total", 1),
        ("context_digest", "b" * 64),
        ("policy_digest", "b" * 64),
        ("verdict_digest", "b" * 64),
        ("provider_id", "other"),
        ("provider_version", "other"),
        ("provider_config_digest", "b" * 64),
    ],
)
def test_baseline_accounting_uses_actual_pipeline_result(field, value):
    """Accounting binds actual scalar, complete records and policy's provider configuration."""
    from dataclasses import replace

    policy, values = inputs()
    values["accounting"] = replace(values["accounting"], **{field: value})
    adapter = CausalRewardAdapter(
        RecordingPipeline(0.8),
        policy=policy,
        authenticate_verdict=lambda p: True,
        authorize_use=lambda p: True,
    )
    with pytest.raises(ValueError, match="mismatch"):
        adapter.score(values["context"].target, **values)


@pytest.mark.parametrize("actual,accounted", [(-0.0, 0.0), (0.0, -0.0), (1, 1.0), (1.0, 1)])
def test_baseline_encoding_mismatch_rejected_before_callbacks(actual, accounted):
    """Numeric equality cannot authenticate accounting for a different encoded scalar."""
    policy, values = inputs(accounted)
    calls, pipeline = [], RecordingPipeline(actual)
    adapter = CausalRewardAdapter(
        pipeline,
        policy=policy,
        authenticate_verdict=lambda p: calls.append("authenticate") or True,
        authorize_use=lambda p: calls.append("authorize") or True,
    )
    with pytest.raises(ValueError, match="accounting baseline mismatch"):
        adapter.score(values["context"].target, **values)
    assert len(pipeline.calls) == 1
    assert calls == []


@pytest.mark.parametrize("total,encoded", [(-0.0, "-0.0"), (0.0, "0.0"), (1, "1"), (1.0, "1.0")])
def test_matching_baseline_encoding_reaches_authorization(total, encoded):
    """Exact accounting preserves scalar encodings in the complete use request."""
    requests = []
    result = run_score(total=total, use=lambda p: requests.append(p) or True)
    assert result.disposition == "applied"
    assert result.optimizer_reward() == pytest.approx(total - 0.1)
    assert len(requests) == 1
    assert json.dumps(requests[0]["baseline_total"]) == encoded
    assert json.dumps(requests[0]["accounting"]["baseline_total"]) == encoded


@pytest.mark.parametrize(
    "layers,reason,allowed",
    [
        ((), "accounting_unknown", False),
        ((("provider", "unknown"),), "accounting_unknown", False),
        ((("provider", "already_counted"),), "already_counted", True),
    ],
)
def test_semantic_accounting_dispositions(layers, reason, allowed):
    """Same-causal-failure overlap preserves the baseline and never duplicates a debit."""
    from dataclasses import replace

    policy, values = inputs()
    values["accounting"] = replace(values["accounting"], layers=layers)
    result = CausalRewardAdapter(
        RecordingPipeline(0.8),
        policy=policy,
        authenticate_verdict=lambda p: True,
        authorize_use=lambda p: True,
    ).score(values["context"].target, **values)
    assert reason in result.reasons
    assert result.deduction == 0
    if allowed:
        assert result.optimizer_reward() == 0.8
    else:
        with pytest.raises(ValueError):
            result.optimizer_reward()


@pytest.mark.parametrize("severe", [True, False, None])
def test_severe_remains_visible_when_other_gates_fail(severe):
    """Missing accounting and rejected provenance cannot erase an authenticated severe event."""
    policy, values = inputs()
    data = values["context"].to_dict()
    data["sources"][0][1]["source_record"] = {"training_eligibility": "HIDDEN_EVAL"}
    context = CausalRewardContext.from_dict(data)
    verdict = make_verdict(context, policy, severe_violation=severe)
    result = CausalRewardAdapter(
        RecordingPipeline(0.8), policy=policy, authenticate_verdict=lambda p: True
    ).score(context.target, context=context, verdict=verdict)
    assert (
        result.severe_status
        == {True: "verified_violation", False: "verified_clear", None: "unknown"}[severe]
    )
    assert "admission_rejected" in result.reasons
    assert "accounting_unknown" in result.reasons
    assert ("severe_unknown" in result.reasons) is (severe is None)
    assert result.diagnostic_total == 0.8
    with pytest.raises(ValueError):
        result.optimizer_reward()


def test_adjustment_is_bounded_negative_and_never_accumulates():
    """Fresh scoring computes one bounded debit and performs fresh host checks each time."""
    policy, values = inputs()
    calls, pipeline = [], RecordingPipeline(0.8)
    adapter = CausalRewardAdapter(
        pipeline,
        policy=policy,
        authenticate_verdict=lambda p: calls.append(p) or True,
        authorize_use=lambda p: calls.append(p) or True,
    )
    for _ in range(2):
        result = adapter.score(values["context"].target, **values)
        assert result.deduction == pytest.approx(0.1)
        assert result.optimizer_reward() == pytest.approx(0.7)
        assert result.disposition == "applied"
        assert result.reasons[-1] == "adjustment_applied"
        assert not hasattr(result, "total")
        assert not hasattr(type(result), "from_dict")
        with pytest.raises(TypeError):
            float(result)
        report = result.to_dict()
        assert report["training_eligibility"] == "DEVELOPMENT"
        assert "_authorized" not in report
        with pytest.raises(TypeError):
            CausalRewardDecision(**report)
    assert len(calls) == 4 and len(pipeline.calls) == 2
    assert calls[1]["baseline_total"] == 0.8
    assert calls[1]["integrity_snapshot"] is None


@pytest.mark.parametrize(
    "facts",
    [
        {"after_correct": True},
        {"before_correct": False},
        {"change_justified": True},
        {"single_factor_verified": False},
    ],
)
def test_exemptions_preserve_signed_zero(facts):
    """No causal bonus or arithmetic changes an admitted nonpenalizable baseline."""
    result = run_score(total=-0.0, **facts)
    assert result.deduction == 0
    assert math.copysign(1, result.optimizer_reward()) == -1


def test_arithmetic_overflow_rejected():
    """Finite inputs do not permit a nonfinite adjusted optimizer value."""
    policy = make_policy(enabled=True, weight=1e308, max_adjustment=1e308)
    context = make_context()
    verdict = make_verdict(context, policy)
    accounting = make_accounting(context, policy, verdict, -1e308)
    adapter = CausalRewardAdapter(
        RecordingPipeline(-1e308),
        policy=policy,
        authenticate_verdict=lambda p: True,
        authorize_use=lambda p: True,
    )
    with pytest.raises(ValueError, match="finite"):
        adapter.score(context.target, context=context, verdict=verdict, accounting=accounting)


def test_audit_retains_original_callback_inputs():
    """The audit records exact source objects and independent gate outcomes."""
    result = run_score()
    audit = json.loads(result.audit)
    assert audit["context"] == make_context().to_dict()
    assert audit["gates"]["authenticated"] is True
    assert audit["gates"]["use_authorized"] is True
