"""Real reward composition and host trust-boundary regression controls."""

from dataclasses import replace

import pytest
from causal_reward_fixtures import make_accounting, make_context, make_policy, make_verdict

from gepa_mindfulness.core.evidence import EvidenceSourceKind
from gepa_mindfulness.core.reward_integrity import RewardIntegrityCalculator, RewardObservation
from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute
from gepa_mindfulness.core.rewards import GEPARewardCalculator, HallucinationConfig, RewardWeights
from gepa_mindfulness.training.causal_reward import CausalRewardAdapter
from gepa_mindfulness.training.causal_reward_records import CausalRewardContext
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.reward_pipeline import RewardPipeline
from gepa_mindfulness.training.trajectory import Trajectory


class CorrectnessProvider:
    """Exercise unchanged production reward formulas with authored correct-answer evidence."""

    def __init__(self):
        self.calls = 0
        self.calculator = GEPARewardCalculator(
            weights=RewardWeights(1, 1, 1, 1),
            hallucination=HallucinationConfig(0.75, -1, -0.5, 0.5, -1),
        )

    def score(self, request):
        """Compute the existing correctness penalty without observing the comparison arm."""
        self.calls += 1
        return self.calculator.compute_reward(
            response=request.trajectory.response,
            reference_answers=("correct",),
            gepa_scores=None,
            imperatives=None,
            confidence=1.0,
            trace_summary={},
        )


def enabled_policy():
    """Return explicit authored experiment settings, not live authorization."""
    return make_policy(enabled=True, weight=0.25, max_adjustment=0.1)


@pytest.mark.parametrize("route", list(VerificationRoute))
@pytest.mark.parametrize("overlap", ["not_counted", "already_counted", "unknown", "missing"])
def test_existing_overlay_and_causal_accounting_do_not_double_penalize(route, overlap):
    """Incorrectness stays penalized; only separately accounted causal failures add a debit."""
    context, policy = make_context(), enabled_policy()
    evaluator = make_verdict(context, policy).evaluator
    provenance = RewardProvenance(
        "verified_causal_penalty",
        "authored comparison",
        route,
        evidence_refs=(
            context.target_event_refs if route is VerificationRoute.OBSERVABLE_EVIDENCE else ()
        ),
        evaluator=evaluator if route is VerificationRoute.TRUSTED_EVALUATOR else None,
    )
    verdict = make_verdict(context, policy, provenance=provenance)
    refs = context.target_event_refs
    observation = RewardObservation(
        objective_fidelity=-1,
        observable_references=refs,
        observable_evidence={"objective_fidelity": refs},
        reward_component_provenance={
            "objective_fidelity": RewardProvenance(
                "objective_fidelity",
                "authored causal-overlap review",
                route,
                evidence_refs=refs if route is VerificationRoute.OBSERVABLE_EVIDENCE else (),
                evaluator=evaluator if route is VerificationRoute.TRUSTED_EVALUATOR else None,
            )
        },
    )
    provider = CorrectnessProvider()
    pipeline = RewardPipeline(
        provider, integrity_calculator=RewardIntegrityCalculator(), overlay_weight=0.4
    )
    baseline = pipeline.score(context.target, observation=observation)
    assert baseline.base_result.hallucination == -1
    assert baseline.total < baseline.base_reward
    layers = (("provider", "not_counted"),)
    if overlap != "missing":
        layers += (("reward_integrity", overlap),)
    accounting = make_accounting(context, policy, verdict, baseline.total, layers=layers)
    calls = []
    adapter = CausalRewardAdapter(
        pipeline,
        policy=policy,
        authenticate_verdict=lambda p: True,
        authorize_use=lambda p: calls.append(p) or True,
    )
    result = adapter.score(
        context.target,
        observation=observation,
        context=context,
        verdict=verdict,
        accounting=accounting,
    )
    assert provider.calls == 2  # One separate baseline measurement and one adapter call.
    assert result.baseline_total == baseline.total
    assert result.base_result.base_result == baseline.base_result
    assert calls[0]["integrity_snapshot"]["aggregate"] == baseline.integrity_breakdown.aggregate
    assert calls[0]["integrity_snapshot"]["reward_component_provenance"]
    if overlap == "not_counted":
        assert result.optimizer_reward() == pytest.approx(baseline.total - 0.1)
    elif overlap == "already_counted":
        assert result.optimizer_reward() == baseline.total
        assert result.deduction == 0
    else:
        assert result.diagnostic_total == baseline.total
        with pytest.raises(ValueError):
            result.optimizer_reward()


@pytest.mark.parametrize("restriction", ["DEVELOPMENT", "REGRESSION", "HIDDEN_EVAL", "failed_v5"])
def test_source_relabel_cannot_authorize_reward(restriction):
    """Retaining the original restricted source defeats an outer TRAIN label."""
    data = make_context().to_dict()
    nested = {"training_eligibility": restriction}
    if restriction == "failed_v5":
        nested = {"v5_record": {"case": {}, "system": {}, "outcome": {"passed": False}}}
    data["sources"][0][1]["source_record"] = nested
    context, policy = CausalRewardContext.from_dict(data), enabled_policy()
    verdict = make_verdict(context, policy, severe_violation=True)
    pipeline = RewardPipeline(CorrectnessProvider())
    result = CausalRewardAdapter(
        pipeline, policy=policy, authenticate_verdict=lambda p: True
    ).score(context.target, context=context, verdict=verdict)
    assert result.severe_status == "verified_violation"
    assert "admission_rejected" in result.reasons and "accounting_unknown" in result.reasons
    assert result.deduction == 0
    with pytest.raises(ValueError):
        result.optimizer_reward()


@pytest.mark.parametrize("revocation", ["source", "evaluator", "policy"])
def test_current_host_revocation_is_rechecked(revocation):
    """A previously exported report cannot bypass the next call's current host permissions."""
    context, policy = make_context(), enabled_policy()
    verdict = make_verdict(context, policy)
    provider = CorrectnessProvider()
    baseline = provider.score(context.target).total
    accounting = make_accounting(context, policy, verdict, baseline)
    permitted = {"source": True, "evaluator": True, "policy": True}
    adapter = CausalRewardAdapter(
        RewardPipeline(provider),
        policy=policy,
        authenticate_verdict=lambda p: permitted["source"] and permitted["evaluator"],
        authorize_use=lambda p: all(permitted.values()),
    )
    first = adapter.score(context.target, context=context, verdict=verdict, accounting=accounting)
    assert first.deduction > 0
    report = first.to_dict()
    permitted[revocation] = False
    second = adapter.score(context.target, context=context, verdict=verdict, accounting=accounting)
    assert second.diagnostic_total == baseline
    with pytest.raises(ValueError):
        second.optimizer_reward()
    assert report["training_eligibility"] == "DEVELOPMENT"


@pytest.mark.parametrize(
    "host_reason", ["stripped_origin", "same_event_alias", "renamed_component"]
)
def test_alternate_event_alias_requires_host_semantic_accounting(host_reason):
    """Local hashes cannot authenticate source completeness or detect semantic aliases."""
    context, policy = make_context(), enabled_policy()
    if host_reason == "same_event_alias":
        data = context.to_dict()
        for reference in (
            data["target"]["observable_references"][0],
            data["target"]["trajectory"]["evidence_references"][0],
            data["target_event_refs"][0],
            data["sources"][1][0],
        ):
            reference["reference_id"] = "target-alias"
        context = CausalRewardContext.from_dict(data)
    verdict = make_verdict(context, policy)
    pipeline = RewardPipeline(CorrectnessProvider())
    baseline = pipeline.score(context.target).total
    layers = (("provider", "not_counted"), (host_reason, "not_counted"))
    accounting = make_accounting(context, policy, verdict, baseline, layers=layers)
    reviewed = []

    def host_review(payload):
        reviewed.append(payload["accounting"]["layers"])
        # This authored host has retained origin and semantic identities beyond local hashes.
        if host_reason == "stripped_origin":
            origin_registry = {"before": "HIDDEN_EVAL", "target": "TRAIN"}
            return all(
                origin_registry[ref["reference_id"]] == source["training_eligibility"]
                for ref, source in payload["context"]["sources"]
            )
        if host_reason == "same_event_alias":
            aliases = {"target": "event-1", "target-alias": "event-1"}
            counted_events = {"event-1"}
            return all(
                aliases[ref["reference_id"]] not in counted_events
                for ref in payload["context"]["target_event_refs"]
            )
        actual_layer_meanings = {"renamed_component": "verified_causal_penalty"}
        return not any(
            actual_layer_meanings.get(name) == "verified_causal_penalty" and status == "not_counted"
            for name, status in payload["accounting"]["layers"]
        )

    result = CausalRewardAdapter(
        pipeline, policy=policy, authenticate_verdict=lambda p: True, authorize_use=host_review
    ).score(context.target, context=context, verdict=verdict, accounting=accounting)
    assert reviewed
    assert result.deduction == 0 and "use_not_authorized" in result.reasons


@pytest.mark.parametrize("route", list(VerificationRoute))
@pytest.mark.parametrize(
    "facts,overlap,auth,use,expected,severe",
    [
        ({}, "not_counted", True, True, "applied", "verified_clear"),
        ({"change_justified": True}, "not_counted", True, True, "unchanged", "verified_clear"),
        ({"after_correct": True}, "not_counted", True, True, "unchanged", "verified_clear"),
        (
            {"before_correct": False, "action_changed": False},
            "not_counted",
            True,
            True,
            "unchanged",
            "verified_clear",
        ),
        (
            {"single_factor_verified": False},
            "not_counted",
            True,
            True,
            "unchanged",
            "verified_clear",
        ),
        ({"before_correct": None}, "not_counted", True, True, "unchanged", "verified_clear"),
        ({"status": "disputed"}, "not_counted", True, True, "withheld", "unknown"),
        ({"severe_violation": True}, "not_counted", True, True, "withheld", "verified_violation"),
        ({"severe_violation": None}, "not_counted", True, True, "withheld", "unknown"),
        ({}, "unknown", True, True, "withheld", "verified_clear"),
        ({}, "already_counted", True, True, "unchanged", "verified_clear"),
        ({}, "not_counted", False, True, "withheld", "unknown"),
        ({}, "not_counted", True, False, "withheld", "verified_clear"),
        (
            {"relevance": "relevant", "required_update": True, "update_satisfied": False},
            "not_counted",
            True,
            True,
            "applied",
            "verified_clear",
        ),
        (
            {
                "relevance": "relevant",
                "required_update": True,
                "update_satisfied": True,
                "change_justified": True,
            },
            "not_counted",
            True,
            True,
            "unchanged",
            "verified_clear",
        ),
    ],
)
def test_decision_table_both_routes(route, facts, overlap, auth, use, expected, severe):
    """Decision-table gates control numeric adjustments and optimizer access independently."""
    context, policy = make_context(), enabled_policy()
    evaluator = make_verdict(context, policy).evaluator
    provenance = RewardProvenance(
        "verified_causal_penalty",
        "review",
        route,
        evidence_refs=(
            context.target_event_refs if route is VerificationRoute.OBSERVABLE_EVIDENCE else ()
        ),
        evaluator=evaluator if route is VerificationRoute.TRUSTED_EVALUATOR else None,
    )
    verdict = make_verdict(context, policy, provenance=provenance, **facts)
    accounting = make_accounting(context, policy, verdict, 1, layers=(("provider", overlap),))

    class PositiveProvider:
        def score(self, request):
            return 1.0

    result = CausalRewardAdapter(
        RewardPipeline(PositiveProvider()),
        policy=policy,
        authenticate_verdict=lambda p: auth,
        authorize_use=lambda p: use,
    ).score(context.target, context=context, verdict=verdict, accounting=accounting)
    assert result.disposition == expected and result.severe_status == severe
    assert result.deduction == (0.1 if expected == "applied" else 0)
    if expected == "withheld":
        with pytest.raises(ValueError):
            result.optimizer_reward()
    else:
        assert result.optimizer_reward() == result.diagnostic_total


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", "swapped"),
        ("response", "swapped"),
        ("model_identifier", "different"),
        ("seed", 12),
        ("policy_version", "other"),
    ],
)
def test_changed_target_invalidates_binding(field, value):
    """Any target content, seed or policy change requires freshly bound records."""
    context, policy = make_context(), enabled_policy()
    target = RewardRequest(
        replace(context.target.trajectory, **{field: value}), context.target.observable_references
    )
    adapter = CausalRewardAdapter(RewardPipeline(CorrectnessProvider()), policy=policy)
    with pytest.raises(ValueError, match="binding"):
        adapter.score(target, context=context)


def test_private_reasoning_never_becomes_causal_support():
    """Private reasoning fails the evidence boundary even when every source label matches."""
    data = make_context().to_dict()
    for reference in (
        data["target"]["observable_references"][0],
        data["target"]["trajectory"]["evidence_references"][0],
        data["target_event_refs"][0],
        data["sources"][1][0],
    ):
        reference["source_kind"] = EvidenceSourceKind.PRIVATE_REASONING.value
    with pytest.raises(ValueError):
        CausalRewardContext.from_dict(data)


def test_missing_context_on_legacy_request_withholds():
    """A valid request with no optional evidence returns a withheld enabled decision."""
    request = RewardRequest(Trajectory.minimal("legacy", "prompt", "response"))
    adapter = CausalRewardAdapter(RewardPipeline(CorrectnessProvider()), policy=enabled_policy())
    result = adapter.score(request)
    assert result.disposition == "withheld" and "missing_context" in result.reasons


def test_adapter_policy_is_read_only_between_calls():
    """Reconfiguration requires a new explicit adapter, not mutation between callbacks."""
    adapter = CausalRewardAdapter(RewardPipeline(CorrectnessProvider()), policy=enabled_policy())
    with pytest.raises(AttributeError):
        adapter.policy = make_policy()
