"""Independent replay, exact authority and diagnostic-only mechanistic registration."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest
from test_native_causal_interventions import Clock, trial

# Local
from evaluation.causal_records import canonical_json
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.interpret.causal_interventions import execute_causal_plan, freeze_causal_plan
from gepa_mindfulness.interpret.mechanistic_evidence import (
    EvidenceLevel,
    MechanisticRegistry,
    MechanisticVerifierContext,
    admit_mechanistic_finding,
    mechanistic_update_scope,
    verify_causal_runs,
)
from gepa_mindfulness.interpret.native_neurons import FrozenNativeModel
from gepa_mindfulness.interpret.native_records import NativeInput
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.verification.runtime_governance import (
    ActionAuthorityPolicy,
    AuthorityGrant,
    AuthorityGrantRegistry,
    RuntimeCapability,
    RuntimeRole,
    action_record_digest,
    authorize_action,
)
from mindful_trace_gepa.action_bound_events import ActionRecord, PredictionCommit

torch = pytest.importorskip("torch")


def known_cause():
    source, _, plan, _ = trial()
    with torch.no_grad():
        for p in source.parameters():
            p.zero_()
        for block in source.transformer.h:
            block.ln_1.weight.fill_(1.0)
            block.ln_2.weight.fill_(1.0)
        source.transformer.ln_f.weight.fill_(1.0)
        for token, axis in ((1, 0), (3, 2), (4, 1), (5, 0), (6, 1)):
            source.transformer.wte.weight[token, axis] = 1.0
        source.transformer.wte.weight[5, 3] = 0.1
        source.transformer.wte.weight[6, 3] = 0.2
        source.transformer.h[0].mlp.c_fc.weight[0, 0] = 1.0
        source.transformer.h[0].mlp.c_proj.weight[0, 2] = 1.0
    native = FrozenNativeModel(
        source, model_id="engineered-positive-control", revision="construction-v1", enabled=True
    )
    from gepa_mindfulness.interpret.native_causal_records import EffectPrediction

    plan = replace(
        plan,
        model_digest=native.identity.digest,
        metric=replace(plan.metric, position=0),
        prediction=EffectPrediction(1, 0.1, 20.0, 0.1),
    )
    return native, plan


def measure(native, plan):
    action_id = "execute:" + plan.experiment_id
    pred = PredictionCommit(
        "predict:" + plan.experiment_id,
        {"causal_plan_digest": plan.digest},
        0.5,
        tuple(r.reference_id for r in plan.evidence_refs),
    )
    action = ActionRecord(
        action_id,
        "mechanistic_intervention",
        True,
        "mechanistic-plan:" + plan.digest,
        pred.prediction_commit_id,
    )
    commit = freeze_causal_plan(plan, prediction=pred, action=action, clock=Clock(), enabled=True)
    return execute_causal_plan(native, commit, clock=Clock(), enabled=True)


def context(native, authenticate=lambda packet: True):
    return MechanisticVerifierContext(
        native,
        TrustedEvaluatorContract("independent-checker", "1", "mechanistic-host-v1"),
        "discovery-researcher",
        Clock(),
        authenticate,
    )


def test_known_finite_cause_is_replayed_without_becoming_training_authority() -> None:
    native, plan = known_cause()
    report = measure(native, plan)
    assert report["prediction_result"] == "MET"
    finding = verify_causal_runs((report,), context=context(native), enabled=True)
    assert finding.level is EvidenceLevel.INTERVENTION_SUPPORTED
    assert finding.source_reports_json == (canonical_json(report),)
    assert type(finding).from_dict(finding.to_dict()) == finding
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(finding.to_dict())


def test_replication_requires_distinct_causal_contexts_not_renamed_duplicates() -> None:
    native, plan = known_cause()
    report = measure(native, plan)
    renamed = replace(
        plan,
        experiment_id="renamed",
        donor=replace(plan.donor, example_id="new-donor"),
        recipient=replace(plan.recipient, example_id="new-recipient"),
    )
    with pytest.raises(ValueError, match="distinct"):
        verify_causal_runs(
            (report, measure(native, renamed)), context=context(native), enabled=True
        )
    second = replace(
        plan,
        experiment_id="replicate",
        donor=NativeInput("d2", (5, 2, 3), (1, 1, 1)),
        recipient=NativeInput("r2", (6, 2, 3), (1, 1, 1)),
    )
    finding = verify_causal_runs(
        (report, measure(native, second)), context=context(native), enabled=True
    )
    assert finding.level is EvidenceLevel.REPLICATED_CAUSAL_EVIDENCE
    assert finding.replication_count == 2


@pytest.mark.parametrize("authenticate", [None, lambda packet: False, lambda packet: 1])
def test_missing_or_truthy_authentication_does_not_verify_evidence(authenticate) -> None:
    native, plan = known_cause()
    finding = verify_causal_runs(
        (measure(native, plan),), context=context(native, authenticate), enabled=True
    )
    assert finding.level is EvidenceLevel.UNRESOLVED


def test_mutating_or_throwing_host_callback_fails_closed() -> None:
    native, plan = known_cause()
    report = measure(native, plan)
    for mutate in (False, True):

        def authenticate(packet):
            if mutate:
                object.__setattr__(packet, "actor_id", "rewritten")
            else:
                raise RuntimeError("host unavailable")
            return True

        result = verify_causal_runs((report,), context=context(native, authenticate), enabled=True)
        assert result.level is EvidenceLevel.UNRESOLVED


def test_self_review_and_coherent_forgery_cannot_supply_causal_verification() -> None:
    native, plan = known_cause()
    with pytest.raises(ValueError, match="self-review"):
        replace(context(native), actor_id="independent-checker")
    report = measure(native, plan)
    forged = json.loads(canonical_json(report))
    forged["conditions"]["intervention"]["logits"][0][3] += 1.0
    forged["conditions"]["intervention"]["score"] += 1.0
    forged["conditions"]["intervention"]["effect"] += 1.0
    forged["primary_effect"] += 1.0
    from gepa_mindfulness.interpret.native_causal_records import run_measurement_digest

    forged["measurement_digest"] = run_measurement_digest(forged)
    forged["events"][-1]["payload"]["actual_outcome"]["measurement_digest"] = forged[
        "measurement_digest"
    ]
    result = verify_causal_runs((forged,), context=context(native), enabled=True)
    assert result.level is EvidenceLevel.UNRESOLVED


def write_authority(before, finding):
    scope = mechanistic_update_scope(before, finding)
    action = ActionRecord(
        "register-1", "mechanistic_evidence_registration", True, scope, "pred-register"
    )
    digest = action_record_digest(action)
    policy = ActionAuthorityPolicy(
        "policy", action.action_id, digest, scope, RuntimeCapability.WRITE
    )
    grant = AuthorityGrant(
        "grant",
        "host",
        RuntimeRole.EXECUTOR,
        (RuntimeCapability.WRITE,),
        action.action_id,
        digest,
        scope,
    )
    registry = AuthorityGrantRegistry.enroll((grant,), (policy,))
    authorization = authorize_action(
        action,
        principal_id="host",
        role=RuntimeRole.EXECUTOR,
        capability=RuntimeCapability.WRITE,
        policy_id="policy",
        grant_registry=registry,
        grant_ids=("grant",),
        clock=Clock(),
    )
    return dict(action=action, authorization=authorization, grant_registry=registry, clock=Clock())


def test_registry_admission_consumes_exact_write_and_preserves_diagnostics() -> None:
    native, plan = known_cause()
    finding = verify_causal_runs((measure(native, plan),), context=context(native), enabled=True)
    before = MechanisticRegistry("registry", 0, ())
    args = write_authority(before, finding)
    after = admit_mechanistic_finding(
        before, finding, context=context(native), enabled=True, **args
    )
    assert len(after.entries) == 1
    assert after.revision == 1 and before.entries == ()
    with pytest.raises(PermissionError, match="consumed"):
        admit_mechanistic_finding(before, finding, context=context(native), enabled=True, **args)
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(after.to_dict())


def test_admission_rechecks_sources_and_rejects_missing_authority() -> None:
    native, plan = known_cause()
    finding = verify_causal_runs((measure(native, plan),), context=context(native), enabled=True)
    before = MechanisticRegistry("registry", 0, ())
    args = write_authority(before, finding)
    with pytest.raises(PermissionError):
        admit_mechanistic_finding(
            before, finding, context=context(native, None), enabled=True, **args
        )
    with pytest.raises(PermissionError):
        admit_mechanistic_finding(
            before,
            finding,
            context=context(native),
            enabled=True,
            **{**args, "authorization": None},
        )
    with pytest.raises(ValueError, match="enabled=True"):
        verify_causal_runs(None)


@pytest.mark.parametrize("change", ["extra", "missing", "event", "metric"])
def test_typed_report_and_trace_cannot_be_relabelled_before_verification(change) -> None:
    native, plan = known_cause()
    report = measure(native, plan)
    if change == "extra":
        report["reviewer_accepted"] = True
    elif change == "missing":
        del report["conditions"]
    elif change == "event":
        report["events"][0]["payload"]["predicted_outcome"]["causal_plan_digest"] = "0" * 64
    else:
        report["conditions"]["intervention"]["score"] += 1.0
        from gepa_mindfulness.interpret.native_causal_records import run_measurement_digest

        report["measurement_digest"] = run_measurement_digest(report)
    with pytest.raises(ValueError):
        verify_causal_runs((report,), context=context(native), enabled=True)


def test_falsified_predictions_are_retained_and_not_erased_by_an_acceptance_flag() -> None:
    native, plan = known_cause()
    failed = replace(plan, prediction=replace(plan.prediction, direction=-1))
    report = measure(native, failed)
    finding = verify_causal_runs((report,), context=context(native), enabled=True)
    assert finding.level is EvidenceLevel.CONTRADICTED
    assert finding.source_reports_json == (canonical_json(report),)
    assert any("falsified" in reason for reason in finding.remaining_uncertainty)
    assert not any("are supported" in reason for reason in finding.remaining_uncertainty)


def test_future_only_context_changes_do_not_count_as_replication() -> None:
    native, plan = known_cause()
    future_changed = replace(
        plan,
        experiment_id="future-only",
        donor=NativeInput("d2", (1, 3, 2), (1, 1, 1)),
        recipient=NativeInput("r2", (4, 3, 2), (1, 1, 1)),
    )
    with pytest.raises(ValueError, match="distinct"):
        verify_causal_runs(
            (measure(native, plan), measure(native, future_changed)),
            context=context(native),
            enabled=True,
        )


@pytest.mark.parametrize("irrelevant", ["future_site", "masked_prefix"])
def test_irrelevant_intervention_context_cannot_manufacture_replication(irrelevant) -> None:
    native, plan = known_cause()
    if irrelevant == "future_site":
        plan = replace(plan, source=replace(plan.source, token_positions=(0, 2)))
        donor, recipient = (1, 2, 5), (4, 2, 6)
    else:
        plan = replace(
            plan,
            source=replace(plan.source, token_positions=(2,)),
            metric=replace(plan.metric, position=2),
            donor=NativeInput("d", (1, 2, 1), (1, 0, 1)),
            recipient=NativeInput("r", (4, 2, 4), (1, 0, 1)),
        )
        donor, recipient = (1, 3, 1), (4, 3, 4)
    other = replace(
        plan,
        experiment_id="irrelevant-change",
        donor=NativeInput("other-d", donor, plan.donor.attention_mask),
        recipient=NativeInput("other-r", recipient, plan.recipient.attention_mask),
    )
    first, second = measure(native, plan), measure(native, other)
    assert first["primary_effect"] == second["primary_effect"]
    with pytest.raises(ValueError, match="distinct"):
        verify_causal_runs((first, second), context=context(native), enabled=True)


def test_serialized_level_and_stale_write_scope_cannot_grant_admission() -> None:
    native, plan = known_cause()
    finding = verify_causal_runs((measure(native, plan),), context=context(native), enabled=True)
    before = MechanisticRegistry("registry", 0, ())
    forged = replace(finding, level=EvidenceLevel.REPLICATED_CAUSAL_EVIDENCE)
    args = write_authority(before, forged)
    with pytest.raises(PermissionError):
        admit_mechanistic_finding(before, forged, context=context(native), enabled=True, **args)
    args = write_authority(before, finding)
    after = admit_mechanistic_finding(
        before, finding, context=context(native), enabled=True, **args
    )
    second_plan = replace(
        plan,
        experiment_id="new-evidence",
        donor=NativeInput("d2", (5, 2, 3), (1, 1, 1)),
        recipient=NativeInput("r2", (6, 2, 3), (1, 1, 1)),
    )
    second = verify_causal_runs(
        (measure(native, second_plan),), context=context(native), enabled=True
    )
    stale = write_authority(before, second)
    with pytest.raises(PermissionError):
        admit_mechanistic_finding(after, second, context=context(native), enabled=True, **stale)


def prediction_fixture():
    from gepa_mindfulness.interpret.explanation_tests import (
        ExampleRef,
        ExplanationHypothesis,
        ExplanationKind,
        SplitManifest,
        TokenFeature,
        evaluate_explanation,
        fit_ridge_predictor,
        freeze_explanation,
    )
    from gepa_mindfulness.interpret.native_records import HookKind

    native, _ = known_cause()
    inputs = {
        "generation": (
            NativeInput("g1", (1, 2, 3), (1, 1, 1)),
            NativeInput("g2", (4, 1, 2), (1, 1, 1)),
            NativeInput("g3", (3, 4, 1), (1, 1, 1)),
        ),
        "development": (NativeInput("d1", (4, 3, 2), (1, 1, 1)),),
        "locked": (
            NativeInput("l1", (1, 3, 4), (1, 1, 1)),
            NativeInput("l2", (2, 1, 4), (1, 1, 1)),
        ),
    }
    groups = {
        k: tuple(ExampleRef.from_input(i, distribution="id") for i in v) for k, v in inputs.items()
    }
    splits = SplitManifest(groups["generation"], groups["development"], groups["locked"])
    hook = native.hook(layer=0, kind=HookKind.MLP_POST)

    def capture(kind):
        return native.capture(
            hook,
            torch.tensor([i.token_ids for i in inputs[kind]]),
            component_indices=(0,),
            example_ids=tuple(i.example_id for i in inputs[kind]),
            token_positions=(1, 2),
            enabled=True,
        )

    generation = capture("generation")
    predictor = fit_ridge_predictor(
        splits,
        generation,
        features=(
            TokenFeature("one", "token_is", 0, 1),
            TokenFeature("zero_embedding", "token_is", 0, 2),
        ),
        ridge=1e-6,
        enabled=True,
    )
    hypothesis = ExplanationHypothesis(
        "constructed-lookup",
        "Constructed token-to-MLP relation.",
        ExplanationKind.RELATED_CONTEXT,
        ("constant activation",),
        ("New contexts with the same current token preserve activation.",),
        predictor,
        tuple(i.example_id for i in inputs["generation"]),
    )
    commit = freeze_explanation(
        hypothesis,
        splits,
        generation,
        clock=Clock(),
        max_mae=1e-4,
        activation_threshold=0.5,
        enabled=True,
    )
    return native, evaluate_explanation(commit, capture("locked"), enabled=True)


def test_actual_heldout_prediction_admission_remains_distinct_from_causal_evidence() -> None:
    from gepa_mindfulness.interpret.mechanistic_evidence import verify_predictive_run

    native, report = prediction_fixture()
    assert report["prediction_result"] == "PREREGISTERED_CRITERION_MET"
    finding = verify_predictive_run(report, context=context(native), enabled=True)
    assert finding.level is EvidenceLevel.HELD_OUT_SUPPORTED
    assert finding.source_kind == "prediction"
    before = MechanisticRegistry("prediction-registry", 0, ())
    admitted = admit_mechanistic_finding(
        before, finding, context=context(native), enabled=True, **write_authority(before, finding)
    )
    assert admitted.entries[0].level is EvidenceLevel.HELD_OUT_SUPPORTED
    missing = verify_predictive_run(report, context=context(native, None), enabled=True)
    assert missing.level is EvidenceLevel.UNRESOLVED


def test_nonspecific_replay_keeps_explicit_unresolved_reasons() -> None:
    native, plan = known_cause()
    plan = replace(plan, prediction=replace(plan.prediction, minimum_control_gap=1000000.0))
    report = measure(native, plan)
    assert report["prediction_result"] == "NOT_SPECIFIC"
    finding = verify_causal_runs((report,), context=context(native), enabled=True)
    assert finding.level is EvidenceLevel.UNRESOLVED
    assert any("specific" in reason for reason in finding.remaining_uncertainty)
    assert not any("are supported" in reason for reason in finding.remaining_uncertainty)
