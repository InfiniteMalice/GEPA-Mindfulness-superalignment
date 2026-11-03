"""Live credit admission, evidence, replay and canonical reward boundary tests."""

from __future__ import annotations

import importlib
import json
import pickle
from dataclasses import replace

import pytest
from test_research_credit_ledger import proposal, request, use

from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger
from gepa_mindfulness.training.reward_pipeline import RewardPipeline


class Baseline:
    """Real scalar provider whose calls expose admission ordering and input mutation."""

    def __init__(self, value=0.25):
        self.value = value
        self.calls = 0

    def score(self, target):
        self.calls += 1
        return self.value


def api():
    return importlib.import_module("gepa_mindfulness.training.research_credit_runtime")


def policy(**changes):
    values = dict(
        policy_id="host-policy-v1",
        run_id="run",
        task_id="task",
        harness_version="harness-v1",
        dataset_digest="a" * 64,
        coefficients=(("knowledge", 0.5), ("progress", 0.5), ("process", 0.5)),
        max_abs_adjustment=1.0,
        enabled=True,
    )
    values.update(changes)
    return api().ResearchRewardPolicy(**values)


def submission(target=None, *, claims=None, selection=None):
    target = target or request()
    claims = claims or (proposal(target, action_id=target.trajectory.trajectory_id),)
    return api().ResearchRewardSubmission(claims, selection or use(target))


def provider(tmp_path, *, target=None, baseline=None, pipeline=None, **changes):
    target = target or request()
    baseline = baseline or Baseline()
    values = dict(
        ledger=ResearchCreditLedger(tmp_path / "credits.sqlite", authority_domain="host"),
        policy=policy(),
        propose=lambda actual: submission(actual),
        authenticate_claims=lambda envelope: True,
        authorize_optimizer_use=lambda envelope: True,
        authorize_accounting=lambda envelope: True,
    )
    values.update(changes)
    return (
        api().ResearchRewardProvider(pipeline or RewardPipeline(baseline), **values),
        baseline,
        target,
    )


def test_disabled_preserves_pipeline_and_does_not_inspect_inputs(tmp_path):
    runtime, baseline, _ = provider(
        tmp_path, policy=policy(enabled=False), ledger=object(), propose=object()
    )
    result = runtime.score(object())
    assert result.total == 0.25
    assert baseline.calls == 1
    accepted, _, target = provider(tmp_path)
    assert accepted.score(target).optimizer_reward() == 0.5


def test_live_decision_is_reviewed_and_exports_no_optimizer_authority(tmp_path):
    runtime, baseline, target = provider(tmp_path)
    result = runtime.score(target)
    assert result.optimizer_reward() == 0.5
    assert baseline.calls == 1
    wire = result.to_dict()
    assert wire["schema_version"] == "research-reward-decision-v1"
    assert wire["status"] == "authorized"
    assert wire["training_eligibility"] == "DEVELOPMENT"
    assert wire["authority_granted"] is False
    assert wire["baseline_total"] == 0.25
    assert wire["adjustment"] == 0.25
    assert wire["components"] == [
        {"claim_id": "claim-1", "kind": "knowledge", "raw": 0.5, "weight": 0.5, "weighted": 0.25}
    ]
    assert "total" not in wire
    with pytest.raises(ValueError):
        require_training_eligible(wire)
    with pytest.raises(TypeError):
        api().ResearchRewardDecision()
    with pytest.raises(TypeError):
        pickle.dumps(result)
    wire["diagnostic_total"] = 900
    assert result.optimizer_reward() == 0.5
    metrics = result.metrics()
    metrics["research_adjustment"] = 900
    assert result.metrics()["research_adjustment"] == 0.25
    assert json.loads(json.dumps(result.to_dict()))["diagnostic_total"] == 0.5


@pytest.mark.parametrize("stage", ["authenticate_claims", "authorize_optimizer_use"])
@pytest.mark.parametrize("outcome", [None, False, 1, "yes", "exception", "mutation"])
def test_unavailable_or_untrusted_review_withholds_credit_without_reservation(
    tmp_path, stage, outcome
):
    def review(envelope):
        if outcome == "exception":
            raise RuntimeError("private credential material")
        if outcome == "mutation":
            envelope["policy"]["coefficients"] = []
            return True
        return outcome

    callback = None if outcome is None else review
    runtime, _, target = provider(tmp_path, **{stage: callback})
    result = runtime.score(target)
    assert result.to_dict()["status"] == "withheld"
    with pytest.raises(ValueError) as error:
        result.optimizer_reward()
    assert "private credential material" not in str(error.value)
    # A denied live review must leave the durable event available to an authorized call.
    accepted, _, _ = provider(tmp_path)
    assert accepted.score(target).optimizer_reward() == 0.5


@pytest.mark.parametrize("outcome", [None, False, 1, "exception", "mutation"])
def test_existing_accounting_review_remains_mandatory(tmp_path, outcome):
    def review(envelope):
        if outcome == "exception":
            raise RuntimeError("private accounting details")
        if outcome == "mutation":
            envelope["use"]["policy_id"] = "replaced"
            return True
        return outcome

    runtime, _, target = provider(
        tmp_path, authorize_accounting=None if outcome is None else review
    )
    with pytest.raises(ValueError) as error:
        runtime.score(target)
    assert "private accounting details" not in str(error.value)
    accepted, _, _ = provider(tmp_path)
    assert accepted.score(target).optimizer_reward() == 0.5


@pytest.mark.parametrize(
    "changes",
    [
        {"admissibility": "unresolved"},
        {"admissibility": "rejected"},
        {"provenance": None},
        {"credit_kind": "teacher_agreement"},
        {"credit_kind": "diagnostic"},
        {"source_metadata_json": '{"training_eligibility":"HIDDEN_EVAL"}'},
        {"source_metadata_json": '{"training_eligibility":"TRAIN","x":{"holdout_status":"held"}}'},
        {"action_id": "other-action"},
        {"model_version": "other-model"},
        {"harness_version": "other-harness"},
        {"run_id": "other-run"},
        {"task_id": "other-task"},
        {"dataset_digest": "f" * 64},
    ],
)
def test_ineligible_or_foreign_proposals_fail_before_baseline_or_host_review(tmp_path, changes):
    target = request()
    invalid = replace(proposal(target, action_id=target.trajectory.trajectory_id), **changes)
    baseline = Baseline()
    runtime, _, _ = provider(
        tmp_path, baseline=baseline, propose=lambda actual: submission(actual, claims=(invalid,))
    )
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0


def test_nested_heldout_target_is_rejected_before_proposal(tmp_path):
    attempted = []
    target = request()
    target = replace(
        target,
        trajectory=replace(
            target.trajectory,
            sampling_parameters={
                "training_eligibility": "TRAIN",
                "source_record": {"training_eligibility": "REGRESSION"},
            },
        ),
    )

    def unavailable(actual):
        attempted.append(True)
        raise AssertionError("held-out target reached the proposer")

    runtime, baseline, _ = provider(tmp_path, propose=unavailable)
    with pytest.raises((ValueError, TypeError)):
        runtime.score(target)
    assert baseline.calls == 0
    assert attempted == []


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": 1},
        {"coefficients": (("knowledge", True),)},
        {"coefficients": (("knowledge", float("nan")),)},
        {"coefficients": (("knowledge", -1.0),)},
        {"coefficients": (("knowledge", 0.0),)},
        {"coefficients": (("knowledge", 0.5), ("knowledge", 0.5))},
        {"coefficients": (("teacher_agreement", 1.0),)},
        {"coefficients": (("constitutional_values", 0.00001),)},
        {"coefficients": ()},
        {"max_abs_adjustment": float("inf")},
        {"max_abs_adjustment": 0},
        {"dataset_digest": "malformed"},
        {"policy_id": ""},
    ],
)
def test_policy_rejects_invalid_scalarization_or_identity(changes):
    with pytest.raises((ValueError, TypeError)):
        policy(**changes)


def test_policy_and_submission_roundtrip_reject_unknown_versions_and_fields():
    original = policy()
    restored = api().ResearchRewardPolicy.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    for change in ({"schema_version": "other"}, {"extra": True}):
        wire = original.to_dict() | change
        with pytest.raises(ValueError):
            api().ResearchRewardPolicy.from_dict(wire)
    item = submission()
    assert api().ResearchRewardSubmission.from_dict(item.to_dict()) == item


def test_proposer_and_baseline_cannot_mutate_reviewed_target(tmp_path):
    target = request()
    original = target.trajectory.to_dict()

    def mutate(actual):
        value = submission(actual)
        object.__setattr__(actual.trajectory, "response", "replaced")
        return value

    runtime, baseline, _ = provider(tmp_path, propose=mutate)
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0
    assert target.trajectory.to_dict() == original

    class MutatingBaseline(Baseline):
        def score(self, actual):
            object.__setattr__(actual.trajectory, "response", "changed by provider")
            return 0.25

    runtime, _, _ = provider(tmp_path, baseline=MutatingBaseline())
    with pytest.raises(ValueError):
        runtime.score(target)
    assert target.trajectory.to_dict() == original


def test_producer_cannot_be_the_independent_evaluator(tmp_path):
    target = replace(request(), trajectory=replace(request().trajectory, model_identifier="model"))
    invalid = proposal(
        target,
        action_id=target.trajectory.trajectory_id,
        evaluator=TrustedEvaluatorContract("model", "v1", "review"),
    )
    runtime, baseline, _ = provider(
        tmp_path, propose=lambda actual: submission(actual, claims=(invalid,))
    )
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0


def test_durable_event_reservation_blocks_replay_after_restart(tmp_path):
    runtime, _, target = provider(tmp_path)
    assert runtime.score(target).optimizer_reward() == 0.5
    reopened, _, _ = provider(
        tmp_path,
        propose=lambda actual: submission(actual, selection=use(actual, use_id="other-use")),
    )
    with pytest.raises(ValueError, match="semantic event already reserved"):
        reopened.score(target)


def test_same_verified_event_cannot_fund_knowledge_and_progress(tmp_path):
    target = request()
    first = proposal(target, action_id=target.trajectory.trajectory_id)
    second = replace(first, claim_id="claim-2", credit_kind="progress")
    original = use(target)
    selection = replace(
        original,
        selected_claim_ids=("claim-1", "claim-2"),
        layers=(original.layers[0], ("claim-2", original.layers[0][1])),
    )
    runtime, baseline, _ = provider(
        tmp_path,
        propose=lambda actual: submission(actual, claims=(first, second), selection=selection),
    )
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0


def test_foreign_policy_and_budget_overflow_fail_before_callbacks(tmp_path):
    target = request()
    for item in (
        submission(target, selection=use(target, policy_id="other-policy")),
        submission(target, claims=(proposal(target, action_id="trajectory", score=3.0),)),
    ):
        runtime, baseline, _ = provider(tmp_path, propose=lambda actual: item)
        with pytest.raises(ValueError):
            runtime.score(target)
        assert baseline.calls == 0


def test_signed_progress_binds_observable_evidence_to_canonical_trajectory(tmp_path):
    from gepa_mindfulness.training.engine import _bind_reward, _reward_value

    target = request()
    item = submission(
        target,
        claims=(proposal(target, action_id="trajectory", credit_kind="progress", score=-0.5),),
    )
    runtime, _, _ = provider(tmp_path, baseline=Baseline(0.0), propose=lambda actual: item)
    result = runtime.score(target)
    assert _reward_value(result) == -0.25
    bound = _bind_reward(target.trajectory, result)
    assert bound.reward_total == -0.25
    assert bound.reward_components["research_credit:claim-1"] == -0.25
    assert (
        bound.reward_component_evidence["research_credit:claim-1"] == target.observable_references
    )
    assert bound.reward_component_provenance["research_credit:claim-1"].evaluator.evaluator_id == (
        "independent"
    )
    assert bound.to_dict()["sampling_parameters"] == {"training_eligibility": "TRAIN"}
    foreign = replace(target.trajectory, response="substituted response")
    with pytest.raises(ValueError, match="target"):
        _bind_reward(foreign, result)


def test_unverified_negative_baseline_keeps_existing_engine_rejection(tmp_path):
    runtime, _, target = provider(tmp_path, baseline=Baseline(-0.25))
    with pytest.raises(ValueError, match="negative reward"):
        runtime.score(target)


@pytest.mark.parametrize("disposition", ["unknown", "already_counted", "omitted"])
def test_invalid_prior_accounting_fails_before_baseline_or_review(tmp_path, disposition):
    target = request()
    layers = (
        ("provider", disposition),
        ("reward_integrity", "not_counted"),
        ("causal_reward", "not_counted"),
    )
    if disposition == "omitted":
        layers = layers[1:]
    selection = use(target, layers=(("claim-1", layers),))
    runtime, baseline, _ = provider(
        tmp_path, propose=lambda actual: submission(actual, selection=selection)
    )
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0


def test_complete_report_budget_is_reserved_before_any_host_review(tmp_path):
    target = replace(
        request(), trajectory=replace(request().trajectory, prompt="x" * (2 * 1024 * 1024))
    )
    reviewed = []

    def review(envelope):
        reviewed.append(True)
        return True

    runtime, _, _ = provider(tmp_path, authenticate_claims=review)
    with pytest.raises(ValueError, match="4 MiB"):
        runtime.score(target)
    assert reviewed == []


def test_exported_or_duck_typed_diagnostic_cannot_enter_optimizer():
    from gepa_mindfulness.training.engine import _reward_value

    class FakeDecision:
        diagnostic_total = 100.0

        def optimizer_reward(self):
            return 100.0

    for value in (FakeDecision(), {"diagnostic_total": 100.0}):
        with pytest.raises(ValueError):
            _reward_value(value)


def test_unselected_heldout_material_is_not_exposed_to_training_review(tmp_path):
    target = request()
    selected = proposal(target, action_id="trajectory")
    hidden = replace(
        selected,
        claim_id="heldout",
        event_id="heldout-event",
        source_metadata_json='{"training_eligibility":"HIDDEN_EVAL"}',
    )
    runtime, baseline, _ = provider(
        tmp_path, propose=lambda actual: submission(actual, claims=(selected, hidden))
    )
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0


def test_proposer_failure_does_not_expose_private_transport_details(tmp_path):
    def unavailable(actual):
        raise RuntimeError("private transport credential")

    runtime, baseline, target = provider(tmp_path, propose=unavailable)
    with pytest.raises(ValueError) as error:
        runtime.score(target)
    assert "private transport credential" not in str(error.value)
    assert baseline.calls == 0


def test_overflowing_weighted_sum_is_rejected_before_scoring(tmp_path):
    target = request()
    first = proposal(target, action_id="trajectory", score=1e308)
    second = replace(first, claim_id="claim-2", event_id="event-2")
    original = use(target)
    selection = replace(
        original,
        selected_claim_ids=("claim-1", "claim-2"),
        layers=(original.layers[0], ("claim-2", original.layers[0][1])),
    )
    runtime, baseline, _ = provider(
        tmp_path,
        policy=policy(coefficients=(("knowledge", 1.0),)),
        propose=lambda actual: submission(actual, claims=(first, second), selection=selection),
    )
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0


def test_live_composition_preserves_existing_integrity_overlay_and_evidence(tmp_path):
    from gepa_mindfulness.core.reward_integrity import COMPONENT_NAMES, RewardIntegrityCalculator
    from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute
    from gepa_mindfulness.training.engine import _bind_reward

    original = request()
    ref = original.observable_references[0]
    target = replace(
        original,
        trajectory=replace(
            original.trajectory,
            reward_components={
                name: -1.0 if name == "objective_fidelity" else 0.0 for name in COMPONENT_NAMES
            },
            reward_component_evidence={"objective_fidelity": (ref,)},
            reward_component_provenance={
                "objective_fidelity": RewardProvenance(
                    "objective_fidelity",
                    "external observed failure",
                    VerificationRoute.OBSERVABLE_EVIDENCE,
                    (ref,),
                )
            },
        ),
    )
    baseline = Baseline(0.5)
    pipeline = RewardPipeline(
        baseline, integrity_calculator=RewardIntegrityCalculator(), overlay_weight=0.8
    )
    runtime, _, _ = provider(tmp_path, baseline=baseline, pipeline=pipeline)
    result = runtime.score(target)
    assert result.optimizer_reward() == 0.65
    assert result.to_dict()["baseline_total"] == 0.4
    bound = _bind_reward(target.trajectory, result)
    assert bound.reward_components["reward_integrity_overlay:objective_fidelity"] == -0.1
    assert bound.reward_component_evidence["reward_integrity_overlay:objective_fidelity"] == (ref,)
    assert baseline.calls == 1


@pytest.mark.parametrize(
    "kind,score,base,want",
    [
        ("knowledge", 0.5, 0.25, 0.5),
        ("progress", -0.5, 0.0, -0.25),
    ],
)
def test_structured_baseline_retains_original_contribution_references(
    tmp_path, kind, score, base, want
):
    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
    from gepa_mindfulness.training.engine import RewardAssessment, _bind_reward

    baseline_ref = EvidenceReference("baseline-only", EvidenceSourceKind.EXTERNAL_RECORD)

    class StructuredBaseline(Baseline):
        def score(self, target):
            return RewardAssessment(base, {"task_success": base}, {}, (baseline_ref,))

    target = request()
    claim = proposal(target, action_id="trajectory", credit_kind=kind, score=score)
    runtime, _, _ = provider(
        tmp_path,
        baseline=StructuredBaseline(),
        propose=lambda actual: submission(actual, claims=(claim,)),
    )
    result = runtime.score(target)
    assert result.optimizer_reward() == want
    bound = _bind_reward(target.trajectory, result)
    assert set(bound.evidence_references) == {baseline_ref, *target.observable_references}
    assert (
        bound.reward_component_evidence["research_credit:claim-1"] == target.observable_references
    )


def test_structured_baseline_cannot_replace_a_source_reference_kind(tmp_path):
    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
    from gepa_mindfulness.training.engine import RewardAssessment

    original = request()
    source = EvidenceReference("additional-source", EvidenceSourceKind.EXTERNAL_RECORD)
    target = replace(
        original,
        trajectory=replace(
            original.trajectory, evidence_references=(*original.observable_references, source)
        ),
        observable_references=(*original.observable_references, source),
    )
    conflict = EvidenceReference("additional-source", EvidenceSourceKind.OBSERVABLE_OUTPUT)

    class ConflictingBaseline(Baseline):
        def score(self, actual):
            return RewardAssessment(
                0.25, {"task_success": 0.25}, {}, (*original.observable_references, conflict)
            )

    runtime, _, _ = provider(tmp_path, baseline=ConflictingBaseline())
    with pytest.raises(ValueError, match="conflicting"):
        runtime.score(target)


@pytest.mark.parametrize("selection", [(0,), (2, 0)])
def test_live_binding_preserves_the_original_observable_reference_selection(tmp_path, selection):
    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

    original = request()
    refs = (
        *original.trajectory.evidence_references,
        EvidenceReference("not-selected", EvidenceSourceKind.EXTERNAL_RECORD),
        EvidenceReference("second-selection", EvidenceSourceKind.EXTERNAL_RECORD),
    )
    target = replace(
        original,
        trajectory=replace(original.trajectory, evidence_references=refs),
        observable_references=tuple(refs[index] for index in selection),
    )
    runtime, _, _ = provider(tmp_path)
    result = runtime.score(target)
    bound = result.bind_trajectory(target.trajectory)
    assert bound.reward_total == 0.5
    assert bound.reward_component_evidence["research_credit:claim-1"] == (refs[0],)
    assert result.to_dict()["review"]["target_request"]["observable_references"] == [
        refs[index].to_dict() for index in selection
    ]


def test_unselected_observable_reference_stays_ineligible_for_credit(tmp_path):
    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
    from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute

    original = request()
    unauthorized = EvidenceReference("not-authorized", EvidenceSourceKind.EXTERNAL_RECORD)
    target = replace(
        original,
        trajectory=replace(
            original.trajectory,
            evidence_references=(*original.trajectory.evidence_references, unauthorized),
        ),
    )
    claim = proposal(
        target,
        action_id="trajectory",
        evidence_refs=(unauthorized,),
        provenance=RewardProvenance(
            "knowledge", "external", VerificationRoute.OBSERVABLE_EVIDENCE, (unauthorized,)
        ),
    )
    runtime, baseline, _ = provider(
        tmp_path, propose=lambda actual: submission(actual, claims=(claim,))
    )
    with pytest.raises(ValueError):
        runtime.score(target)
    assert baseline.calls == 0
