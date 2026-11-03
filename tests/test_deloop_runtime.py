"""Native preference updates on authored CPU traces, without effectiveness claims."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_deloop import REVIEWER, fixture, review
from test_locd_runtime import ContextModel
from test_model_harness_coevolution import _base, _record

from gepa_mindfulness import (
    MetricAggregation,
    MetricDirection,
    MetricPolicy,
    MetricSpec,
    ValidationBundle,
    ValidationSplit,
    append_epoch_record,
    close_evaluation_epoch,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.backends import TorchPolicyBackend
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.deloop_runtime import DeLoopTrainingConfig, run_loop_preferences
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.subspace_tuning import SSTConfig
from gepa_mindfulness.training.trajectory import RolloutRequest


class ActionTokenizer:
    """A fixed local tokenizer with different original/recovery histories."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, steps):
        self.words = {
            2: steps[1].request.trajectory.prompt,
            3: steps[1].request.trajectory.response,
            4: steps[-1].request.trajectory.prompt,
            5: steps[-1].request.trajectory.response,
            6: steps[0].request.trajectory.prompt,
        }
        self.tokens = {value: key for key, value in self.words.items()}
        for step in steps[2:-1]:
            self.tokens[step.request.trajectory.prompt] = 6

    def encode(self, text, *, add_special_tokens=False):
        return [self.tokens[text]]

    def decode(self, tokens, *, skip_special_tokens=True):
        return " ".join(self.words[token] for token in tokens if token not in (0, 1))


def setup_preferences(tmp_path: Path, *, sst=SSTConfig(), regress=False, two_pairs=False):
    store, history, original, held, protected = _base(tmp_path / "authority")
    events, steps = fixture()
    if two_pairs:
        events, steps = fixture(
            actions=(steps[0].action, steps[1].action, steps[1].action, steps[2].action)
        )
    tokenizer = ActionTokenizer(steps)
    backend = TorchPolicyBackend(
        policy_model=ContextModel(),
        tokenizer=tokenizer,
        max_new_tokens=1,
        learning_rate=0.25,
        sst_config=sst,
        model_identifier="student",
    )
    if not sst.enabled:
        backend.optimizer = torch.optim.SGD(
            [*backend.policy_parameters(), *backend.value_head.parameters()], lr=0.25
        )
    steps = tuple(
        replace(
            step,
            request=RewardRequest(
                replace(
                    step.request.trajectory,
                    policy_version="model:v1",
                    prompt_token_ids=tuple(tokenizer.encode(step.request.trajectory.prompt)),
                    response_token_ids=tuple(tokenizer.encode(step.request.trajectory.response)),
                    sampling_parameters={
                        **step.request.trajectory.sampling_parameters,
                        "policy_parameter_checksum": backend.policy_parameter_checksum(),
                        "source_record_sha256": "a" * 64,
                    },
                ),
                step.request.observable_references,
            ),
        )
        for step in steps
    )
    updated = []
    for event in events:
        index = int(event.event_id.split("-")[1])
        payload = event.to_dict()["payload"]
        if event.event_type == "prediction_commit":
            payload["predicted_outcome"]["deloop"] = steps[index].commitment()
        verifier_refs = event.verifier_refs
        if index == 1 and event.event_type == "verification_result":
            verifier_refs = original.localization_verifier_refs
            payload["verifier_refs"] = list(verifier_refs)
        updated.append(
            replace(
                event,
                payload=payload,
                model_version="model:v1",
                harness_version="harness:v1",
                verifier_refs=verifier_refs,
            )
        )
    events = tuple(updated)
    refs = {ref.reference_id: ref for step in steps for ref in step.request.observable_references}
    typed = tuple((event.event_id, refs[ref]) for event in events for ref in event.evidence_refs)
    binding = store.register_trajectory(
        trajectory_id="action-1",
        source_epoch_id="epoch:source",
        events=events,
        event_evidence_refs=typed,
        source_evidence_refs=tuple(ref for _, ref in typed),
    )
    graph = original.failure_graph
    node = replace(
        graph.nodes[0],
        event_id="event-1-3",
        evidence_refs=(refs["observation-evidence-1"],),
    )
    correction = replace(
        original,
        source_trajectory_id="action-1",
        source_action_id="action-1",
        failure_graph=replace(graph, nodes=(node,)),
        source_evidence_refs=binding.source_evidence_refs,
        trajectory_digest=binding.trajectory_digest,
        teacher_evidence_refs=(EvidenceReference("host", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    config = DeLoopTrainingConfig(
        enabled=True,
        policy_id="loop-training:v1",
        reviewer_id=REVIEWER.evaluator_id,
        reviewer_version=REVIEWER.evaluator_version,
        reviewer_contract_id=REVIEWER.contract_id,
        evaluator_id="held-out-host:v1",
        dataset_digest="a" * 64,
        beta=1.0,
        seed=7,
    )
    evaluations = []

    def evaluate(candidate, generate):
        rollout = generate(
            RolloutRequest(
                steps[1].request.trajectory.prompt,
                policy_version=candidate.model_version,
                sampling_parameters={"max_new_tokens": 1, "do_sample": False},
                seed=7,
            )
        )
        evaluations.append(rollout)
        rows = tuple(
            _record("model:v2", "harness:v1", i, total=0.0 if regress else 0.9) for i in range(3)
        )
        for row in rows:
            append_epoch_record(history, row)
        close_evaluation_epoch(history)
        sh = store.issue_source_validation_receipt(
            epoch_id="epoch:source", split=ValidationSplit.HELD_OUT, records=(held,)
        )
        sp = store.issue_source_validation_receipt(
            epoch_id="epoch:source", split=ValidationSplit.PROTECTED, records=protected
        )
        suite = store.register_protected_suite(suite_id="protected:v5", source_receipt=sp)
        policy = MetricPolicy(
            "loop-metrics:v1",
            tuple(
                MetricSpec(name, MetricDirection.HIGHER_IS_BETTER, 0.0)
                for name in (
                    "correctness",
                    "calibration",
                    "abstention",
                    "epistemic_process",
                    "total",
                )
            ),
            "total",
            MetricAggregation.ARITHMETIC_MEAN,
        )
        store.register_metric_policy(policy)
        ch = store.issue_candidate_validation_receipt(
            candidate_id=candidate.candidate_id, split=ValidationSplit.HELD_OUT, records=rows[:1]
        )
        cp = store.issue_candidate_validation_receipt(
            candidate_id=candidate.candidate_id, split=ValidationSplit.PROTECTED, records=rows[1:]
        )
        comparison = store.issue_metric_comparison(
            candidate_id=candidate.candidate_id,
            policy_id=policy.policy_id,
            source_receipt=sh,
            candidate_receipt=ch,
        )
        return ValidationBundle(candidate, ch, cp, comparison, suite.suite_id)

    args = dict(
        task_id="task",
        config=config,
        review=review,
        authorize_source=lambda packet: True,
        authorize_training=lambda packet: True,
        evaluate=evaluate,
        candidate_id="candidate:loop",
        artifact_root=tmp_path / "trials",
    )
    return backend, events, steps, store, correction, args, evaluations


def test_default_off_never_inspects_inputs():
    assert run_loop_preferences(None, None, None, None, None) is None


@pytest.mark.parametrize("sst", (SSTConfig(), SSTConfig(enabled=True, rank_fraction=0.0)))
def test_real_preference_update_uses_original_state_and_existing_admission(tmp_path, sst):
    backend, events, steps, store, correction, args, evaluations = setup_preferences(
        tmp_path, sst=sst
    )
    checksum = backend.parameter_checksum()
    original_state = deepcopy(backend.policy_model.state_dict())
    rng = torch.get_rng_state().clone()
    result = run_loop_preferences(backend, events, steps, store, correction, **args)
    assert result.decision.accepted is True
    assert result.decision.execute_candidate is False
    assert result.backend is not backend
    assert backend.parameter_checksum() == checksum
    assert all(
        torch.equal(value, original_state[name])
        for name, value in backend.policy_model.state_dict().items()
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert (
        evaluations
        and result.backend.policy_parameter_checksum() != backend.policy_parameter_checksum()
    )
    assert result.backend.policy_model.gradient_inputs == [(2, 5), (2, 3)]
    assert all(
        torch.equal(value, backend.reference_model.state_dict()[name])
        for name, value in result.backend.reference_model.state_dict().items()
    )
    report = result.to_dict()
    assert report["pair_count"] == 1
    assert report["optimizer_step"] == 1
    assert report["loss"] == pytest.approx(0.69314718)
    with torch.no_grad():
        before = backend.policy_model(torch.tensor([[2]])).logits[0, -1]
        after = result.backend.policy_model(torch.tensor([[2]])).logits[0, -1]
    assert (after[5] - after[3]) > (before[5] - before[3])
    assert report["authority_granted"] is False
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(report | {"training_eligibility": "TRAIN"})
    restored = result.backend
    restored.load_checkpoint(Path(report["checkpoint_path"]))
    assert restored.policy_parameter_checksum() == report["candidate_policy_checksum"]


def test_protected_regression_retains_checkpoint_without_returning_model(tmp_path):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path, regress=True)
    result = run_loop_preferences(backend, events, steps, store, correction, **args)
    assert not result.decision.accepted
    assert result.backend is None
    assert Path(result.to_dict()["checkpoint_path"]).is_file()


@pytest.mark.parametrize("field", ("authorize_source", "authorize_training"))
@pytest.mark.parametrize("failure", ("withhold", "truthy", "mutate", "raise"))
def test_host_approval_fails_closed_without_mutating_source(tmp_path, field, failure):
    backend, events, steps, store, correction, args, evaluations = setup_preferences(tmp_path)
    checksum = backend.parameter_checksum()

    def callback(packet):
        if failure == "raise":
            raise RuntimeError("private callback details")
        if failure == "mutate":
            packet["status"] = "changed"
            return True
        return 1 if failure == "truthy" else False

    args[field] = callback
    with pytest.raises(ValueError):
        run_loop_preferences(backend, events, steps, store, correction, **args)
    assert backend.parameter_checksum() == checksum
    assert not evaluations
    assert not (tmp_path / "trials").exists()


def test_unsupported_original_state_recovery_cannot_train(tmp_path):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path)
    args["review"] = lambda candidate: review(candidate, original=False)
    result = run_loop_preferences(backend, events, steps, store, correction, **args)
    assert result.backend is None
    assert result.to_dict()["status"] == "no_eligible_pairs"
    assert not (tmp_path / "trials").exists()


@pytest.mark.parametrize("failure", ("held_out", "stale_policy", "wrong_source", "bad_tokens"))
def test_invalid_sources_rejected_before_callbacks(tmp_path, failure):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path)
    request = steps[1].request
    trajectory = request.trajectory
    if failure == "held_out":
        trajectory = replace(
            trajectory,
            sampling_parameters={
                **trajectory.sampling_parameters,
                "nested": {"training_eligibility": "HIDDEN_EVAL"},
            },
        )
    elif failure == "stale_policy":
        trajectory = replace(
            trajectory,
            sampling_parameters={
                **trajectory.sampling_parameters,
                "policy_parameter_checksum": "0" * 64,
            },
        )
    elif failure == "wrong_source":
        correction = replace(correction, source_trajectory_id="foreign")
    else:
        trajectory = replace(trajectory, response_token_ids=(5,))
    steps = (
        steps[0],
        replace(steps[1], request=RewardRequest(trajectory, request.observable_references)),
        steps[2],
    )
    calls = []
    args["authorize_source"] = lambda packet: calls.append(packet) or True
    args["review"] = lambda candidate: calls.append(candidate) or review(candidate)
    with pytest.raises(ValueError):
        run_loop_preferences(backend, events, steps, store, correction, **args)
    assert not calls


def test_config_is_closed_and_versioned():
    config = DeLoopTrainingConfig()
    assert DeLoopTrainingConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError):
        DeLoopTrainingConfig.from_dict(config.to_dict() | {"unexpected": True})


def test_changed_registered_peo_is_rejected_before_host_review(tmp_path):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path)
    changed = list(events)
    observed = changed[8]
    changed[8] = replace(observed, payload={**observed.payload, "actual_outcome": "changed"})
    calls = []
    args["authorize_source"] = lambda packet: calls.append(packet) or True
    with pytest.raises(ValueError, match="registered PEO source mismatch"):
        run_loop_preferences(backend, tuple(changed), steps, store, correction, **args)
    assert not calls


@pytest.mark.parametrize("beta", (None, 0.0, -1.0, float("nan"), True))
def test_enabled_training_requires_explicit_positive_finite_beta(tmp_path, beta):
    *_, args, _ = setup_preferences(tmp_path)
    with pytest.raises(ValueError):
        replace(args["config"], beta=beta)


def test_pair_budget_rejects_before_calling_reviewer_beyond_capacity(tmp_path):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path, two_pairs=True)
    calls = []
    args["config"] = replace(args["config"], max_pairs=1)
    # Count even rejected review attempts. A caller-owned review cannot stretch
    # this operator budget by rejecting arbitrarily many proposals.
    args["review"] = lambda candidate: calls.append(candidate) or review(candidate)
    with pytest.raises(ValueError, match="pair.*budget"):
        run_loop_preferences(backend, events, steps, store, correction, **args)
    assert len(calls) == 1


def test_evaluation_exception_is_sanitized_and_retains_trial(tmp_path):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path)

    def fail(candidate, generate):
        raise RuntimeError("private evaluator secret")

    args["evaluate"] = fail
    with pytest.raises(ValueError, match="independent candidate evaluation failed") as error:
        run_loop_preferences(backend, events, steps, store, correction, **args)
    assert "private evaluator" not in str(error.value)
    reports = list((tmp_path / "trials").glob("*/trial.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["status"] == "evaluation_failed"


def test_pair_reviewer_exception_does_not_expose_private_text(tmp_path):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path)

    def fail(candidate):
        raise RuntimeError("private pair reviewer secret")

    args["review"] = fail
    with pytest.raises(ValueError, match="independent pair review failed") as error:
        run_loop_preferences(backend, events, steps, store, correction, **args)
    assert "private pair" not in str(error.value)


def test_two_pairs_share_one_optimizer_step_and_original_rejected_histories(tmp_path):
    backend, events, steps, store, correction, args, _ = setup_preferences(tmp_path, two_pairs=True)
    result = run_loop_preferences(backend, events, steps, store, correction, **args)
    assert result.to_dict()["pair_count"] == 2
    assert result.to_dict()["optimizer_step"] == 1
    assert result.to_dict()["loss"] == pytest.approx(0.69314718)
    assert result.backend.policy_model.gradient_inputs == [(2, 5), (2, 3), (6, 5), (6, 3)]
