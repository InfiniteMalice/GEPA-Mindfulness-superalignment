"""Functional CPU trials; synthetic fixtures do not establish research effectiveness."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_model_harness_coevolution import _base, _record
from test_rl_torch_backend import TinyTokenizer
from torch import nn

from gepa_mindfulness import (
    CandidateSystem,
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
from gepa_mindfulness.training.locd_runtime import LoCDTrainingConfig, run_local_correction
from gepa_mindfulness.training.subspace_tuning import SSTConfig
from gepa_mindfulness.training.trajectory import RolloutRequest


class ContextModel(nn.Module):
    """A seven-token autoregressive fixture whose predictions depend on its weights."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=7, _name_or_path="context-fixture")
        self.embedding = nn.Embedding(7, 7)
        self.lm_head = nn.Linear(7, 7, bias=False)
        self.inputs: list[tuple[int, ...]] = []
        self.gradient_inputs: list[tuple[int, ...]] = []
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(7))
            self.lm_head.weight.zero_()
            self.lm_head.weight[3, 2] = 5
            self.lm_head.weight[4, 3] = 0.1
            self.lm_head.weight[5, 6] = 5

    def forward(self, input_ids, *, attention_mask=None, output_hidden_states=False):
        self.inputs.append(tuple(input_ids[0].tolist()))
        if torch.is_grad_enabled():
            self.gradient_inputs.append(tuple(input_ids[0].tolist()))
        hint = (input_ids == 6).cumsum(-1).clamp(max=1).unsqueeze(-1)
        hidden = self.embedding(input_ids) + hint * self.embedding.weight[6]
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))

    def generate(self, input_ids, *, attention_mask, max_new_tokens, **_sampling):
        tokens = input_ids
        for _ in range(max_new_tokens):
            next_token = self(tokens).logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=-1)
        return tokens


def setup_trial(tmp_path: Path, *, sst: SSTConfig = SSTConfig()):
    store, history, correction, held, protected = _base(tmp_path / "authority")
    tokenizer = TinyTokenizer()
    tokenizer._tokens["hint"] = 6
    tokenizer._words[6] = "hint"
    backend = TorchPolicyBackend(
        policy_model=ContextModel(),
        tokenizer=tokenizer,
        max_new_tokens=2,
        learning_rate=0.5,
        sst_config=sst,
    )
    # Use SGD to make the analytically constructed one-step correction transparent.
    # Native SST owns its optimizer; keep the native AdamW when SST is enabled.
    if not sst.enabled:
        backend.optimizer = torch.optim.SGD(
            [*backend.policy_parameters(), *backend.value_head.parameters()], lr=0.5
        )
    source = backend.generate((RolloutRequest("calm", policy_version="model:v1", seed=7),))[0]
    assert source.response == "breath now"
    source = replace(
        source,
        trajectory_id="trajectory:1",
        sampling_parameters={
            **source.sampling_parameters,
            "training_eligibility": "TRAIN",
            "policy_parameter_checksum": backend.policy_parameter_checksum(),
            "source_record_sha256": "a" * 64,
        },
        evidence_references=correction.source_evidence_refs,
    )
    correction = replace(
        correction,
        teacher_evidence_refs=(
            EvidenceReference("teacher:source", EvidenceSourceKind.EXTERNAL_RECORD),
        ),
    )
    config = LoCDTrainingConfig(
        enabled=True,
        policy_id="local-training:v1",
        reviewer_id="host-review:v1",
        evaluator_id="host-evaluation:v1",
        dataset_digest="b" * 64,
        seed=7,
    )
    calls = []

    def evaluate(candidate: CandidateSystem, generate):
        # No training context/labels or trainable backend is supplied to this host role.
        rollout = generate(RolloutRequest("calm", policy_version="model:v2", seed=7))
        calls.append(rollout)
        passed = rollout.response == "breath slowly"
        total = 0.9 if passed else 0.0
        rows = tuple(_record("model:v2", "harness:v1", i, total=total) for i in range(3))
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
            "local-metrics:v1",
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
            candidate_id=candidate.candidate_id,
            split=ValidationSplit.HELD_OUT,
            records=rows[:1],
        )
        cp = store.issue_candidate_validation_receipt(
            candidate_id=candidate.candidate_id,
            split=ValidationSplit.PROTECTED,
            records=rows[1:],
        )
        comparison = store.issue_metric_comparison(
            candidate_id=candidate.candidate_id,
            policy_id=policy.policy_id,
            source_receipt=sh,
            candidate_receipt=ch,
        )
        return ValidationBundle(candidate, ch, cp, comparison, suite.suite_id)

    arguments = dict(
        config=config,
        start=1,
        end=2,
        teacher_context="hint",
        authorize_source=lambda _payload: True,
        authorize_correction=lambda _payload: True,
        evaluate=evaluate,
        candidate_id="candidate:local",
        artifact_root=tmp_path / "trials",
    )
    return backend, RewardRequest(source), store, correction, arguments, calls


def test_default_off_does_not_inspect_inputs():
    assert run_local_correction(None, None, None, None) is None


@pytest.mark.parametrize("sst", (SSTConfig(), SSTConfig(enabled=True, rank_fraction=0.0)))
def test_real_local_update_isolated_and_evaluated_without_context(tmp_path: Path, sst: SSTConfig):
    backend, request, store, correction, args, evaluations = setup_trial(tmp_path, sst=sst)
    original_checksum = backend.parameter_checksum()
    rng = torch.get_rng_state().clone()
    original_inputs = list(backend.policy_model.inputs)
    result = run_local_correction(backend, request, store, correction, **args)
    assert backend.parameter_checksum() == original_checksum
    assert backend.policy_model.inputs == original_inputs
    assert torch.equal(torch.get_rng_state(), rng)
    assert result.decision.accepted is True
    assert result.decision.execute_candidate is False
    assert result.backend is not backend
    assert result.backend.policy_parameter_checksum() != backend.policy_parameter_checksum()
    report = result.to_dict()
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["authority_granted"] is False
    assert report["segment"]["loss_mask"] == [False, True]
    assert report["segment"]["student_inputs"]["response_token_ids"] == [3, 5]
    assert report["public_reevaluation"]["response_token_ids"] == [3, 5]
    assert evaluations[0].response == "breath slowly"
    assert result.backend.policy_model.gradient_inputs == [(2, 3, 5)]
    assert (2, 6, 3, 5) in result.backend.policy_model.inputs
    checkpoint = Path(report["checkpoint_path"])
    assert "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        result.decision.candidate.artifact_digest
    )
    assert checkpoint.parent.joinpath("trial.json").is_file()
    restored = deepcopy(backend)
    restored.load_checkpoint(checkpoint)
    assert restored.parameter_checksum() == result.backend.parameter_checksum()
    restored_state = restored.optimizer.state_dict()
    expected_state = result.backend.optimizer.state_dict()
    assert restored_state["param_groups"] == expected_state["param_groups"]
    assert restored_state["state"].keys() == expected_state["state"].keys()
    for key, state in expected_state["state"].items():
        for field, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(restored_state["state"][key][field], value)
            else:
                assert restored_state["state"][key][field] == value


@pytest.mark.parametrize("stage", ("authorize_source", "authorize_correction"))
@pytest.mark.parametrize("mode", ("deny", "truthy", "mutate", "exception"))
def test_reviews_fail_closed_without_modifying_source(tmp_path: Path, stage: str, mode: str):
    backend, request, store, correction, args, evaluations = setup_trial(tmp_path)
    before = backend.parameter_checksum()

    def review(payload):
        if mode == "mutate":
            payload["teacher_context"] = "rewritten"
            return True
        if mode == "exception":
            raise RuntimeError("private detail must not escape")
        return 1 if mode == "truthy" else False

    args[stage] = review
    with pytest.raises(ValueError, match="review withheld"):
        run_local_correction(backend, request, store, correction, **args)
    assert backend.parameter_checksum() == before
    assert evaluations == []
    assert not args["artifact_root"].exists()


@pytest.mark.parametrize("restriction", ("HIDDEN_EVAL", "REGRESSION", "DEVELOPMENT"))
def test_nested_nontraining_provenance_rejected(tmp_path: Path, restriction: str):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    source = replace(
        request.trajectory,
        sampling_parameters={
            **request.trajectory.sampling_parameters,
            "source_record": {"training_eligibility": restriction},
        },
    )
    args["authorize_source"] = lambda _value: pytest.fail("review reached")
    with pytest.raises(ValueError, match="forbids optimization"):
        run_local_correction(backend, RewardRequest(source), store, correction, **args)


def test_full_rank_sst_reports_no_change_without_catalog_admission(tmp_path: Path):
    backend, request, store, correction, args, evaluations = setup_trial(
        tmp_path, sst=SSTConfig(enabled=True, rank_fraction=1.0)
    )
    result = run_local_correction(backend, request, store, correction, **args)
    assert result.backend is None
    assert result.decision is None
    assert result.to_dict()["status"] == "no_model_change"
    assert evaluations == []


def test_configuration_roundtrip_and_strict_schema():
    config = LoCDTrainingConfig()
    assert LoCDTrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        LoCDTrainingConfig.from_dict({**config.to_dict(), "authorize": True})
    with pytest.raises(ValueError):
        LoCDTrainingConfig(enabled=True)


@pytest.mark.parametrize(
    "field,value",
    (
        ("max_parameters", 1),
        ("max_tokens", 1),
        ("max_logit_elements", 1),
    ),
)
def test_model_budgets_fail_before_host_review(tmp_path: Path, field: str, value: int):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    args["config"] = replace(args["config"], **{field: value})
    args["authorize_source"] = lambda _value: pytest.fail("review reached")
    with pytest.raises(ValueError, match="budget"):
        run_local_correction(backend, request, store, correction, **args)


@pytest.mark.parametrize(
    "change,expected",
    (
        ({"policy_parameter_checksum": "0" * 64}, "checksum mismatch"),
        ({"training_eligibility": "DEVELOPMENT"}, "TRAIN declaration"),
        ({"source_record_sha256": "unbound"}, "SHA"),
    ),
)
def test_source_admission_bindings(tmp_path: Path, change: dict, expected: str):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    source = replace(
        request.trajectory,
        sampling_parameters={
            **request.trajectory.sampling_parameters,
            **change,
        },
    )
    with pytest.raises(ValueError, match=expected):
        run_local_correction(backend, RewardRequest(source), store, correction, **args)


def test_private_teacher_evidence_cannot_authorize_training(tmp_path: Path):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    correction = replace(
        correction,
        teacher_evidence_refs=(EvidenceReference("private", EvidenceSourceKind.PRIVATE_REASONING),),
    )
    with pytest.raises(ValueError, match="observable trusted teacher"):
        run_local_correction(backend, request, store, correction, **args)


def test_failed_independent_evaluation_preserves_checkpoint_and_original(tmp_path: Path):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    before = backend.parameter_checksum()

    def broken(_candidate, _generate):
        raise RuntimeError("private evaluator detail")

    args["evaluate"] = broken
    with pytest.raises(ValueError, match="independent candidate evaluation failed"):
        run_local_correction(backend, request, store, correction, **args)
    assert backend.parameter_checksum() == before
    assert len(list(args["artifact_root"].glob("*/backend.pt"))) == 1
    report = json.loads(next(args["artifact_root"].glob("*/trial.json")).read_text())
    assert report["status"] == "evaluation_failed"
    assert "private evaluator detail" not in canonical_report(report)


def canonical_report(report):
    """Render detached audit data for a private-detail assertion."""
    return json.dumps(report, sort_keys=True)


def test_protected_regression_rejects_candidate(tmp_path: Path):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    evaluate = args["evaluate"]

    def regress(candidate, generate):
        # Actual held-out fixture output determines this independent result.
        return evaluate(candidate, lambda rollout: replace(generate(rollout), response="failed"))

    args["evaluate"] = regress
    result = run_local_correction(backend, request, store, correction, **args)
    assert result.decision.accepted is False
    assert result.backend is None
    assert result.to_dict()["status"] == "catalog_rejected"
    assert result.decision.rollback_target_epoch_id == "epoch:source"


def test_evaluation_sampling_cannot_allocate_unbounded_output(tmp_path: Path):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)

    def unbounded(_candidate, generate):
        return generate(
            RolloutRequest(
                "calm", policy_version="model:v2", sampling_parameters={"max_new_tokens": 10**12}
            )
        )

    args["evaluate"] = unbounded
    with pytest.raises(ValueError, match="independent candidate evaluation failed"):
        run_local_correction(backend, request, store, correction, **args)


def test_invalid_bundle_retains_completed_public_rollout_and_terminal_status(tmp_path: Path):
    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    args["evaluate"] = lambda _candidate, _generate: None
    with pytest.raises(ValueError, match="validation bundle"):
        run_local_correction(backend, request, store, correction, **args)
    report = json.loads(next(args["artifact_root"].glob("*/trial.json")).read_text())
    assert report["status"] == "validation_failed"
    assert report["public_reevaluation"]["response"] == "breath slowly"


@pytest.mark.parametrize("stop_after_teacher", (False, True))
@pytest.mark.parametrize("inherited_mode", ("beam", "contrastive"))
def test_real_transformers_inherited_beams_cannot_expand_generation_budget(
    tmp_path: Path,
    stop_after_teacher: bool,
    inherited_mode: str,
):
    from transformers import GPT2Config, GPT2LMHeadModel

    backend, request, store, correction, args, _ = setup_trial(tmp_path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
        )
    )
    model.generation_config.num_beams = 4
    backend = TorchPolicyBackend(
        policy_model=model,
        tokenizer=backend.tokenizer,
        max_new_tokens=2,
        learning_rate=0.01,
    )
    captured = backend.generate(
        (
            RolloutRequest(
                "calm",
                policy_version="model:v1",
                seed=7,
                sampling_parameters={"num_beams": 1, "num_return_sequences": 1, "do_sample": False},
            ),
        )
    )[0]
    captured = replace(
        captured,
        trajectory_id=request.trajectory.trajectory_id,
        sampling_parameters={
            **request.trajectory.sampling_parameters,
            "policy_parameter_checksum": backend.policy_parameter_checksum(),
        },
        evidence_references=request.trajectory.evidence_references,
    )
    if inherited_mode == "contrastive":
        model.generation_config.penalty_alpha = 0.6
        model.generation_config.top_k = 4
    shapes = []

    def inspect_forward(_model, positional, keyword):
        tokens = positional[0] if positional else keyword["input_ids"]
        shapes.append(tuple(tokens.shape))

    model.register_forward_pre_hook(inspect_forward, with_kwargs=True)
    args["config"] = replace(args["config"], max_logit_elements=28)
    if stop_after_teacher:
        args["authorize_correction"] = lambda _payload: False
        expected = "correction review withheld"
    else:
        args["evaluate"] = lambda _candidate, _generate: None
        expected = "validation bundle"
    with pytest.raises(ValueError, match=expected):
        run_local_correction(backend, RewardRequest(captured), store, correction, **args)
    assert shapes
    assert all(batch == 1 and batch * time * 7 <= 28 for batch, time in shapes), shapes
