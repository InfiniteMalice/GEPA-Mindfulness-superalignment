"""Black-box generation, independent eligibility and native Eq. 7 integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from test_actionopd_verified_credit import supported_outcome
from test_research_credit_ledger import request
from test_research_credit_runtime import policy

from gepa_mindfulness.training.actionopd import (
    DecisionKind,
    SemanticAction,
    TeacherActionSample,
    actionopd_advantages,
)
from gepa_mindfulness.training.actionopd_runtime import (
    ActionOPDBatchPreparer,
    ActionOPDConfig,
    build_actionopd_provider,
)
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger
from gepa_mindfulness.training.runtime_config import AlgorithmConfig, PolicyConfig, RLRunConfig


def settings(**changes):
    values = dict(
        enabled=True,
        teacher_id="teacher",
        teacher_version="v1",
        teacher_contract_id="generation:v1",
        verifier_id="independent",
        verifier_version="v1",
        verifier_contract_id="verification:v1",
        source_review_policy_id="operator-source:v1",
        teacher_samples=2,
        sampling_config_json='{"seed":7}',
    )
    values.update(changes)
    return ActionOPDConfig(**values)


def runtime(tmp_path, *, config=None, **changes):
    config = config or settings()
    selected_policy = policy(coefficients=(("verified_decision", config.credit_weight),))
    queries = []

    def sample(packet, index):
        queries.append(packet)
        action = SemanticAction(DecisionKind.GROUNDED_ANSWER, '{"answer":"answer"}')
        if packet["prompt"] == "other" and index == 1:
            action = None
        return TeacherActionSample(
            packet["state_digest"] + ":" + str(index),
            config.teacher,
            packet["state_digest"],
            config.sampling_config_json,
            action,
        )

    def verify(source, _action):
        return supported_outcome(source)

    arguments = dict(
        policy=selected_policy,
        ledger=ResearchCreditLedger(tmp_path / "credits.sqlite", authority_domain="host"),
        authorize_source=lambda _payload: True,
        sample_teacher=sample,
        parse_action=lambda source: SemanticAction(
            DecisionKind.GROUNDED_ANSWER, json.dumps({"answer": source.trajectory.response})
        ),
        verify_action=verify,
        account_event=lambda source: (
            "semantic:" + source.trajectory.trajectory_id,
            (
                ("provider", "not_counted"),
                ("reward_integrity", "not_counted"),
                ("causal_reward", "not_counted"),
                ("knowledge", "not_counted"),
                ("progress", "not_counted"),
                ("process", "not_counted"),
            ),
        ),
        authenticate_claims=lambda _envelope: True,
        authorize_optimizer_use=lambda _envelope: True,
        authorize_accounting=lambda _envelope: True,
    )
    arguments.update(changes)
    return build_actionopd_provider(config, **arguments), queries, selected_policy


def test_default_off_does_not_inspect_host_objects():
    assert build_actionopd_provider() is None
    assert build_actionopd_provider(policy=object(), ledger=object()) is None


def test_blackbox_teacher_gets_only_public_state_and_is_cached_per_state(tmp_path):
    provider, queries, _ = runtime(tmp_path)
    source = request()
    source = replace(
        source,
        trajectory=replace(
            source.trajectory,
            sampling_parameters={
                **source.trajectory.sampling_parameters,
                "source_record": {"private_expected_answer": "secret-ground-truth"},
            },
        ),
    )
    first = provider.score(source)
    second_source = replace(source, trajectory=replace(source.trajectory, trajectory_id="second"))
    second = provider.score(second_source)
    assert first.optimizer_reward() == second.optimizer_reward() == 1.0
    assert len(queries) == 2
    for query in queries:
        encoded = json.dumps(query)
        assert "response" not in query
        assert "secret-ground-truth" not in encoded
        assert "private_expected_answer" not in encoded
    report = first.to_dict()
    claim = report["review"]["audit"]["claims"][0]
    assert (
        json.loads(claim["source_metadata_json"])["verified_decision"]["verification"][
            "claimed_outcome_supported"
        ]
        is True
    )


def test_source_review_denial_precedes_teacher_access(tmp_path):
    provider, queries, _ = runtime(tmp_path, authorize_source=lambda _payload: False)
    with pytest.raises(ValueError, match="proposer failed"):
        provider.score(request())
    assert queries == []


@pytest.mark.parametrize("mutation", ("identity", "state", "sampling", "duplicate"))
def test_teacher_generation_bindings_fail_closed(tmp_path, mutation):
    config = settings()

    def invalid(packet, index):
        return TeacherActionSample(
            "same" if mutation == "duplicate" else str(index),
            (
                replace(config.teacher, evaluator_version="wrong")
                if mutation == "identity"
                else config.teacher
            ),
            "0" * 64 if mutation == "state" else packet["state_digest"],
            "{}" if mutation == "sampling" else config.sampling_config_json,
            SemanticAction(DecisionKind.GROUNDED_ANSWER, '{"answer":"answer"}'),
        )

    provider, _, _ = runtime(tmp_path, config=config, sample_teacher=invalid)
    with pytest.raises(ValueError, match="proposer failed"):
        provider.score(request())


def test_prohibited_teacher_match_cannot_receive_optimizer_credit(tmp_path):
    config = settings()
    shortcut = SemanticAction(DecisionKind.TOOL_USE, '{"tool":"unauthorized-delete"}')

    def sample(packet, index):
        return TeacherActionSample(
            str(index),
            config.teacher,
            packet["state_digest"],
            config.sampling_config_json,
            shortcut,
        )

    provider, _, _ = runtime(
        tmp_path,
        sample_teacher=sample,
        parse_action=lambda _source: shortcut,
        verify_action=lambda source, _action: replace(
            supported_outcome(source), authorization_scope_valid=False
        ),
    )
    with pytest.raises(ValueError, match="proposer failed"):
        provider.score(request())


@pytest.mark.parametrize("binding", ("absent", "disabled", "different"))
def test_missing_or_different_native_config_fails_before_teacher_calls(tmp_path, binding):
    from gepa_mindfulness.training.engine import RLTrainingEngine

    provider, queries, credit_policy = runtime(tmp_path)
    method = {
        "absent": None,
        "disabled": ActionOPDConfig(),
        "different": settings(teacher_version="v2"),
    }[binding]
    with pytest.raises(ValueError, match="matching configured host teacher"):
        RLTrainingEngine._score(
            (request().trajectory,),
            provider,
            research_policy=credit_policy,
            actionopd_config=method,
        )
    assert queries == []


@pytest.mark.parametrize("operation", ("response", "metadata"))
def test_source_capture_cannot_change_actual_output_or_retained_origin(tmp_path, operation):
    from gepa_mindfulness.training.actionopd_runtime import capture_actionopd_sources
    from gepa_mindfulness.training.trajectory import RolloutRequest

    def capture(trajectory):
        if operation == "response":
            trajectory = replace(trajectory, response="forged")
        else:
            trajectory.sampling_parameters["source_record"]["marker"] = "forged"
        return RewardRequest(trajectory, request().observable_references)

    provider, queries, _ = runtime(tmp_path, capture_source=capture)
    source = request().trajectory
    selected = RolloutRequest(
        source.prompt,
        policy_version=source.policy_version,
        metadata={"training_eligibility": "TRAIN", "marker": "original"},
    )
    with pytest.raises(ValueError, match="changed the actual student trajectory"):
        capture_actionopd_sources(settings(), provider, (selected,), (source,))
    assert selected.metadata["marker"] == "original"
    assert queries == []


@pytest.mark.parametrize("restriction", ("HIDDEN_EVAL", "REGRESSION", "DEVELOPMENT"))
def test_heldout_source_cannot_be_relabelled_during_capture(tmp_path, restriction):
    from gepa_mindfulness.training.actionopd_runtime import capture_actionopd_sources
    from gepa_mindfulness.training.trajectory import RolloutRequest

    provider, queries, _ = runtime(
        tmp_path, capture_source=lambda _source: pytest.fail("capture reached")
    )
    source = request().trajectory
    selected = RolloutRequest(
        source.prompt,
        policy_version=source.policy_version,
        metadata={
            "training_eligibility": "TRAIN",
            "source_record": {
                "training_eligibility": restriction,
            },
        },
    )
    with pytest.raises(ValueError, match="forbids optimization"):
        capture_actionopd_sources(settings(), provider, (selected,), (source,))
    assert queries == []


def test_global_whitening_matches_eq7_across_distinct_student_states(tmp_path):
    config = settings()
    provider, _, credit_policy = runtime(tmp_path, config=config)
    targets = tuple(
        RewardRequest(
            replace(
                request().trajectory,
                trajectory_id=f"sample:{index}",
                prompt=prompt,
                response=response,
                prompt_token_ids=(2,),
                response_token_ids=(3,),
            ),
            request().observable_references,
        )
        for index, (prompt, response) in enumerate(
            (
                ("question", "answer"),
                ("question", "other-answer"),
                ("other", "answer"),
                ("other", "other-answer"),
            )
        )
    )
    rewards = tuple(provider.score(target) for target in targets)
    scored = tuple(
        reward.bind_trajectory(target.trajectory) for reward, target in zip(rewards, targets)
    )
    run_config = RLRunConfig(
        algorithm=AlgorithmConfig(name="grpo", group_size=2),
        policy=PolicyConfig(do_sample=True),
        research_credit=credit_policy,
        actionopd=config,
    )
    prepared = ActionOPDBatchPreparer(run_config).prepare(scored, rewards, object())
    expected = actionopd_advantages(((1.5, 0.0), (1.0, 0.0)), epsilon=config.whitening_epsilon)
    assert len(prepared.trajectories) == 4
    observed = [row.advantage[0] for row in prepared.trajectories]
    assert observed == pytest.approx([*expected[0], *expected[1]])
    assert abs(observed[0]) != pytest.approx(abs(observed[2]))


def test_native_config_is_hashed_and_default_hash_preserved():
    from gepa_mindfulness.training.engine import _config_hash

    baseline = RLRunConfig()
    assert (
        _config_hash(baseline) == "9f0306ad79e6ed93e0bc40b3918270e12538960e4061456be7f91d6fb1f6f1b2"
    )
    config = settings()
    credit_policy = policy(coefficients=(("verified_decision", config.credit_weight),))
    configured = replace(
        baseline,
        algorithm=AlgorithmConfig(name="grpo", group_size=2),
        policy=PolicyConfig(do_sample=True),
        research_credit=credit_policy,
        actionopd=config,
    )
    assert _config_hash(configured) != _config_hash(baseline)
    assert _config_hash(replace(configured, actionopd=replace(config, teacher_version="v2"))) != (
        _config_hash(configured)
    )


@pytest.mark.parametrize("with_sst", (False, True))
def test_canonical_native_actionopd_updates_real_weights_and_resumes(tmp_path, with_sst):
    import torch
    from test_rl_engine_cpu import TinyLocalCausalLM, TinyLocalTokenizer, _config

    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
    from gepa_mindfulness.training.backends import TorchPolicyBackend
    from gepa_mindfulness.training.engine import RLTrainingEngine
    from gepa_mindfulness.training.subspace_tuning import SSTConfig

    class SamplingModel(TinyLocalCausalLM):
        """Sample two legitimate fixture answers from the actual current policy."""

        def generate(self, input_ids, *, attention_mask, max_new_tokens, **_parameters):
            logits = self(input_ids).logits[:, -1, 4:6]
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1) + 4
            return torch.cat((input_ids, next_token), dim=-1)

    config = _config(tmp_path, "grpo")
    source_path = tmp_path / "pairs.jsonl"
    # A new admitted prompt source with two valid answers; preference is not eligibility.
    record = {
        "record_id": "valid-answers:1",
        "training_eligibility": "TRAIN",
        "prompt": "practice slowly",
        "valid_answers": ["chosen", "rejected"],
    }
    source_path.write_text(json.dumps(record, sort_keys=True) + "\n")

    class Dataset:
        """Existing host DatasetProvider interface over the pinned fixture source."""

        def materialize(self, _mode):
            from gepa_mindfulness.training.trajectory import RolloutRequest

            return (RolloutRequest(record["prompt"], case_id=record["record_id"], metadata=record),)

    method = settings()
    credit_policy = policy(
        coefficients=(("verified_decision", method.credit_weight),),
        dataset_digest=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    config = replace(
        config,
        actionopd=method,
        research_credit=credit_policy,
        algorithm=replace(config.algorithm, group_size=8),
        sst=SSTConfig(enabled=with_sst, rank_fraction=0.5),
    )
    observations = {}
    queries = []
    backends = []

    def capture(trajectory):
        reference = EvidenceReference(
            trajectory.trajectory_id + ":observed", EvidenceSourceKind.EXTERNAL_RECORD
        )
        # Host fixture observation, not a student-generated certificate.
        observations[reference.reference_id] = trajectory.response in {"chosen", "rejected"}
        observed = replace(trajectory, evidence_references=(reference,))
        return RewardRequest(observed, (reference,))

    def sample(packet, index):
        queries.append(packet)
        return TeacherActionSample(
            packet["state_digest"] + ":" + str(index),
            method.teacher,
            packet["state_digest"],
            method.sampling_config_json,
            SemanticAction(DecisionKind.GROUNDED_ANSWER, '{"answer":"chosen"}'),
        )

    def verify(source, _action):
        assert all(observations[ref.reference_id] for ref in source.observable_references)
        return supported_outcome(source)

    ledger = ResearchCreditLedger(tmp_path / "credits.sqlite", authority_domain="host")

    def build_provider(_config):
        return build_actionopd_provider(
            method,
            policy=credit_policy,
            ledger=ledger,
            capture_source=capture,
            authorize_source=lambda _review: True,
            sample_teacher=sample,
            parse_action=lambda source: SemanticAction(
                DecisionKind.GROUNDED_ANSWER, json.dumps({"answer": source.trajectory.response})
            ),
            verify_action=verify,
            account_event=lambda source: (
                "fixture-decision:" + source.trajectory.trajectory_id,
                (
                    ("provider", "not_counted"),
                    ("reward_integrity", "not_counted"),
                    ("causal_reward", "not_counted"),
                ),
            ),
            authenticate_claims=lambda _review: True,
            authorize_optimizer_use=lambda _review: True,
            authorize_accounting=lambda _review: True,
        )

    def build_backend(value):
        backend = TorchPolicyBackend(
            policy_model=SamplingModel(),
            tokenizer=TinyLocalTokenizer(),
            learning_rate=value.algorithm.learning_rate,
            max_new_tokens=1,
            model_identifier=value.policy.model_name,
            sst_config=value.sst,
        )
        backends.append(backend)
        return backend

    result = RLTrainingEngine(
        config,
        dataset_factory=lambda _value: Dataset(),
        backend_factory=build_backend,
        reward_factory=build_provider,
    ).train(max_steps=1)
    assert result.global_step == 1
    assert result.policy_parameters_updated is True
    assert len(queries) == 2
    rows = [
        json.loads(line)
        for line in (result.log_directory / "metrics.jsonl").read_text().splitlines()
    ]
    assert any(
        row["schema_version"] == 4 and "research_verified_decision_raw" in row["metrics"]
        for row in rows
    )
    resumed = RLTrainingEngine(
        config,
        dataset_factory=lambda _value: Dataset(),
        backend_factory=build_backend,
        reward_factory=build_provider,
    ).resume(result.checkpoint.path, max_steps=1)
    assert resumed.global_step == 2
    assert resumed.policy_parameters_updated is True
    assert len(queries) == 4
