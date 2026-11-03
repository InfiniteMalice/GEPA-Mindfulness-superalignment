"""Active improvement dispatch and bounded JSON-only isolated harness trials."""

# Standard library
import hashlib
import json
import os
from dataclasses import replace

# Third-party
import pytest

# isort: split
# Local
from test_controlled_improvement import _estimate, _supported, _triage
from test_improvement_workflows import EVALUATOR, diagnosis, review
from test_model_harness_coevolution import _NAMES, _base, _record

from evaluation.causal_records import canonical_json
from gepa_mindfulness import (
    CandidateComponent,
    MetricAggregation,
    MetricDirection,
    MetricPolicy,
    MetricSpec,
    ValidationBundle,
    ValidationSplit,
    append_epoch_record,
    close_evaluation_epoch,
)
from gepa_mindfulness.governed_improvement_runtime import (
    HarnessProcedureConfig,
    HarnessTrialConfig,
    dispatch_improvement_workflow,
    run_harness_improvement,
)
from gepa_mindfulness.learning_surfaces import LessonKind


def setup_trial(tmp_path, *, protected_pass=True, worker_text=None, policy_id="policy:v1"):
    store, history, proposal, held, protected = _base(
        tmp_path / "catalog", (CandidateComponent.HARNESS,)
    )
    proposal = _supported(proposal)
    request = diagnosis(proposal, LessonKind.STABLE_PROCEDURAL_CONVENTION)
    worker = tmp_path / "worker.py"
    worker.write_text(
        worker_text
        or (
            "import json,os,sys\n"
            "job=json.load(sys.stdin)\n"
            "print(json.dumps({'order':job['harness']['parameters']['order'],"
            "'pid':os.getpid(),'cwd':os.getcwd(),'model':job['model_version']}))\n"
        )
    )
    before = HarnessProcedureConfig(
        "harness:v1", "retrieval_sequencing", canonical_json({"order": ["source:a", "source:b"]})
    )
    after = replace(
        before,
        harness_version="harness:v2",
        parameters_json=canonical_json({"order": ["source:b", "source:a"]}),
    )
    observations = []

    def evaluate(candidate, run):
        observation = run({"task_id": "held:1", "prompt": "Use the public evidence"})
        observations.append(observation)
        model, harness = candidate.model_version, candidate.harness_version
        candidate_held = _record(model, harness, 0, total=0.9, calibration=0.7)
        candidate_protected = tuple(
            _record(model, harness, i, total=0.9, passed=protected_pass) for i in (1, 2)
        )
        for record in (candidate_held, *candidate_protected):
            append_epoch_record(history, record)
        close_evaluation_epoch(history)
        source_held = store.issue_source_validation_receipt(
            epoch_id="epoch:source", split=ValidationSplit.HELD_OUT, records=(held,)
        )
        source_protected = store.issue_source_validation_receipt(
            epoch_id="epoch:source", split=ValidationSplit.PROTECTED, records=protected
        )
        suite = store.register_protected_suite(
            suite_id="protected:v5", source_receipt=source_protected
        )
        policy = MetricPolicy(
            policy_id,
            tuple(
                MetricSpec(
                    name,
                    (
                        MetricDirection.LOWER_IS_BETTER
                        if name == "calibration"
                        else MetricDirection.HIGHER_IS_BETTER
                    ),
                    0.0,
                )
                for name in _NAMES
            ),
            "total",
            MetricAggregation.ARITHMETIC_MEAN,
        )
        store.register_metric_policy(policy)
        held_receipt = store.issue_candidate_validation_receipt(
            candidate_id=candidate.candidate_id,
            split=ValidationSplit.HELD_OUT,
            records=(candidate_held,),
        )
        protected_receipt = store.issue_candidate_validation_receipt(
            candidate_id=candidate.candidate_id,
            split=ValidationSplit.PROTECTED,
            records=candidate_protected,
        )
        comparison = store.issue_metric_comparison(
            candidate_id=candidate.candidate_id,
            policy_id=policy.policy_id,
            source_receipt=source_held,
            candidate_receipt=held_receipt,
        )
        return ValidationBundle(
            candidate, held_receipt, protected_receipt, comparison, suite.suite_id
        )

    args = dict(
        config=HarnessTrialConfig(
            enabled=True,
            trial_id="harness-trial:1",
            worker_path=str(worker),
            worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        ),
        evaluator=EVALUATOR,
        review=review,
        authenticate=lambda packet: True,
        evaluate=evaluate,
        candidate_id="candidate:harness:1",
        artifact_root=tmp_path / "artifacts",
    )
    return store, proposal, request, before, after, args, observations


def trial(values):
    store, proposal, request, before, after, args, _ = values
    return run_harness_improvement(
        store, proposal, _estimate(), _triage(), request, before, after, **args
    )


def test_default_off_before_catalog_and_files():
    assert run_harness_improvement(None, None, None, None, None, None, None) is None
    assert dispatch_improvement_workflow(None, None, None, None, None) is None


def test_actual_subprocess_trial_uses_existing_protected_admission_without_deployment(tmp_path):
    values = setup_trial(tmp_path)
    result = trial(values)
    assert result.decision.accepted
    assert result.configuration == values[4]
    assert result.decision.execute_candidate is False
    assert result.decision.rollback_target_epoch_id == "epoch:source"
    observed = values[-1][0]
    assert observed["status"] == "completed"
    assert observed["output"]["order"] == ["source:b", "source:a"]
    assert observed["output"]["pid"] != os.getpid()
    assert observed["output"]["model"] == "model:v1"
    assert result.to_dict()["training_eligibility"] == "HIDDEN_EVAL"
    assert result.to_dict()["authority_granted"] is False


def test_protected_regression_withholds_live_configuration(tmp_path):
    values = setup_trial(tmp_path, protected_pass=False)
    # Existing receipt authority rejects a failed protected outcome before acceptance.
    with pytest.raises(ValueError, match="outcome.*pass"):
        trial(values)
    audit = next(values[5]["artifact_root"].rglob("trial.json"))
    assert json.loads(audit.read_text())["status"] == "validation_failed"


def test_host_denial_does_not_launch_worker_or_register_candidate(tmp_path):
    values = setup_trial(tmp_path)
    values[5]["authenticate"] = lambda packet: False
    result = trial(values)
    assert result.decision is None and result.configuration is None
    assert not values[-1]
    assert not values[5]["artifact_root"].exists()


def test_human_review_route_never_invokes_authentication_or_worker(tmp_path):
    values = list(setup_trial(tmp_path))
    values[2] = replace(
        values[2], characteristics=replace(values[2].characteristics, normative=True)
    )
    values[5]["authenticate"] = lambda _: pytest.fail("human route reached execution")
    result = trial(values)
    assert result.to_dict()["status"] == "human_review"
    assert not values[-1]


def test_forbidden_policy_fields_cannot_enter_procedure_configuration():
    with pytest.raises(ValueError):
        HarnessProcedureConfig(
            "h:v2",
            "retrieval_sequencing",
            canonical_json({"order": ["source:a"], "verifier_policy": "weaken"}),
        )
    with pytest.raises(ValueError):
        HarnessProcedureConfig(
            "h:v2",
            "context_compaction",
            canonical_json(
                {"max_tokens": 128, "preserve_sources": False, "preserve_action_history": True}
            ),
        )


def test_worker_digest_mismatch_fails_before_any_trial(tmp_path):
    values = setup_trial(tmp_path)
    worker = values[5]["config"].worker_path
    with open(worker, "a") as stream:
        stream.write("# changed\n")
    with pytest.raises(ValueError, match="worker"):
        trial(values)
    assert not values[-1]


def test_incompatible_source_harness_version_fails_closed(tmp_path):
    values = list(setup_trial(tmp_path))
    values[3] = replace(values[3], harness_version="unregistered:harness")
    with pytest.raises(ValueError, match="version"):
        trial(values)


def test_changed_candidate_artifact_is_not_accepted(tmp_path):
    values = setup_trial(tmp_path)
    evaluate = values[5]["evaluate"]

    def mutate(candidate, run):
        bundle = evaluate(candidate, run)
        artifact = next(values[5]["artifact_root"].rglob("harness.json"))
        artifact.write_text("{}")
        return bundle

    values[5]["evaluate"] = mutate
    with pytest.raises(ValueError, match="changed"):
        trial(values)


@pytest.mark.parametrize(
    "worker_text,status",
    (
        ("import time; time.sleep(10)", "timeout"),
        ("print('x'*100000)", "failed"),
    ),
)
def test_worker_resource_failures_are_retained_and_not_accepted(tmp_path, worker_text, status):
    values = setup_trial(tmp_path, worker_text=worker_text)
    values[5]["config"] = replace(
        values[5]["config"],
        timeout_seconds=0.1 if status == "timeout" else 5.0,
        max_output_bytes=1024,
    )
    result = trial(values)
    assert values[-1][0]["status"] == status
    assert result.configuration is None
    assert result.to_dict()["status"] == "worker_failed"


def test_workflow_dispatch_uses_supported_surface_and_fresh_host_admission(tmp_path):
    store, _, proposal, *_ = _base(tmp_path / "dispatch")
    proposal = _supported(proposal)
    calls = []
    result = dispatch_improvement_workflow(
        store,
        proposal,
        _estimate(),
        _triage(),
        diagnosis(proposal),
        enabled=True,
        evaluator=EVALUATOR,
        review=review,
        authenticate=lambda _: True,
        handlers={"model_correction": lambda packet: calls.append(packet["workflow"]) or "trial"},
    )
    assert result.invoked and result.result == "trial"
    assert calls == ["model_correction"]
    assert result.to_dict()["authority_granted"] is False


def test_config_roundtrip_and_unknown_fields_rejected():
    cfg = HarnessTrialConfig()
    assert type(cfg).from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        type(cfg).from_dict(cfg.to_dict() | {"modify_verifier": True})


def test_candidate_cannot_replace_the_fixed_eligible_reviewer_pool(tmp_path):
    values = list(setup_trial(tmp_path))
    request = replace(values[2], procedure="review_routing")
    values[2] = request
    values[3] = HarnessProcedureConfig(
        "harness:v1",
        "review_routing",
        canonical_json({"preferred_reviewers": ["independent:one", "independent:two"]}),
    )
    values[4] = HarnessProcedureConfig(
        "harness:v2", "review_routing", canonical_json({"preferred_reviewers": ["self:reviewer"]})
    )
    with pytest.raises(ValueError, match="eligible"):
        trial(values)
    assert not values[-1]


def test_denied_workflow_dispatch_never_invokes_model_repair(tmp_path):
    store, _, proposal, *_ = _base(tmp_path / "dispatch")
    proposal = _supported(proposal)
    result = dispatch_improvement_workflow(
        store,
        proposal,
        _estimate(),
        _triage(),
        diagnosis(proposal),
        enabled=True,
        evaluator=EVALUATOR,
        review=review,
        authenticate=lambda _: False,
        handlers={"model_correction": lambda _: pytest.fail("denied handler invoked")},
    )
    assert not result.invoked


def test_evaluator_input_mutation_cannot_rewrite_retained_public_evidence(tmp_path):
    values = setup_trial(tmp_path)
    evaluate = values[5]["evaluate"]

    def change_input(candidate, run):
        def invoke(task):
            result = run(task)
            task["prompt"] = "rewritten after execution"
            return result

        return evaluate(candidate, invoke)

    values[5]["evaluate"] = change_input
    result = trial(values)
    assert result.to_dict()["observations"][0]["task"]["prompt"] == "Use the public evidence"


def test_public_worker_callable_expires_after_independent_evaluation(tmp_path):
    values = setup_trial(tmp_path)
    evaluate = values[5]["evaluate"]
    saved = []

    def capture(candidate, run):
        saved.append(run)
        return evaluate(candidate, run)

    values[5]["evaluate"] = capture
    result = trial(values)
    with pytest.raises(ValueError, match="closed"):
        saved[0]({"task_id": "late", "prompt": "outside the admitted evaluation"})
    assert len(result.to_dict()["observations"]) == 1


def test_concurrent_worker_calls_cannot_exceed_the_evaluation_budget(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from threading import Event, current_thread

    values = setup_trial(tmp_path)
    values[5]["config"] = replace(values[5]["config"], max_calls=1)
    evaluate = values[5]["evaluate"]
    entered, second_read = Event(), Event()
    read_bytes = Path.read_bytes
    errors = []

    def delayed_read(path):
        if path.name == "harness.json" and current_thread().name.startswith("worker-budget"):
            if not entered.is_set():
                entered.set()
                second_read.wait(0.2)
            else:
                second_read.set()
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", delayed_read)

    def concurrent(candidate, run):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="worker-budget") as executor:
            first = executor.submit(run, {"task_id": "first", "prompt": "Public evidence"})
            assert entered.wait(1)
            second = executor.submit(run, {"task_id": "second", "prompt": "Public evidence"})
            observed = first.result()
            try:
                second.result()
            except ValueError as error:
                errors.append(str(error))
        return evaluate(candidate, lambda task: observed)

    values[5]["evaluate"] = concurrent
    result = trial(values)
    assert result.decision.accepted
    assert len(result.to_dict()["observations"]) == 1
    assert len(errors) == 1


def test_maximum_advertised_output_budget_runs_small_valid_worker(tmp_path):
    values = setup_trial(tmp_path)
    values[5]["config"] = replace(values[5]["config"], max_output_bytes=1048576)
    result = trial(values)
    assert result.decision.accepted
    assert result.to_dict()["observations"][0]["status"] == "completed"


def test_worker_hash_seed_is_reproducible_without_cwd_or_user_site_imports(tmp_path):
    worker = (
        "import json,os,sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'hash':str(hash('same task')),"
        "'no_user_site':sys.flags.no_user_site, 'ignore_environment':sys.flags.ignore_environment,"
        "'cwd_import': '' in sys.path or os.getcwd() in sys.path}))\n"
    )
    values = setup_trial(tmp_path, worker_text=worker)
    evaluate = values[5]["evaluate"]

    def repeated(candidate, run):
        rows = [run({"task_id": "same", "prompt": "same task"}) for _ in range(3)]
        return evaluate(candidate, lambda task: rows[0])

    values[5]["evaluate"] = repeated
    result = trial(values)
    rows = result.to_dict()["observations"]
    assert all(row["output"]["ignore_environment"] == 0 for row in rows)
    assert all(row["output"]["no_user_site"] == 1 for row in rows)
    assert not any(row["output"]["cwd_import"] for row in rows)
    assert len({row["output"]["hash"] for row in rows}) == 1


def test_output_that_exceeds_enclosing_audit_depth_is_a_retained_failure(tmp_path):
    worker = (
        "import json\nvalue={}\n"
        "for _ in range(29): value={'nested':value}\n"
        "print(json.dumps(value))\n"
    )
    values = setup_trial(tmp_path, worker_text=worker)
    result = trial(values)
    assert result.configuration is None
    row = result.to_dict()["observations"][0]
    assert row["status"] == "failed" and row["output"] is None
    audit = next(values[5]["artifact_root"].rglob("trial.json"))
    assert json.loads(audit.read_text())["observations"][0]["status"] == "failed"


def test_large_decision_metadata_stays_durable_without_overflowing_observation_audit(tmp_path):
    worker = "import json; print(json.dumps({'blob':'x'*1019000}))"
    values = setup_trial(tmp_path, worker_text=worker, policy_id="policy:" + "p" * 120000)
    values[5]["config"] = replace(values[5]["config"], max_output_bytes=1020000)
    evaluate = values[5]["evaluate"]

    def capture_many(candidate, run):
        rows = [run({"task_id": "large:" + str(i), "prompt": "Public evidence"}) for i in range(4)]
        return evaluate(candidate, lambda task: rows[0])

    values[5]["evaluate"] = capture_many
    result = trial(values)
    assert result.decision.accepted
    audit_path = next(values[5]["artifact_root"].rglob("trial.json"))
    audit = json.loads(audit_path.read_text())
    assert len(canonical_json(audit).encode("utf-8")) <= 4 * 1024 * 1024
    reference = audit["decision_reference"]
    assert reference["decision_digest"] == result.decision.decision_digest
    assert reference["input_digest"] == result.decision.input_digest
    assert values[0].read_decision(reference["decision_id"]) == result.decision
    assert "decision" not in audit
