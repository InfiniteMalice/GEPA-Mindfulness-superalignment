"""Private promotion integration contracts using real durable evaluation catalogs."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from test_model_harness_coevolution import (
    _NAMES,
    _base,
    _event_evidence,
    _events,
    _record,
    _ref,
)

from evaluation import SystemIdentity
from gepa_mindfulness import (
    CoevolutionStore,
    EvaluationEpochStore,
    MetricAggregation,
    MetricDirection,
    MetricPolicy,
    MetricSpec,
    ValidationBundle,
    ValidationSplit,
    append_epoch_record,
    begin_candidate_epoch,
    close_evaluation_epoch,
)
from gepa_mindfulness.private_promotion import (
    ComputeBudget,
    ExperimentRequest,
    PrivatePromotionStore,
    PrivateProtocol,
    SeedUsage,
)
from gepa_mindfulness.private_promotion import (
    ExperimentOperation as Op,
)
from gepa_mindfulness.training.eligibility import require_training_eligible

DIGEST = "sha256:" + "a" * 64
SEEDS = (4200, 4201)
USAGE = tuple(SeedUsage(seed, 90, 4, 900) for seed in SEEDS)


def _protocol(**changes):
    return replace(
        PrivateProtocol(
            "private:v1",
            "metrics:v1",
            "protected:v1",
            SEEDS,
            ComputeBudget(100, 5, 1000),
            "sha256:" + "b" * 64,
            "sha256:" + "c" * 64,
            "sha256:" + "d" * 64,
        ),
        **changes,
    )


def _records(model, total=0.9):
    return tuple(
        replace(
            _record(model, "harness:v1", i, total=total),
            system=SystemIdentity(i, SEEDS[i % 2], model, "harness:v1"),
        )
        for i in range(4)
    )


class Flow:
    def __init__(self, tmp_path):
        # Reuse the already tested correction fixture, but register its trajectory here.
        _, _, proposal, _, _ = _base(tmp_path / "proposal")
        self.evaluation = EvaluationEpochStore(tmp_path / "epochs.sqlite", "private-eval")
        self.history = self.evaluation.create_root(
            lineage_id="main",
            epoch_id="epoch:source",
            model_version="model:v1",
            harness_version="harness:v1",
        )
        self.source = _records("model:v1", 0.8)
        for record in self.source:
            append_epoch_record(self.history, record)
        close_evaluation_epoch(self.history)
        begin_candidate_epoch(
            self.history,
            epoch_id="epoch:candidate",
            model_version="model:v2",
            harness_version="harness:v1",
        )
        self.coevolution = CoevolutionStore(
            tmp_path / "coevolution.sqlite",
            "private",
            self.evaluation,
            lineage_id="main",
        )
        binding = self.coevolution.register_trajectory(
            trajectory_id="trajectory:1",
            source_epoch_id="epoch:source",
            events=_events(),
            event_evidence_refs=_event_evidence(),
            source_evidence_refs=(_ref("evidence:prediction"), _ref()),
        )
        self.candidate = self.coevolution.register_candidate(
            candidate_id="candidate:1",
            artifact_digest=DIGEST,
            correction=replace(proposal, trajectory_digest=binding.trajectory_digest),
        )
        self.baseline = self.coevolution.issue_source_validation_receipt(
            epoch_id="epoch:source",
            split=ValidationSplit.HELD_OUT,
            records=self.source[:2],
        )
        protected = self.coevolution.issue_source_validation_receipt(
            epoch_id="epoch:source",
            split=ValidationSplit.PROTECTED,
            records=self.source[2:],
        )
        self.coevolution.register_protected_suite(suite_id="protected:v1", source_receipt=protected)
        self.coevolution.register_metric_policy(
            MetricPolicy(
                "metrics:v1",
                tuple(MetricSpec(n, MetricDirection.HIGHER_IS_BETTER, 0.0) for n in _NAMES),
                "total",
                MetricAggregation.ARITHMETIC_MEAN,
            )
        )
        self.path = tmp_path / "promotion.sqlite"
        self.store = self.reopen()

    def reopen(self, **changes):
        options = dict(enabled=True)
        options.update(changes)
        return PrivatePromotionStore(
            self.path,
            self.coevolution,
            self.evaluation,
            _protocol(),
            self.baseline,
            USAGE,
            **options,
        )

    def dispatch(self, operation, **changes):
        return self.store.dispatch(
            ExperimentRequest(
                operation,
                changes.get("candidate_id", "candidate:1"),
                changes.get("artifact_digest", DIGEST),
            )
        )

    def request(self):
        assert self.dispatch(Op.SUBMIT_CANDIDATE)["status"] == "submitted"
        assert self.dispatch(Op.REQUEST_PRIVATE_EVALUATION)["status"] == "pending"

    def bundle(self, total=0.9):
        records = _records("model:v2", total)
        for record in records:
            append_epoch_record(self.history, record)
        close_evaluation_epoch(self.history)
        held = self.coevolution.issue_candidate_validation_receipt(
            candidate_id="candidate:1",
            split=ValidationSplit.HELD_OUT,
            records=records[:2],
        )
        protected = self.coevolution.issue_candidate_validation_receipt(
            candidate_id="candidate:1",
            split=ValidationSplit.PROTECTED,
            records=records[2:],
        )
        comparison = self.coevolution.issue_metric_comparison(
            candidate_id="candidate:1",
            policy_id="metrics:v1",
            source_receipt=self.baseline,
            candidate_receipt=held,
        )
        return ValidationBundle(self.candidate, held, protected, comparison, "protected:v1")


@pytest.mark.parametrize("total,status", [(0.9, "accepted"), (0.7, "rejected")])
def test_private_flow_restarts_and_only_returns_allowlisted_status(tmp_path, total, status):
    flow = Flow(tmp_path)
    flow.request()
    flow.store = flow.reopen()
    result = flow.store.complete_evaluation("candidate:1", flow.bundle(total), USAGE)
    assert result == {
        "schema_version": "private-promotion-v1",
        "candidate_id": "candidate:1",
        "status": status,
        "training_eligibility": "HIDDEN_EVAL",
        "execute_candidate": False,
    }
    with pytest.raises(ValueError):
        require_training_eligible(result)
    flow.store = flow.reopen()
    assert flow.dispatch(Op.READ_PROMOTION_STATUS) == result
    assert flow.dispatch(Op.REQUEST_REVIEW)["status"] == "review_requested"
    events = flow.store.audit_events()
    assert len(events) == 5  # protocol, submission, request, completion, review
    assert events[-2]["payload"]["decision_id"]
    events.clear()
    assert len(flow.store.audit_events()) == 5


def test_disabled_before_database_creation(tmp_path):
    path = tmp_path / "disabled.sqlite"
    with pytest.raises(ValueError, match="enabled"):
        PrivatePromotionStore(path, None, None, None, None, None)
    assert not path.exists()


@pytest.mark.parametrize("operation", ["edit_evaluator", "launch_training", Op.REQUEST_REVIEW])
def test_invalid_operations_and_transitions_do_not_write(tmp_path, operation):
    flow = Flow(tmp_path)
    before = flow.store.audit_events()
    if isinstance(operation, str) and not isinstance(operation, Op):
        with pytest.raises(ValueError):
            ExperimentRequest(operation, "candidate:1", DIGEST)
    else:
        assert flow.dispatch(operation)["status"] == "invalid_request"
    assert flow.store.audit_events() == before


def test_replay_and_concurrent_requests_append_once(tmp_path):
    flow = Flow(tmp_path)
    assert flow.dispatch(Op.REQUEST_PRIVATE_EVALUATION)["status"] == "invalid_request"
    flow.dispatch(Op.SUBMIT_CANDIDATE)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: flow.dispatch(Op.REQUEST_PRIVATE_EVALUATION), range(8)))
    assert {r["status"] for r in results} == {"pending"}
    assert len(flow.store.audit_events()) == 3
    assert flow.dispatch(Op.SUBMIT_CANDIDATE)["status"] == "submitted"
    assert len(flow.store.audit_events()) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", (4200,)),
        ("seeds", (4200, 4200)),
        ("seeds", (True, 4201)),
        ("seeds", [4200, 4201]),
        ("generator_digest", "secret-path"),
        ("worlds_digest", ""),
        ("renderings_digest", DIGEST.upper()),
    ],
)
def test_protocol_rejects_invalid_inputs(field, value):
    with pytest.raises(ValueError):
        _protocol(**{field: value})


@pytest.mark.parametrize("value", [True, -1, 0, 1.5, float("nan"), "100"])
def test_budget_requires_positive_exact_integers(value):
    with pytest.raises(ValueError):
        ComputeBudget(value, 5, 1000)


@pytest.mark.parametrize(
    "usage",
    [
        (SeedUsage(4200, 90, 4, 900),),
        (SeedUsage(4200, 90, 4, 900), SeedUsage(4200, 90, 4, 900)),
        (SeedUsage(4200, 101, 4, 900), USAGE[1]),
        (SeedUsage(4200, 90, 6, 900), USAGE[1]),
        (SeedUsage(4200, 90, 4, 1001), USAGE[1]),
        (SeedUsage(4202, 90, 4, 900), USAGE[1]),
    ],
)
def test_candidate_usage_rejected_before_decision(tmp_path, usage):
    flow = Flow(tmp_path)
    flow.request()
    bundle = flow.bundle()
    with pytest.raises(ValueError):
        flow.store.complete_evaluation("candidate:1", bundle, usage)
    assert len(flow.store.audit_events()) == 3
    assert flow.dispatch(Op.READ_PROMOTION_STATUS)["status"] == "pending"


def test_completion_and_failure_are_terminal(tmp_path):
    flow = Flow(tmp_path)
    flow.request()
    assert flow.store.record_failure("candidate:1")["status"] == "failed"
    assert flow.store.record_failure("candidate:1")["status"] == "failed"
    with pytest.raises(ValueError):
        flow.store.complete_evaluation("candidate:1", flow.bundle(), USAGE)
    assert flow.dispatch(Op.REQUEST_REVIEW)["status"] == "review_requested"


def test_completion_requires_request_and_open_empty_request_epoch(tmp_path):
    flow = Flow(tmp_path)
    bundle = flow.bundle()
    assert flow.dispatch(Op.SUBMIT_CANDIDATE)["status"] == "invalid_request"
    with pytest.raises(ValueError):
        flow.store.complete_evaluation("candidate:1", bundle, USAGE)


def test_identity_substitution_and_mutated_request_fail_closed(tmp_path):
    flow = Flow(tmp_path)
    assert (
        flow.dispatch(Op.SUBMIT_CANDIDATE, artifact_digest="sha256:" + "f" * 64)["status"]
        == "invalid_request"
    )
    request = ExperimentRequest(Op.SUBMIT_CANDIDATE, "candidate:1", DIGEST)
    object.__setattr__(request, "operation", "edit_evaluator")
    assert flow.store.dispatch(request)["status"] == "invalid_request"
    assert len(flow.store.audit_events()) == 1


def test_sql_audit_update_and_delete_are_rejected(tmp_path):
    flow = Flow(tmp_path)
    flow.request()
    with sqlite3.connect(flow.path) as db:
        for sql in ("DELETE FROM promotion_events", "UPDATE promotion_events SET payload='{}'"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute(sql)
    assert len(flow.store.audit_events()) == 3


def test_changed_pinned_protocol_is_rejected(tmp_path):
    flow = Flow(tmp_path)
    with pytest.raises(ValueError, match="configuration"):
        PrivatePromotionStore(
            flow.path,
            flow.coevolution,
            flow.evaluation,
            _protocol(budget=ComputeBudget(200, 5, 1000)),
            flow.baseline,
            USAGE,
            enabled=True,
        )


def test_private_catalog_error_is_not_disclosed(tmp_path):
    flow = Flow(tmp_path)
    with sqlite3.connect(flow.path) as db:
        db.execute("DROP TRIGGER promotion_events_no_update")
        db.execute(
            "UPDATE promotion_events SET payload=?",
            (json.dumps({"secret": "withheld-secret-canary"}),),
        )
    result = flow.dispatch(Op.SUBMIT_CANDIDATE)
    assert result["status"] == "invalid_request"
    assert "withheld-secret-canary" not in json.dumps(result)
    with pytest.raises(ValueError, match="audit"):
        flow.store.audit_events()


@pytest.mark.parametrize("field", ["metric_policy_id", "protected_suite_id"])
def test_unknown_protocol_dependencies_fail_before_creating_database(tmp_path, field):
    flow = Flow(tmp_path)
    path = tmp_path / "other.sqlite"
    with pytest.raises(ValueError, match="registered"):
        PrivatePromotionStore(
            path,
            flow.coevolution,
            flow.evaluation,
            _protocol(**{field: "unknown"}),
            flow.baseline,
            USAGE,
            enabled=True,
        )
    assert not path.exists()


def test_wrong_evaluation_catalog_is_rejected(tmp_path):
    flow = Flow(tmp_path)
    other = Flow(tmp_path / "other")
    with pytest.raises(ValueError):
        PrivatePromotionStore(
            tmp_path / "other-promotion.sqlite",
            flow.coevolution,
            other.evaluation,
            _protocol(),
            other.baseline,
            USAGE,
            enabled=True,
        )


def test_source_seed_coverage_and_usage_are_required(tmp_path):
    flow = Flow(tmp_path)
    for protocol, usage in (
        (_protocol(seeds=(4200, 4202)), (USAGE[0], SeedUsage(4202, 1, 1, 1))),
        (_protocol(), (USAGE[0],)),
        (_protocol(), (SeedUsage(4200, 101, 1, 1), USAGE[1])),
    ):
        with pytest.raises(ValueError):
            PrivatePromotionStore(
                tmp_path / "new.sqlite",
                flow.coevolution,
                flow.evaluation,
                protocol,
                flow.baseline,
                usage,
                enabled=True,
            )
    assert not (tmp_path / "new.sqlite").exists()


def test_source_split_overlap_is_rejected(tmp_path):
    flow = Flow(tmp_path)
    receipt = flow.coevolution.issue_source_validation_receipt(
        epoch_id="epoch:source",
        split=ValidationSplit.PROTECTED,
        records=flow.source[:2],
    )
    flow.coevolution.register_protected_suite(suite_id="overlap", source_receipt=receipt)
    with pytest.raises(ValueError, match="disjoint"):
        PrivatePromotionStore(
            tmp_path / "new.sqlite",
            flow.coevolution,
            flow.evaluation,
            _protocol(protected_suite_id="overlap"),
            flow.baseline,
            USAGE,
            enabled=True,
        )


@pytest.mark.parametrize("substitution", ["policy", "baseline", "suite", "candidate", "seed"])
def test_completion_cannot_substitute_private_protocol_evidence(tmp_path, substitution):
    flow = Flow(tmp_path)
    flow.request()
    bundle = flow.bundle()
    if substitution == "suite":
        bundle = replace(bundle, protected_suite_id="other")
    elif substitution == "candidate":
        other = Flow(tmp_path / "other")
        bundle = other.bundle()
    elif substitution == "seed":
        single = flow.coevolution.issue_candidate_validation_receipt(
            candidate_id="candidate:1",
            split=ValidationSplit.HELD_OUT,
            records=(_records("model:v2")[0],),
        )
        bundle = replace(bundle, held_out_receipt=single)
    else:
        metric = bundle.metric_receipt.to_dict()
        metric["policy_id" if substitution == "policy" else "source_receipt_id"] = "other"
        bundle = replace(bundle, metric_receipt=type(bundle.metric_receipt).from_dict(metric))
    with pytest.raises(ValueError):
        flow.store.complete_evaluation("candidate:1", bundle, USAGE)
    assert len(flow.store.audit_events()) == 3


def test_request_must_precede_first_record_not_only_epoch_closure(tmp_path):
    flow = Flow(tmp_path)
    flow.dispatch(Op.SUBMIT_CANDIDATE)
    append_epoch_record(flow.history, _records("model:v2")[0])
    assert flow.dispatch(Op.REQUEST_PRIVATE_EVALUATION)["status"] == "invalid_request"
    assert len(flow.store.audit_events()) == 2


def test_completed_status_revalidates_canonical_dependencies(tmp_path):
    flow = Flow(tmp_path)
    flow.request()
    flow.store.complete_evaluation("candidate:1", flow.bundle(), USAGE)
    with sqlite3.connect(tmp_path / "coevolution.sqlite") as db:
        db.execute("DELETE FROM metric_receipts")
    assert flow.dispatch(Op.READ_PROMOTION_STATUS)["status"] == "invalid_request"


def test_crash_after_decision_before_audit_can_retry(tmp_path, monkeypatch):
    flow = Flow(tmp_path)
    flow.request()
    bundle = flow.bundle()
    append = PrivatePromotionStore._append

    def fail_append(*args):
        raise RuntimeError("simulated process failure")

    monkeypatch.setattr(PrivatePromotionStore, "_append", staticmethod(fail_append))
    with pytest.raises(RuntimeError):
        flow.store.complete_evaluation("candidate:1", bundle, USAGE)
    monkeypatch.setattr(PrivatePromotionStore, "_append", staticmethod(append))
    assert flow.store.complete_evaluation("candidate:1", bundle, USAGE)["status"] == "accepted"
    with sqlite3.connect(tmp_path / "coevolution.sqlite") as db:
        assert db.execute("SELECT count(*) FROM decisions").fetchone()[0] == 1


def test_boundary_usage_is_accepted_and_completion_is_not_repeated(tmp_path):
    flow = Flow(tmp_path)
    flow.request()
    bundle = flow.bundle()
    usage = tuple(SeedUsage(seed, 100, 5, 1000) for seed in SEEDS)
    assert flow.store.complete_evaluation("candidate:1", bundle, usage)["status"] == "accepted"
    with pytest.raises(ValueError, match="pending"):
        flow.store.complete_evaluation("candidate:1", bundle, usage)
    with pytest.raises(ValueError, match="pending"):
        flow.store.record_failure("candidate:1")
    assert len(flow.store.audit_events()) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens", True),
        ("tool_calls", -1),
        ("wall_time_ms", float("inf")),
        ("seed", 2**53),
    ],
)
def test_usage_validates_exact_values(field, value):
    with pytest.raises(ValueError):
        replace(USAGE[0], **{field: value})
