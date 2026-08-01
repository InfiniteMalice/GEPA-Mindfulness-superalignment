"""Closed-schema and append-safety tests for shared RL run logging."""

from __future__ import annotations

import json
import multiprocessing
from copy import deepcopy
from pathlib import Path

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.run_logging import (
    LOG_SCHEMA_VERSION,
    JSONLLoggingSink,
    MetricRecord,
    RunManifest,
    TrajectoryRecord,
)
from gepa_mindfulness.training.trajectory import Trajectory


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="run-1",
        algorithm="ppo",
        backend="torch_portable",
        actor_backend="torch_portable",
        learner_backend="torch_portable",
        model="tiny-local",
        reference_model="tiny-local-reference",
        adapter=None,
        dataset_hash="a" * 64,
        config_hash="b" * 64,
        seed=42,
        software_versions={"python": "3.11", "torch": "2.9"},
        device_capabilities={"device": "cpu", "supports_backward": True},
        start_time="2026-07-31T12:00:00Z",
        checkpoint_parent=None,
    )


def _metric(*, record_id: str = "metric-1", scope: str = "aggregate") -> MetricRecord:
    return MetricRecord(
        record_id=record_id,
        run_id="run-1",
        timestamp="2026-07-31T12:00:01Z",
        global_step=3,
        scope=scope,
        backend="torch_portable",
        actor_backend="torch_portable",
        learner_backend="torch_portable",
        policy_version="policy-3",
        metrics={
            "total_reward": 0.7,
            "policy_loss": -0.2,
            "value_loss": 0.1,
            "entropy": 0.3,
            "kl": 0.02,
            "learning_rate": 1e-5,
            "honesty": 0.8,
        },
    )


def _trajectory_record() -> TrajectoryRecord:
    evidence = EvidenceReference("output-1", EvidenceSourceKind.OBSERVABLE_OUTPUT)
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        case_id="case-1",
        prompt="Question?",
        response="Observable answer",
        prompt_token_ids=(1, 2),
        response_token_ids=(3, 4),
        old_log_probs=(-0.2, -0.3),
        reference_log_probs=(-0.4, -0.5),
        value_predictions=(0.1, 0.2),
        reward_total=0.5,
        reward_components={"honesty": -0.2, "task_success": 0.7},
        backend_name="torch_portable",
        backend_version="2.9",
        model_identifier="tiny-local",
        policy_version="policy-3",
        trace_references=("legacy-trace",),
        evidence_references=(evidence,),
        reward_component_evidence={"honesty": (evidence,)},
    )
    return TrajectoryRecord(
        record_id="trajectory-record-1",
        run_id="run-1",
        timestamp="2026-07-31T12:00:02Z",
        global_step=3,
        backend="torch_portable",
        actor_backend="torch_portable",
        learner_backend="torch_portable",
        policy_version="policy-3",
        trajectory=trajectory,
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _concurrent_duplicate_writer(directory: str, start: object) -> None:
    event = start
    event.wait()
    sink = JSONLLoggingSink(Path(directory), rank=1)
    record = _metric(record_id="shared-response-" + "x" * 500_000, scope="response")
    for _ in range(10):
        sink.log_metrics(record)


def _racing_manifest_writer(
    directory: str,
    run_id: str,
    start: object,
    results: object,
) -> None:
    manifest = _manifest().to_dict()
    manifest["run_id"] = run_id
    event = start
    queue = results
    event.wait()
    try:
        started = JSONLLoggingSink(Path(directory), rank=0).start_run(manifest)
        queue.put((run_id, "started" if started else "same"))
    except Exception as exc:  # pragma: no cover - asserted through process result
        queue.put((run_id, f"error:{type(exc).__name__}:{exc}"))


def test_rank_zero_start_run_writes_exact_manifest_and_empty_jsonl_files(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)

    sink.start_run(_manifest())

    payload = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "actor_backend",
        "adapter",
        "algorithm",
        "backend",
        "checkpoint_parent",
        "config_hash",
        "dataset_hash",
        "device_capabilities",
        "learner_backend",
        "model",
        "reference_model",
        "run_id",
        "schema_version",
        "seed",
        "software_versions",
        "start_time",
    }
    assert payload["schema_version"] == LOG_SCHEMA_VERSION
    assert (tmp_path / "metrics.jsonl").read_bytes() == b""
    assert (tmp_path / "trajectories.jsonl").read_bytes() == b""


def test_nonzero_rank_does_not_write_manifest_or_aggregate_metrics(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=1)

    assert sink.start_run(_manifest()) is False
    assert sink.log_metrics(_metric()) is False

    assert list(tmp_path.iterdir()) == []


def test_nonzero_rank_can_write_response_metrics_without_duplicate_records(tmp_path: Path) -> None:
    rank_zero = JSONLLoggingSink(tmp_path, rank=0)
    rank_zero.start_run(_manifest())
    rank_one = JSONLLoggingSink(tmp_path, rank=1)
    record = _metric(record_id="response-1", scope="response")

    assert rank_one.log_metrics(record) is True
    assert rank_zero.log_metrics(record) is False

    records = _jsonl(tmp_path / "metrics.jsonl")
    assert len(records) == 1
    assert records[0]["record_id"] == "response-1"


def test_parallel_ranks_append_one_complete_record_for_a_shared_id(tmp_path: Path) -> None:
    rank_zero = JSONLLoggingSink(tmp_path, rank=0)
    rank_zero.start_run(_manifest())
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_concurrent_duplicate_writer, args=(str(tmp_path), start))
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)

    assert all(process.exitcode == 0 for process in processes)
    records = _jsonl(tmp_path / "metrics.jsonl")
    assert [record["record_id"] for record in records] == ["shared-response-" + "x" * 500_000]


def test_metric_record_preserves_individual_reward_and_loss_components(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())

    assert sink.log_metrics(_metric()) is True

    record = _jsonl(tmp_path / "metrics.jsonl")[0]
    assert record["backend"] == "torch_portable"
    assert record["actor_backend"] == "torch_portable"
    assert record["learner_backend"] == "torch_portable"
    assert record["policy_version"] == "policy-3"
    assert record["metrics"] == {
        "entropy": 0.3,
        "honesty": 0.8,
        "kl": 0.02,
        "learning_rate": 1e-5,
        "policy_loss": -0.2,
        "total_reward": 0.7,
        "value_loss": 0.1,
    }


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object()])
def test_metric_json_values_must_be_serializable_and_finite(
    tmp_path: Path,
    invalid: object,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    payload = _metric().to_dict()
    payload["metrics"] = {"policy_loss": invalid}

    with pytest.raises((TypeError, ValueError), match="JSON|finite"):
        sink.log_metrics(payload)

    assert (tmp_path / "metrics.jsonl").read_bytes() == b""


def test_metric_mapping_uses_a_closed_schema(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    payload = _metric().to_dict()
    payload["unrecognized"] = True

    with pytest.raises(ValueError, match="metric record fields"):
        sink.log_metrics(payload)

    assert (tmp_path / "metrics.jsonl").read_bytes() == b""


def test_metric_names_use_the_closed_cross_backend_schema(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    payload = _metric().to_dict()
    payload["metrics"] = {"invented_metric": 1.0}

    with pytest.raises(ValueError, match="metric name"):
        sink.log_metrics(payload)

    assert (tmp_path / "metrics.jsonl").read_bytes() == b""


def test_trajectory_log_preserves_typed_evidence_and_reward_components(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())

    assert sink.log_trajectory(_trajectory_record()) is True

    record = _jsonl(tmp_path / "trajectories.jsonl")[0]
    trajectory = record["trajectory"]
    assert trajectory["trace_references"] == ["legacy-trace"]
    assert trajectory["evidence_references"] == [
        {"reference_id": "output-1", "source_kind": "observable_output"}
    ]
    assert trajectory["reward_components"] == {"honesty": -0.2, "task_success": 0.7}
    assert trajectory["reward_component_evidence"] == {
        "honesty": [{"reference_id": "output-1", "source_kind": "observable_output"}]
    }


def test_jsonl_appends_complete_newline_delimited_records_and_deduplicates(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    first = _metric(record_id="metric-1")
    second = _metric(record_id="metric-2")

    assert sink.log_metrics(first) is True
    assert sink.log_metrics(second) is True
    assert sink.log_metrics(first) is False

    raw = (tmp_path / "metrics.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2
    assert [record["record_id"] for record in _jsonl(tmp_path / "metrics.jsonl")] == [
        "metric-1",
        "metric-2",
    ]


def test_jsonl_rejects_a_conflicting_record_that_reuses_an_existing_id(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    original = _metric(record_id="metric-1")
    conflicting = original.to_dict()
    conflicting["metrics"] = {"policy_loss": 0.9}
    assert sink.log_metrics(original) is True

    with pytest.raises(ValueError, match="record_id.*different payload"):
        sink.log_metrics(conflicting)

    assert _jsonl(tmp_path / "metrics.jsonl") == [original.to_dict()]


def test_start_run_is_idempotent_only_for_the_same_manifest(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    manifest = _manifest()
    assert sink.start_run(manifest) is True
    assert sink.start_run(manifest) is False
    conflicting = manifest.to_dict()
    conflicting["run_id"] = "different"

    with pytest.raises(ValueError, match="different run manifest"):
        sink.start_run(conflicting)


def test_multiprocess_start_run_publishes_exactly_one_conflicting_manifest(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_racing_manifest_writer,
            args=(str(tmp_path), run_id, start, results),
        )
        for run_id in ("run-a", "run-b")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)

    assert all(process.exitcode == 0 for process in processes)
    outcomes = [results.get(timeout=5) for _ in processes]
    assert sorted(outcome.split(":", 1)[0] for _, outcome in outcomes) == ["error", "started"]
    published = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    winner = next(run_id for run_id, outcome in outcomes if outcome == "started")
    assert published["run_id"] == winner
    assert (tmp_path / "metrics.jsonl").is_file()
    assert (tmp_path / "trajectories.jsonl").is_file()
    assert list(tmp_path.glob(".run_manifest.json.tmp-*")) == []


def test_start_run_rejects_stale_nonempty_stream_before_manifest_publication(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "metrics.jsonl").write_text('{"stale":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty.*metrics.jsonl"):
        JSONLLoggingSink(tmp_path, rank=0).start_run(_manifest())

    assert not (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == '{"stale":true}\n'


def test_start_run_same_manifest_repairs_a_missing_stream(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    assert sink.start_run(_manifest()) is True
    (tmp_path / "trajectories.jsonl").unlink()

    assert sink.start_run(_manifest()) is False

    assert (tmp_path / "trajectories.jsonl").read_bytes() == b""


def test_start_run_rejects_symlink_stream_without_publishing_manifest(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-metrics.jsonl"
    outside.write_text("operator-data", encoding="utf-8")
    try:
        (tmp_path / "metrics.jsonl").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        JSONLLoggingSink(tmp_path, rank=0).start_run(_manifest())

    assert not (tmp_path / "run_manifest.json").exists()
    assert outside.read_text(encoding="utf-8") == "operator-data"


@pytest.mark.parametrize("field", ["backend", "actor_backend", "learner_backend"])
@pytest.mark.parametrize("kind", ["metric", "trajectory"])
def test_records_must_match_run_backend_provenance(
    tmp_path: Path,
    field: str,
    kind: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    if kind == "metric":
        payload = _metric().to_dict()
        stream = tmp_path / "metrics.jsonl"
        log = sink.log_metrics
    else:
        payload = _trajectory_record().to_dict()
        stream = tmp_path / "trajectories.jsonl"
        log = sink.log_trajectory
    payload[field] = "incompatible-backend"
    if kind == "trajectory" and field == "backend":
        trajectory = payload["trajectory"]
        assert isinstance(trajectory, dict)
        trajectory["backend_name"] = "incompatible-backend"

    with pytest.raises(ValueError, match="provenance"):
        log(payload)

    assert stream.read_bytes() == b""


def _exercise_existing_metric_history(
    sink: JSONLLoggingSink,
    path: Path,
    payloads: list[dict[str, object]],
    operation: str,
) -> None:
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    if operation == "start":
        sink.start_run(_manifest())
    else:
        sink.log_metrics(_metric(record_id="new-record"))


@pytest.mark.parametrize("operation", ["start", "append"])
def test_complete_history_rejects_malformed_record_hidden_by_compatible_duplicate(
    tmp_path: Path,
    operation: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    compatible = _metric(record_id="duplicate").to_dict()
    malformed = {"record_id": "duplicate"}

    with pytest.raises(ValueError, match="metric record fields"):
        _exercise_existing_metric_history(
            sink,
            tmp_path / "metrics.jsonl",
            [malformed, compatible],
            operation,
        )


@pytest.mark.parametrize("operation", ["start", "append"])
def test_complete_history_rejects_provenance_conflict_hidden_by_compatible_duplicate(
    tmp_path: Path,
    operation: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    compatible = _metric(record_id="duplicate").to_dict()
    incompatible = dict(compatible)
    incompatible["backend"] = "incompatible-backend"

    with pytest.raises(ValueError, match="provenance"):
        _exercise_existing_metric_history(
            sink,
            tmp_path / "metrics.jsonl",
            [incompatible, compatible],
            operation,
        )


@pytest.mark.parametrize("operation", ["start", "append"])
def test_complete_history_rejects_conflicting_schema_valid_duplicate_payloads(
    tmp_path: Path,
    operation: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    compatible = _metric(record_id="duplicate").to_dict()
    conflicting = _metric(record_id="duplicate").to_dict()
    conflicting["metrics"] = {"policy_loss": 0.9}

    with pytest.raises(ValueError, match="record_id.*different payload"):
        _exercise_existing_metric_history(
            sink,
            tmp_path / "metrics.jsonl",
            [conflicting, compatible],
            operation,
        )


def test_complete_history_accepts_identical_duplicates_as_idempotent(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    compatible = _metric(record_id="duplicate").to_dict()
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(compatible, sort_keys=True) + "\n" + json.dumps(compatible) + "\n",
        encoding="utf-8",
    )

    assert sink.start_run(_manifest()) is False
    assert sink.log_metrics(_metric(record_id="duplicate")) is False
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def _exercise_existing_trajectory_history(
    sink: JSONLLoggingSink,
    path: Path,
    payloads: list[dict[str, object]],
    operation: str,
) -> None:
    path.write_text(
        "".join(json.dumps(payload, sort_keys=True) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    if operation == "start":
        sink.start_run(_manifest())
    else:
        record = _trajectory_record().to_dict()
        record["record_id"] = "new-trajectory-record"
        sink.log_trajectory(record)


@pytest.mark.parametrize("operation", ["start", "append"])
@pytest.mark.parametrize("corruption", ["unknown", "missing"])
def test_trajectory_history_rejects_nested_schema_corruption_hidden_by_duplicate(
    tmp_path: Path,
    operation: str,
    corruption: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    compatible = _trajectory_record().to_dict()
    malformed = deepcopy(compatible)
    trajectory = malformed["trajectory"]
    assert isinstance(trajectory, dict)
    if corruption == "unknown":
        trajectory["unknown_nested_field"] = True
    else:
        trajectory.pop("seed")

    with pytest.raises(ValueError, match="trajectory fields"):
        _exercise_existing_trajectory_history(
            sink,
            tmp_path / "trajectories.jsonl",
            [malformed, compatible],
            operation,
        )


@pytest.mark.parametrize("operation", ["start", "append"])
def test_trajectory_history_rejects_provenance_conflict_hidden_by_duplicate(
    tmp_path: Path,
    operation: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    compatible = _trajectory_record().to_dict()
    incompatible = deepcopy(compatible)
    incompatible["learner_backend"] = "incompatible-backend"

    with pytest.raises(ValueError, match="provenance"):
        _exercise_existing_trajectory_history(
            sink,
            tmp_path / "trajectories.jsonl",
            [incompatible, compatible],
            operation,
        )


@pytest.mark.parametrize("operation", ["start", "append"])
def test_trajectory_history_rejects_conflicting_schema_valid_duplicate(
    tmp_path: Path,
    operation: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    compatible = _trajectory_record().to_dict()
    conflicting = deepcopy(compatible)
    trajectory = conflicting["trajectory"]
    assert isinstance(trajectory, dict)
    trajectory["response"] = "Different observable answer"

    with pytest.raises(ValueError, match="record_id.*different payload"):
        _exercise_existing_trajectory_history(
            sink,
            tmp_path / "trajectories.jsonl",
            [conflicting, compatible],
            operation,
        )


def test_trajectory_history_normalizes_absent_or_empty_optional_evidence_as_identical(
    tmp_path: Path,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    trajectory = Trajectory(
        trajectory_id="trajectory-without-evidence",
        case_id=None,
        prompt="Question?",
        response="Answer",
        backend_name="torch_portable",
        backend_version="2.9",
        model_identifier="tiny-local",
        policy_version="policy-3",
    )
    record = TrajectoryRecord(
        record_id="optional-evidence",
        run_id="run-1",
        timestamp="2026-07-31T12:00:02Z",
        global_step=3,
        backend="torch_portable",
        actor_backend="torch_portable",
        learner_backend="torch_portable",
        policy_version="policy-3",
        trajectory=trajectory,
    ).to_dict()
    explicit_empty = deepcopy(record)
    nested = explicit_empty["trajectory"]
    assert isinstance(nested, dict)
    nested["evidence_references"] = []

    _exercise_existing_trajectory_history(
        sink,
        tmp_path / "trajectories.jsonl",
        [explicit_empty, record],
        "start",
    )

    assert sink.log_trajectory(record) is False


@pytest.mark.parametrize("operation", ["start", "append"])
def test_nonempty_jsonl_history_requires_a_final_newline(
    tmp_path: Path,
    operation: str,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    path = tmp_path / "metrics.jsonl"
    original = json.dumps(_metric(record_id="unterminated").to_dict())
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="final newline"):
        if operation == "start":
            sink.start_run(_manifest())
        else:
            sink.log_metrics(_metric(record_id="next-record"))

    assert path.read_text(encoding="utf-8") == original
