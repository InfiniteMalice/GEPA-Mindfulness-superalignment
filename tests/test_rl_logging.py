"""Closed-schema and append-safety tests for shared RL run logging."""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.policy_versions import PolicyVersion
from gepa_mindfulness.training.run_logging import (
    LOG_SCHEMA_VERSION,
    JSONLLoggingSink,
    MetricRecord,
    PublicationRecord,
    RunManifest,
    TrajectoryRecord,
)
from gepa_mindfulness.training.trajectory import Trajectory


class _ManagedProcess(Protocol):
    exitcode: int | None

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def kill(self) -> None: ...

    def terminate(self) -> None: ...


class _ManagedQueue(Protocol):
    def close(self) -> None: ...

    def join_thread(self) -> None: ...


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


def _hybrid_manifest() -> RunManifest:
    return replace(
        _manifest(),
        algorithm="grpo",
        backend="fake-mojo",
        actor_backend="fake-mojo",
        model="tiny-hybrid-model",
        adapter="tiny-lora",
        device_capabilities={
            "hybrid_actor_policy": {
                "adapter_identifier": "tiny-lora",
                "adapter_sha256": "a" * 64,
                "model_identifier": "tiny-hybrid-model",
                "policy_version": "1",
            }
        },
    )


def _publication(
    *,
    record_id: str,
    parent: str,
    version: str,
    global_step: int,
    checkpoint_id: str,
) -> PublicationRecord:
    return PublicationRecord(
        record_id=record_id,
        run_id="run-1",
        timestamp=f"2026-07-31T12:00:0{global_step}Z",
        global_step=global_step,
        backend="fake-mojo",
        actor_backend="fake-mojo",
        learner_backend="torch_portable",
        parent_policy_version=parent,
        policy_version=version,
        adapter_identifier="tiny-lora",
        adapter_sha256="b" * 64,
        model_identifier="tiny-hybrid-model",
        checkpoint_id=checkpoint_id,
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@contextmanager
def _isolated_logging_registries() -> Iterator[None]:
    """Restore process-global logging registries after a cache or lock test."""
    with JSONLLoggingSink._locks_guard:
        validated_streams = list(JSONLLoggingSink._validated_streams.items())
        path_locks = list(JSONLLoggingSink._path_locks.items())
        JSONLLoggingSink._validated_streams.clear()
        JSONLLoggingSink._path_locks.clear()
    try:
        yield
    finally:
        with JSONLLoggingSink._locks_guard:
            JSONLLoggingSink._validated_streams.clear()
            JSONLLoggingSink._validated_streams.update(validated_streams)
            JSONLLoggingSink._path_locks.clear()
            JSONLLoggingSink._path_locks.update(path_locks)


class _HistoryCountingSink(JSONLLoggingSink):
    historical_rows_validated = 0
    existing_records_calls = 0

    @classmethod
    def _existing_records(cls, stream, path, manifest):
        cls.existing_records_calls += 1
        records, publication_tail = super()._existing_records(stream, path, manifest)
        cls.historical_rows_validated += len(records)
        return records, publication_tail


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


def _sleeping_child(started: object) -> None:
    started.set()  # type: ignore[attr-defined]
    while True:
        time.sleep(1)


class _NoNoteRuntimeError(RuntimeError):
    add_note = None


class _FailingCleanupQueue:
    def close(self) -> None:
        raise OSError("queue close failed")

    def join_thread(self) -> None:
        raise RuntimeError("queue join_thread failed")


@contextmanager
def _multiprocess_cleanup(
    processes: Sequence[_ManagedProcess],
    *,
    queues: Sequence[_ManagedQueue] = (),
    join_timeout: float = 1.0,
) -> Iterator[None]:
    primary_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_failures: list[str] = []

        def attempt(label: str, callback: Callable[[], None]) -> None:
            try:
                callback()
            except BaseException as error:
                cleanup_failures.append(f"{label}: {type(error).__name__}: {error}")

        def is_alive(index: int, process: _ManagedProcess) -> bool:
            try:
                return process.is_alive()
            except BaseException as error:
                cleanup_failures.append(
                    f"process {index} liveness: {type(error).__name__}: {error}"
                )
                return False

        for index, process in enumerate(processes):
            attempt(
                f"process {index} initial join",
                lambda process=process: process.join(join_timeout),
            )
            if is_alive(index, process):
                attempt(f"process {index} terminate", process.terminate)
                attempt(
                    f"process {index} post-terminate join",
                    lambda process=process: process.join(join_timeout),
                )
            if is_alive(index, process):
                attempt(f"process {index} kill", process.kill)
                attempt(
                    f"process {index} final join",
                    lambda process=process: process.join(join_timeout),
                )
            if is_alive(index, process):
                cleanup_failures.append(f"process {index} remained alive after kill")

        for index, queue in enumerate(queues):
            attempt(f"queue {index} close", queue.close)
            attempt(f"queue {index} join_thread", queue.join_thread)

        if cleanup_failures:
            diagnostic = "multiprocess cleanup failures: " + "; ".join(cleanup_failures)
            if primary_error is None:
                raise RuntimeError(diagnostic)
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(diagnostic)
            else:
                cleanup_error = RuntimeError(diagnostic)
                cleanup_error.__cause__ = primary_error.__cause__
                primary_error.__cause__ = cleanup_error


def test_multiprocess_cleanup_preserves_body_error_and_reaps_children_and_queue_threads() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    process = context.Process(target=_sleeping_child, args=(started,))
    queue = context.Queue()
    queue.put("start feeder thread")
    process.start()

    with pytest.raises(RuntimeError, match="sentinel test-body failure"):
        with _multiprocess_cleanup((process,), queues=(queue,), join_timeout=0.05):
            assert started.wait(5)
            raise RuntimeError("sentinel test-body failure")

    assert not process.is_alive()
    assert process.exitcode is not None
    with pytest.raises(ValueError, match="closed"):
        queue.put("must be closed")
    feeder = queue._thread  # type: ignore[attr-defined]
    assert feeder is None or not feeder.is_alive()


def test_multiprocess_cleanup_chains_every_failure_when_primary_cannot_accept_notes() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    running = context.Process(target=_sleeping_child, args=(started,))
    unstarted = context.Process(target=_sleeping_child, args=(context.Event(),))
    queue = _FailingCleanupQueue()
    primary = _NoNoteRuntimeError("primary test-body failure")
    running.start()

    with pytest.raises(_NoNoteRuntimeError) as caught:
        with _multiprocess_cleanup(
            (running, unstarted),
            queues=(queue,),
            join_timeout=0.05,
        ):
            assert started.wait(5)
            raise primary

    assert caught.value is primary
    assert str(caught.value) == "primary test-body failure"
    diagnostic = caught.value.__cause__
    assert isinstance(diagnostic, RuntimeError)
    assert "process 1 initial join: AssertionError" in str(diagnostic)
    assert "queue 0 close: OSError: queue close failed" in str(diagnostic)
    assert "queue 0 join_thread: RuntimeError: queue join_thread failed" in str(diagnostic)
    assert not running.is_alive()
    assert unstarted.exitcode is None


def test_multiprocess_cleanup_raises_diagnostic_without_primary_error() -> None:
    context = multiprocessing.get_context("spawn")
    unstarted = context.Process(target=_sleeping_child, args=(context.Event(),))

    with pytest.raises(RuntimeError, match="process 0 initial join: AssertionError"):
        with _multiprocess_cleanup((unstarted,), join_timeout=0.01):
            pass


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


def test_schema_v1_legacy_trajectory_without_adapter_hash_is_readable(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    manifest = _manifest()
    sink.start_run(manifest)
    payload = _trajectory_record().to_dict()
    trajectory = payload["trajectory"]
    assert isinstance(trajectory, dict)
    trajectory.pop("adapter_sha256")
    (tmp_path / "trajectories.jsonl").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert sink.start_run(manifest) is False


def test_first_publication_parent_must_match_manifest_actor_policy(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_hybrid_manifest())

    with pytest.raises(ValueError, match="publication chain"):
        sink.log_publication(
            _publication(
                record_id="publication-wrong-first-parent",
                parent="2",
                version="3",
                global_step=1,
                checkpoint_id="checkpoint-00000001",
            )
        )


def test_publication_checkpoint_id_supports_canonical_steps_wider_than_eight_digits() -> None:
    record = _publication(
        record_id="publication-large-step",
        parent="1",
        version="2",
        global_step=100_000_000,
        checkpoint_id="checkpoint-100000000",
    )

    assert record.checkpoint_id == "checkpoint-100000000"
    assert record.global_step == 100_000_000


@pytest.mark.parametrize(
    "changes",
    [
        {"record_id": "publication-duplicate-version", "parent": "1", "version": "2"},
        {"record_id": "publication-gap", "parent": "3", "version": "4"},
        {"record_id": "publication-regression", "parent": "0", "version": "1"},
        {"record_id": "publication-step-regression", "global_step": 1},
        {"record_id": "publication-checkpoint-step", "checkpoint_id": "checkpoint-00000003"},
        {"record_id": "publication-checkpoint-shape", "checkpoint_id": "checkpoint-2"},
    ],
)
def test_publication_history_rejects_broken_chains(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_hybrid_manifest())
    assert sink.log_publication(
        _publication(
            record_id="publication-1",
            parent="1",
            version="2",
            global_step=1,
            checkpoint_id="checkpoint-00000001",
        )
    )
    values: dict[str, object] = {
        "record_id": "publication-2",
        "parent": "2",
        "version": "3",
        "global_step": 2,
        "checkpoint_id": "checkpoint-00000002",
    }
    values.update(changes)

    with pytest.raises(ValueError, match="publication chain|checkpoint|global_step"):
        sink.log_publication(_publication(**values))  # type: ignore[arg-type]


def test_nonzero_rank_does_not_write_manifest_or_aggregate_metrics(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=1)

    assert sink.start_run(_manifest()) is False
    assert sink.log_metrics(_metric()) is False

    assert list(tmp_path.iterdir()) == []


def test_nonzero_rank_cannot_write_response_metrics_or_trajectories(tmp_path: Path) -> None:
    rank_zero = JSONLLoggingSink(tmp_path, rank=0)
    rank_zero.start_run(_manifest())
    rank_one = JSONLLoggingSink(tmp_path, rank=1)
    record = _metric(record_id="response-1", scope="response")

    assert rank_one.log_metrics(record) is False
    assert rank_one.log_trajectory(_trajectory_record()) is False
    assert rank_zero.log_metrics(record) is True

    records = _jsonl(tmp_path / "metrics.jsonl")
    assert len(records) == 1
    assert records[0]["record_id"] == "response-1"
    assert (tmp_path / "trajectories.jsonl").read_bytes() == b""


def test_parallel_nonzero_ranks_never_append_to_canonical_stream(tmp_path: Path) -> None:
    rank_zero = JSONLLoggingSink(tmp_path, rank=0)
    rank_zero.start_run(_manifest())
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_concurrent_duplicate_writer, args=(str(tmp_path), start))
        for _ in range(8)
    ]
    with _multiprocess_cleanup(processes):
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(timeout=30)

        assert all(process.exitcode == 0 for process in processes)
    records = _jsonl(tmp_path / "metrics.jsonl")
    assert records == []


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


def test_cached_trajectory_uses_canonical_digest_without_retaining_large_payload(
    tmp_path: Path,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    payload = _trajectory_record().to_dict()
    payload["record_id"] = "large-trajectory"
    trajectory = payload["trajectory"]
    assert isinstance(trajectory, dict)
    sentinel = "large-sensitive-payload-" + "x" * 500_000
    trajectory["response"] = sentinel

    assert sink.log_trajectory(payload) is True
    assert sink.log_trajectory(payload) is False
    conflicting = deepcopy(payload)
    conflicting_trajectory = conflicting["trajectory"]
    assert isinstance(conflicting_trajectory, dict)
    conflicting_trajectory["response"] = "different response"
    with pytest.raises(ValueError, match="record_id.*different payload"):
        sink.log_trajectory(conflicting)

    path = (tmp_path / "trajectories.jsonl").resolve()
    cached_records = sink._validated_streams[path].records
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert cached_records["large-trajectory"] == hashlib.sha256(canonical).hexdigest()
    assert sentinel not in repr(sink._validated_streams[path])


def test_many_same_process_appends_do_not_revalidate_growing_history(tmp_path: Path) -> None:
    _HistoryCountingSink.historical_rows_validated = 0
    _HistoryCountingSink.existing_records_calls = 0
    first_sink = _HistoryCountingSink(tmp_path, rank=0)
    second_sink = _HistoryCountingSink(tmp_path, rank=0)
    first_sink.start_run(_manifest())

    append_count = 120
    for index in range(append_count):
        sink = first_sink if index % 2 == 0 else second_sink
        assert sink.log_metrics(_metric(record_id=f"metric-{index}")) is True

    assert _HistoryCountingSink.historical_rows_validated <= append_count
    assert len(_jsonl(tmp_path / "metrics.jsonl")) == append_count


def test_cached_appends_retain_index_identity_and_parse_history_once(tmp_path: Path) -> None:
    """Copying the growing cache on every append must fail deterministic O(1) evidence."""
    _HistoryCountingSink.existing_records_calls = 0
    sink = _HistoryCountingSink(tmp_path, rank=0)
    second_sink = _HistoryCountingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    path = (tmp_path / "metrics.jsonl").resolve()
    assert sink.log_metrics(_metric(record_id="metric-0")) is True
    cached_records = sink._validated_streams[path].records

    for index in range(1, 80):
        writer = sink if index % 2 else second_sink
        assert writer.log_metrics(_metric(record_id=f"metric-{index}")) is True

    assert sink._validated_streams[path].records is cached_records
    assert len(cached_records) == 80
    assert _HistoryCountingSink.existing_records_calls == 1


def test_fsync_failure_does_not_publish_or_mutate_cached_record_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    assert sink.log_metrics(_metric(record_id="metric-before")) is True
    path = (tmp_path / "metrics.jsonl").resolve()
    cached_records = sink._validated_streams[path].records
    cached_snapshot = dict(cached_records)
    monkeypatch.setattr(os, "fsync", lambda descriptor: (_ for _ in ()).throw(OSError("fault")))

    with pytest.raises(OSError, match="fault"):
        sink.log_metrics(_metric(record_id="metric-fsync-fault"))

    assert sink._validated_streams[path].records is cached_records
    assert cached_records == cached_snapshot


def test_external_cache_mutation_forces_reparse_and_corruption_failure(tmp_path: Path) -> None:
    """In-place caching must not bypass full validation after external stream changes."""
    _HistoryCountingSink.existing_records_calls = 0
    sink = _HistoryCountingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    assert sink.log_metrics(_metric(record_id="metric-before")) is True
    path = tmp_path / "metrics.jsonl"
    original_cache = sink._validated_streams[path.resolve()].records
    with path.open("ab") as stream:
        stream.write(b'{"record_id":"corrupt-external"}\n')

    with pytest.raises(ValueError, match="metric record fields"):
        sink.log_metrics(_metric(record_id="metric-after"))

    with pytest.raises(ValueError, match="metric record fields"):
        sink.log_metrics(_metric(record_id="metric-after-again"))

    assert _HistoryCountingSink.existing_records_calls == 3
    assert sink._validated_streams[path.resolve()].records is original_cache


def test_external_append_invalidates_cached_history(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    assert sink.log_metrics(_metric(record_id="metric-before-external-append")) is True
    path = tmp_path / "metrics.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"record_id":"malformed-external-record"}\n')

    with pytest.raises(ValueError, match="metric record fields"):
        sink.log_metrics(_metric(record_id="metric-after-external-append"))


def test_valid_external_append_publishes_new_cache_only_after_complete_rescan(
    tmp_path: Path,
) -> None:
    _HistoryCountingSink.existing_records_calls = 0
    sink = _HistoryCountingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    assert sink.log_metrics(_metric(record_id="metric-before")) is True
    path = (tmp_path / "metrics.jsonl").resolve()
    old_records = sink._validated_streams[path].records
    external = _metric(record_id="metric-external").to_dict()
    with path.open("ab") as stream:
        stream.write(json.dumps(external, separators=(",", ":")).encode() + b"\n")

    assert sink.log_metrics(_metric(record_id="metric-after")) is True

    new_records = sink._validated_streams[path].records
    assert new_records is not old_records
    assert set(new_records) == {"metric-before", "metric-external", "metric-after"}
    assert set(old_records) == {"metric-before"}
    assert _HistoryCountingSink.existing_records_calls == 2


def test_external_publication_tail_is_used_for_next_transition(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_hybrid_manifest())
    first = _publication(
        record_id="publication-1",
        parent="1",
        version="2",
        global_step=1,
        checkpoint_id="checkpoint-00000001",
    )
    assert sink.log_publication(first) is True
    path = tmp_path / "publications.jsonl"
    external = _publication(
        record_id="publication-2",
        parent="2",
        version="3",
        global_step=2,
        checkpoint_id="checkpoint-00000002",
    )
    with path.open("ab") as stream:
        stream.write(json.dumps(external.to_dict(), separators=(",", ":")).encode() + b"\n")

    assert (
        sink.log_publication(
            _publication(
                record_id="publication-3",
                parent="3",
                version="4",
                global_step=3,
                checkpoint_id="checkpoint-00000003",
            )
        )
        is True
    )


def test_cached_publication_tail_retains_only_immutable_transition_fields(
    tmp_path: Path,
) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_hybrid_manifest())
    payload = _publication(
        record_id="publication-large-payload",
        parent="1",
        version="2",
        global_step=1,
        checkpoint_id="checkpoint-00000001",
    ).to_dict()
    sentinel = "large-publication-payload-" + "x" * 500_000
    payload["timestamp"] = sentinel

    assert sink.log_publication(payload) is True

    path = (tmp_path / "publications.jsonl").resolve()
    state = sink._validated_streams[path]
    assert state.publication_tail == (PolicyVersion.from_json("2"), 1)
    assert sentinel not in repr(state)


def test_validated_stream_registry_never_exceeds_configured_limit(tmp_path: Path) -> None:
    configured_limit = getattr(JSONLLoggingSink, "_VALIDATED_STREAM_LIMIT", 64)
    with _isolated_logging_registries():
        for index in range(configured_limit + 5):
            directory = tmp_path / f"run-{index}"
            sink = JSONLLoggingSink(directory, rank=0)
            sink.start_run(_manifest())
            assert sink.log_metrics(_metric(record_id=f"metric-{index}")) is True

        with JSONLLoggingSink._locks_guard:
            assert len(JSONLLoggingSink._validated_streams) <= configured_limit


def test_validated_stream_lru_refreshes_hits_and_skips_active_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(JSONLLoggingSink, "_VALIDATED_STREAM_LIMIT", 3)
    with _isolated_logging_registries():
        sinks: list[JSONLLoggingSink] = []
        paths: list[Path] = []
        for index in range(3):
            sink = JSONLLoggingSink(tmp_path / f"lru-{index}", rank=0)
            sink.start_run(_manifest())
            assert sink.log_metrics(_metric(record_id=f"lru-metric-{index}")) is True
            sinks.append(sink)
            paths.append((sink.directory / "metrics.jsonl").resolve())

        first_records = JSONLLoggingSink._validated_streams[paths[0]].records
        assert sinks[0].log_metrics(_metric(record_id="lru-metric-0")) is False
        newest = JSONLLoggingSink(tmp_path / "lru-3", rank=0)
        newest.start_run(_manifest())
        assert newest.log_metrics(_metric(record_id="lru-metric-3")) is True
        with JSONLLoggingSink._locks_guard:
            assert paths[0] in JSONLLoggingSink._validated_streams
            assert paths[1] not in JSONLLoggingSink._validated_streams
            assert JSONLLoggingSink._validated_streams[paths[0]].records is first_records

        with JSONLLoggingSink._locked_path(paths[2]):
            fourth = JSONLLoggingSink(tmp_path / "lru-4", rank=0)
            fourth.start_run(_manifest())
            assert fourth.log_metrics(_metric(record_id="lru-metric-4")) is True
            with JSONLLoggingSink._locks_guard:
                assert paths[2] in JSONLLoggingSink._validated_streams
        fifth = JSONLLoggingSink(tmp_path / "lru-5", rank=0)
        fifth.start_run(_manifest())
        assert fifth.log_metrics(_metric(record_id="lru-metric-5")) is True
        with JSONLLoggingSink._locks_guard:
            assert paths[2] not in JSONLLoggingSink._validated_streams


def test_all_active_cache_saturation_retains_new_state_then_prunes_inactive_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain a new state while all cached paths are active, then prune after release."""
    monkeypatch.setattr(JSONLLoggingSink, "_VALIDATED_STREAM_LIMIT", 3)
    with _isolated_logging_registries():
        for index in range(3):
            seed = JSONLLoggingSink(tmp_path / f"seed-{index}", rank=0)
            seed.start_run(_manifest())
            assert seed.log_metrics(_metric(record_id=f"seed-metric-{index}")) is True
        with JSONLLoggingSink._locks_guard:
            cached_paths = tuple(JSONLLoggingSink._validated_streams)
            assert len(cached_paths) == 3

        _HistoryCountingSink.existing_records_calls = 0
        with ExitStack() as active_paths:
            for path in cached_paths:
                active_paths.enter_context(JSONLLoggingSink._locked_path(path))
            saturated = _HistoryCountingSink(tmp_path / "saturated", rank=0)
            saturated.start_run(_manifest())
            record = _metric(record_id="saturated-metric")
            assert saturated.log_metrics(record) is True
            calls_after_append = _HistoryCountingSink.existing_records_calls
            assert saturated.log_metrics(record) is False
            assert _HistoryCountingSink.existing_records_calls == calls_after_append
            with JSONLLoggingSink._locks_guard:
                assert len(JSONLLoggingSink._validated_streams) == 4

        pruning = JSONLLoggingSink(tmp_path / "pruning", rank=0)
        pruning.start_run(_manifest())
        assert pruning.log_metrics(_metric(record_id="pruning-metric")) is True
        with JSONLLoggingSink._locks_guard:
            assert len(JSONLLoggingSink._validated_streams) == 3


def test_path_lock_registry_reuses_active_lock_and_releases_inactive_paths(
    tmp_path: Path,
) -> None:
    with _isolated_logging_registries():
        shared_path = tmp_path / "shared.jsonl"
        first_lock = JSONLLoggingSink._path_lock(shared_path)
        with first_lock:
            second_lock = JSONLLoggingSink._path_lock(shared_path)
            assert second_lock is first_lock
        del second_lock
        del first_lock

        for index in range(200):
            lock = JSONLLoggingSink._path_lock(tmp_path / f"inactive-{index}.jsonl")
            del lock
        gc.collect()

        with JSONLLoggingSink._locks_guard:
            assert len(JSONLLoggingSink._path_locks) == 0


def test_in_place_mutation_invalidates_cached_history(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    assert sink.log_metrics(_metric(record_id="metric-before-in-place-mutation")) is True
    path = tmp_path / "metrics.jsonl"
    before = path.stat()
    content = path.read_bytes()
    mutated = content.replace(
        b'"backend":"torch_portable"',
        b'"backend":"other_backendx"',
        1,
    )
    assert len(mutated) == len(content)
    path.write_bytes(mutated)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    with pytest.raises(ValueError, match="provenance"):
        sink.log_metrics(_metric(record_id="metric-after-in-place-mutation"))


def test_file_replacement_invalidates_cached_history(tmp_path: Path) -> None:
    sink = JSONLLoggingSink(tmp_path, rank=0)
    sink.start_run(_manifest())
    assert sink.log_metrics(_metric(record_id="metric-before-replacement")) is True
    path = tmp_path / "metrics.jsonl"
    before = path.stat()
    content = path.read_bytes()
    replacement_content = content.replace(
        b'"backend":"torch_portable"',
        b'"backend":"other_backendx"',
        1,
    )
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(replacement_content)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)

    with pytest.raises(ValueError, match="provenance"):
        sink.log_metrics(_metric(record_id="metric-after-replacement"))


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
    with _multiprocess_cleanup(processes, queues=(results,)):
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(timeout=30)

        assert all(process.exitcode == 0 for process in processes)
        outcomes = [results.get(timeout=5) for _ in processes]
        assert sorted(outcome.split(":", 1)[0] for _, outcome in outcomes) == [
            "error",
            "started",
        ]
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
