from __future__ import annotations

import json
from pathlib import Path

from mindful_trace_gepa.storage import TraceArchiveWriter, iter_jsonl


def test_sharded_manifest_creation(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = TraceArchiveWriter(trace_path, shard_threshold=5)
    for idx in range(12):
        writer.append({"idx": idx, "content": f"event-{idx}"})
    manifest_path = writer.close()

    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_events"] == 12
    assert len(manifest["shards"]) >= 3

    shard_counts = 0
    for shard in manifest["shards"]:
        shard_file = manifest_path.parent / shard["path"]
        assert shard_file.exists()
        events = list(iter_jsonl(shard_file))
        shard_counts += len(events)
    assert shard_counts == 12
