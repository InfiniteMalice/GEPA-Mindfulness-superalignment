"""JSONL storage utilities supporting streaming and sharding."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover
    zstd = None  # type: ignore


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".zst" or path.name.endswith(".jsonl.zst"):
        if zstd is None:
            raise RuntimeError("zstandard is required to read compressed traces")
        fh = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield JSON rows from the provided file without loading everything into memory."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - guard rail
                LOGGER.warning("Skipping malformed JSONL line in %s: %s", path, exc)


def load_jsonl(path: Path | str, limit: Optional[int] = None) -> List[dict[str, Any]]:
    """Load up to ``limit`` rows into memory, streaming from disk."""

    rows: List[dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        rows.append(row)
    return rows


def read_jsonl(path: Path | str) -> List[dict[str, Any]]:
    """Compatibility helper that loads the entire file."""

    return load_jsonl(path, limit=None)


@dataclass
class ShardInfo:
    path: str
    events: int
    sha256: str


class ShardedTraceWriter:
    """Write trace events to multiple shards with a manifest."""

    def __init__(
        self,
        base_dir: Path | str,
        *,
        prefix: str = "trace",
        max_events: int = 10_000,
        compress: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.prefix = prefix
        self.max_events = max_events
        self.compress = compress and zstd is not None
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.base_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._current_handle: Optional[io.TextIOBase] = None
        self._current_hash: Optional[hashlib._Hash] = None
        self._current_context: Optional[AbstractContextManager[io.TextIOBase]] = None
        self._current_events = 0
        self._total_events = 0
        self._shards: List[ShardInfo] = []
        self._index = 0

    @contextmanager
    def _open_shard(self) -> Iterator[io.TextIOBase]:
        self._index += 1
        suffix = ".jsonl"
        if self.compress:
            suffix += ".zst"
        filename = f"{self.prefix}_{self._index:05d}{suffix}"
        path = self.shard_dir / filename
        stream = None
        if self.compress:
            assert zstd is not None  # for type checkers
            compressor = zstd.ZstdCompressor(level=3)
            fh = path.open("wb")
            stream = compressor.stream_writer(fh)
            text_handle = io.TextIOWrapper(stream, encoding="utf-8")
        else:
            text_handle = path.open("w", encoding="utf-8")
        self._current_hash = hashlib.sha256()
        try:
            yield text_handle
        finally:
            text_handle.flush()
            text_handle.close()
            if stream is not None:
                stream.close()
            events = self._current_events
            if events == 0:
                path.unlink(missing_ok=True)
            else:
                sha_hex = self._current_hash.hexdigest() if self._current_hash else ""
                relative = path.relative_to(self.base_dir)
                self._shards.append(ShardInfo(path=str(relative), events=events, sha256=sha_hex))
            self._current_handle = None
            self._current_hash = None
            self._current_events = 0

    def _ensure_handle(self) -> io.TextIOBase:
        if self._current_handle is None:
            context = self._open_shard()
            handle = context.__enter__()
            self._current_handle = handle
            self._current_context = context
        return self._current_handle

    def append(self, event: dict[str, Any]) -> None:
        handle = self._ensure_handle()
        line = json.dumps(event, ensure_ascii=False)
        payload = f"{line}\n"
        handle.write(payload)
        if self._current_hash is not None:
            self._current_hash.update(payload.encode("utf-8"))
        self._current_events += 1
        self._total_events += 1
        if self._current_events >= self.max_events:
            self._close_current()

    def _close_current(self) -> None:
        if self._current_handle is None:
            return
        if self._current_context:
            self._current_context.__exit__(None, None, None)
        self._current_handle = None
        self._current_context = None

    def close(self) -> Path:
        self._close_current()
        manifest = {
            "version": 1,
            "total_events": self._total_events,
            "shards": [shard.__dict__ for shard in self._shards],
        }
        manifest_path = self.base_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def __enter__(self) -> "ShardedTraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TraceArchiveWriter:
    """Write a primary JSONL trace and optional shards simultaneously."""

    def __init__(
        self,
        trace_path: Path | str,
        *,
        shard_threshold: int = 10_000,
        compress_shards: bool = False,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.trace_path.open("w", encoding="utf-8")
        self._shard_threshold = shard_threshold
        self._buffer: List[dict[str, Any]] = []
        self._sharder: Optional[ShardedTraceWriter] = None
        self._compress = compress_shards
        self._closed = False
        self._manifest_path: Optional[Path] = None

    def _ensure_sharder(self) -> None:
        if self._sharder is not None:
            return
        self._sharder = ShardedTraceWriter(
            self.trace_path.parent,
            prefix=self.trace_path.stem,
            max_events=self._shard_threshold,
            compress=self._compress,
        )
        for row in self._buffer:
            self._sharder.append(row)
        self._buffer.clear()

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        self._handle.write(f"{line}\n")
        if self._sharder is None and len(self._buffer) >= self._shard_threshold:
            self._ensure_sharder()
        if self._sharder is not None:
            self._sharder.append(event)
        else:
            self._buffer.append(event)

    def close(self) -> Optional[Path]:
        if self._closed:
            return self._manifest_path
        manifest: Optional[Path] = None
        if self._sharder is not None:
            for row in self._buffer:
                self._sharder.append(row)
            manifest = self._sharder.close()
        self._buffer.clear()
        self._handle.close()
        self._closed = True
        self._manifest_path = manifest
        return manifest

    def __enter__(self) -> "TraceArchiveWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "iter_jsonl",
    "load_jsonl",
    "read_jsonl",
    "ShardedTraceWriter",
    "ShardInfo",
    "TraceArchiveWriter",
]
