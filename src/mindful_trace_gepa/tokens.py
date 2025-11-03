"""Utilities for capturing token-level metadata."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class TokenTopK:
    t: str
    lp: float | None


@dataclass
class TokenRecord:
    idx: int
    token: str
    logprob: float | None
    topk: Sequence[TokenTopK]
    conf: float
    abstained: bool
    ts: str
    chunk: int
    offset: int
    ppl: float | None = None


class TokenRecorder:
    def __init__(self, *, log_topk: int = 0, sample_every: int = 1) -> None:
        self.records: List[TokenRecord] = []
        self.log_topk = max(0, log_topk)
        self.sample_every = max(1, sample_every)
        self._global_idx = 0
        self._rolling_logprob = 0.0

    def _mock_topk(self, token: str, logprob: float) -> Sequence[TokenTopK]:
        if self.log_topk <= 0:
            return []
        decay = 0.5
        topk: List[TokenTopK] = []
        for rank in range(self.log_topk):
            lp = logprob - decay * rank
            candidate = f"{token}_{rank}" if rank else token
            topk.append(TokenTopK(t=candidate, lp=lp))
        return topk

    def record_text(self, text: str, abstained: bool = False) -> None:
        tokens = text.split()
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        for token in tokens or [""]:
            logprob = -math.log1p(self._global_idx + 1)
            confidence = max(0.0, 1.0 + logprob / 5.0)
            self._global_idx += 1
            if (self._global_idx - 1) % self.sample_every != 0:
                continue
            chunk = (self._global_idx - 1) // self.sample_every
            offset = (self._global_idx - 1) % self.sample_every
            self._rolling_logprob = 0.95 * self._rolling_logprob + 0.05 * logprob
            perplexity = math.exp(-self._rolling_logprob) if self._rolling_logprob else None
            topk = self._mock_topk(token, logprob)
            self.records.append(
                TokenRecord(
                    idx=len(self.records),
                    token=token,
                    logprob=logprob,
                    topk=topk,
                    conf=confidence,
                    abstained=abstained,
                    ts=now,
                    chunk=chunk,
                    offset=offset,
                    ppl=perplexity,
                )
            )

    def extend(self, other: Iterable[TokenRecord]) -> None:
        for record in other:
            normalized = replace(record, idx=len(self.records))
            self.records.append(normalized)
            inferred_global = normalized.chunk * self.sample_every + normalized.offset + 1
            self._global_idx = max(self._global_idx, inferred_global)

    def dump_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                payload = asdict(record)
                payload["topk"] = [asdict(top) for top in record.topk]
                json.dump(payload, handle)
                handle.write("\n")


__all__ = ["TokenRecorder", "TokenRecord", "TokenTopK"]
