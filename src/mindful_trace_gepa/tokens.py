"""Utilities for capturing token-level metadata."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
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


class TokenRecorder:
    def __init__(self) -> None:
        self.records: List[TokenRecord] = []

    def record_text(self, text: str, abstained: bool = False) -> None:
        tokens = text.split()
        now = datetime.utcnow().isoformat()
        for idx, token in enumerate(tokens):
            logprob = -math.log1p(idx + 1)
            confidence = max(0.0, 1.0 + logprob / 5.0)
            topk = [TokenTopK(t=token, lp=logprob)]
            self.records.append(
                TokenRecord(
                    idx=len(self.records),
                    token=token,
                    logprob=logprob,
                    topk=topk,
                    conf=confidence,
                    abstained=abstained,
                    ts=now,
                )
            )
        if not tokens:
            self.records.append(
                TokenRecord(
                    idx=len(self.records),
                    token="",
                    logprob=None,
                    topk=[],
                    conf=0.0,
                    abstained=abstained,
                    ts=now,
                )
            )

    def extend(self, other: Iterable[TokenRecord]) -> None:
        for record in other:
            record.idx = len(self.records)  # type: ignore[attr-defined]
            self.records.append(record)

    def dump_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                payload = asdict(record)
                payload["topk"] = [asdict(top) for top in record.topk]
                json.dump(payload, handle)
                handle.write("\n")


__all__ = ["TokenRecorder", "TokenRecord", "TokenTopK"]
