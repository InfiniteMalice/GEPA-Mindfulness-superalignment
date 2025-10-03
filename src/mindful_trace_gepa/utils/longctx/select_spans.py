"""Utilities for selecting long-context spans via lightweight retrieval."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


@dataclass
class Span:
    text: str
    metadata: dict | None = None
    embedding: Sequence[float] | None = None


def _bm25_score(query_tokens: Sequence[str], doc_tokens: Sequence[str], idf: dict[str, float], avg_len: float) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    doc_len = len(doc_tokens) or 1
    tf = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1
    for token in query_tokens:
        if token not in tf:
            continue
        numerator = tf[token] * (k1 + 1)
        denominator = tf[token] + k1 * (1 - b + b * doc_len / avg_len)
        score += idf.get(token, 0.0) * (numerator / denominator)
    return score


def _build_idf(corpus: Sequence[Sequence[str]]) -> tuple[dict[str, float], float]:
    doc_freq: dict[str, int] = {}
    for tokens in corpus:
        seen = set(tokens)
        for token in seen:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    total_docs = max(len(corpus), 1)
    idf = {
        token: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
        for token, freq in doc_freq.items()
    }
    avg_len = sum(len(tokens) for tokens in corpus) / max(total_docs, 1)
    return idf, avg_len or 1.0


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def select_top_spans(
    query: str,
    spans: Iterable[Span | dict | str],
    *,
    k: int = 3,
    method: str = "bm25",
    query_embedding: Sequence[float] | None = None,
) -> List[Span]:
    """Select the top-k spans matching a query.

    Parameters
    ----------
    query:
        The textual query describing desired content.
    spans:
        Iterable of ``Span`` objects, dictionaries with a ``text`` field, or plain strings.
    k:
        Number of spans to return.
    method:
        Retrieval scoring method. ``"bm25"`` (default) or ``"cosine"`` for embedding cosine similarity.
    query_embedding:
        Optional embedding vector for the query when using cosine similarity.
    """

    normalized: List[Span] = []
    for item in spans:
        if isinstance(item, Span):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(Span(text=item))
        else:
            text = str(item.get("text", ""))
            embedding = item.get("embedding")
            metadata = {k: v for k, v in item.items() if k not in {"text", "embedding"}}
            normalized.append(Span(text=text, metadata=metadata or None, embedding=embedding))

    if not normalized:
        return []

    if method == "cosine":
        if query_embedding is None:
            raise ValueError("query_embedding must be provided for cosine similarity retrieval")
        scored = [
            (idx, _cosine_similarity(query_embedding, span.embedding or []))
            for idx, span in enumerate(normalized)
        ]
    else:
        doc_tokens = [_tokenize(span.text) for span in normalized]
        idf, avg_len = _build_idf(doc_tokens)
        query_tokens = _tokenize(query)
        scored = [
            (idx, _bm25_score(query_tokens, doc_tokens[idx], idf, avg_len))
            for idx in range(len(normalized))
        ]

    scored.sort(key=lambda item: item[1], reverse=True)
    top_spans = [normalized[idx] for idx, score in scored[:k] if score > -1]
    return top_spans


__all__ = ["select_top_spans", "Span"]
