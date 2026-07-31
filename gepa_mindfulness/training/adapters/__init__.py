"""Streaming adapters for dataset-backed rollout requests."""

from .flat_jsonl import FlatJSONLAdapter
from .synthetic_cases import SyntheticCaseAdapter

__all__ = ["FlatJSONLAdapter", "SyntheticCaseAdapter"]
