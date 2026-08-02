"""Streaming adapters for dataset-backed rollout requests."""

from .flat_jsonl import FlatJSONLAdapter
from .pair_records import validate_pair_record
from .synthetic_cases import SyntheticCaseAdapter

__all__ = ["FlatJSONLAdapter", "SyntheticCaseAdapter", "validate_pair_record"]
