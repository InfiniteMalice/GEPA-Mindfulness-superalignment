"""Adversarial regressions for durable, structured skill lifecycle authority."""

from __future__ import annotations

from gepa_mindfulness import (
    ConsolidationProvenance,
    ExecutionEvidenceBundle,
    ExecutionEvidenceReceipt,
    InstantiationProvenance,
    PruningProvenance,
    RefinementProvenance,
    ValidationReceipt,
    ValidationSplit,
)


def test_review_contracts_are_public_and_typed() -> None:
    assert ExecutionEvidenceBundle.__name__ == "ExecutionEvidenceBundle"
    assert ExecutionEvidenceReceipt.__name__ == "ExecutionEvidenceReceipt"
    assert ValidationReceipt.__name__ == "ValidationReceipt"
    assert ValidationSplit.HELD_OUT.value == "held_out"
    assert ConsolidationProvenance.__name__ == "ConsolidationProvenance"
    assert InstantiationProvenance.__name__ == "InstantiationProvenance"
    assert RefinementProvenance.__name__ == "RefinementProvenance"
    assert PruningProvenance.__name__ == "PruningProvenance"
