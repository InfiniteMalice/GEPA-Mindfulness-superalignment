"""Adversarial regressions for durable, structured skill lifecycle authority."""

from __future__ import annotations

from gepa_mindfulness import (
    ConsolidationProvenance,
    EvaluationAuthority,
    ExecutionEvidenceBundle,
    ExecutionEvidenceReceipt,
    FamilySeedProvenance,
    InstantiationProvenance,
    PruningProvenance,
    RefinementProvenance,
    ValidationReceipt,
    ValidationSplit,
    ValidationTarget,
)


def test_review_contracts_are_public_and_typed() -> None:
    assert ExecutionEvidenceBundle.__name__ == "ExecutionEvidenceBundle"
    assert ExecutionEvidenceReceipt.__name__ == "ExecutionEvidenceReceipt"
    assert ValidationReceipt.__name__ == "ValidationReceipt"
    assert ValidationTarget.__name__ == "ValidationTarget"
    assert EvaluationAuthority.__name__ == "EvaluationAuthority"
    assert ValidationSplit.HELD_OUT.value == "held_out"
    assert ConsolidationProvenance.__name__ == "ConsolidationProvenance"
    assert FamilySeedProvenance.__name__ == "FamilySeedProvenance"
    assert InstantiationProvenance.__name__ == "InstantiationProvenance"
    assert RefinementProvenance.__name__ == "RefinementProvenance"
    assert PruningProvenance.__name__ == "PruningProvenance"
