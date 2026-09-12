from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Callable

import pytest

from evaluation.experimental_records import (
    DiagnosticStatus,
    ExperimentalMaturity,
    HypothesisSet,
    InformationGainQuestion,
    MechanisticAuditReference,
    OrchestrationScope,
    OrchestrationScopeDeclaration,
    TopologyProposal,
    experimental_record_from_dict,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind


def _references() -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference("trace:event-1", EvidenceSourceKind.EXTERNAL_RECORD),
        EvidenceReference("verifier:result-1", EvidenceSourceKind.OBSERVABLE_OUTPUT),
    )


def _records() -> tuple[object, ...]:
    common: dict[str, Any] = {
        "source_case_id": 14,
        "uncertainty": 0.35,
        "provenance_refs": _references(),
        "maturity": ExperimentalMaturity.EXPERIMENTAL,
        "diagnostic_status": DiagnosticStatus.DIAGNOSTIC,
    }
    return (
        HypothesisSet(
            record_id="hypotheses-1",
            feature_flag="competing_hypotheses",
            hypotheses=("ambiguous request", "missing authorization"),
            **common,
        ),
        InformationGainQuestion(
            record_id="question-1",
            feature_flag="expected_information_gain_inquiry",
            question="Which destination is authorized?",
            expected_information_gain=0.7,
            **common,
        ),
        TopologyProposal(
            record_id="topology-1",
            feature_flag="adaptive_small_multi_agent_topology",
            agent_roles=("planner", "verifier"),
            directed_edges=(("planner", "verifier"),),
            **common,
        ),
        OrchestrationScopeDeclaration(
            record_id="scope-1",
            feature_flag="declarative_orchestration_scope",
            scope=OrchestrationScope.FOCUS,
            declaration="Limit coordination to the selected evaluation cell.",
            **common,
        ),
        MechanisticAuditReference(
            record_id="audit-1",
            feature_flag="mechanistic_circuit_audit",
            audit_refs=_references(),
            observation="Feature activation correlates with the evaluated behavior.",
            **common,
        ),
    )


def test_every_record_has_bounded_diagnostic_metadata_and_exact_flag() -> None:
    records = _records()

    assert tuple(record.feature_flag for record in records) == (
        "competing_hypotheses",
        "expected_information_gain_inquiry",
        "adaptive_small_multi_agent_topology",
        "declarative_orchestration_scope",
        "mechanistic_circuit_audit",
    )
    for record in records:
        assert record.source_case_id == 14
        assert record.uncertainty == 0.35
        assert record.provenance_refs == _references()
        assert record.maturity is ExperimentalMaturity.EXPERIMENTAL
        assert record.diagnostic_status is DiagnosticStatus.DIAGNOSTIC
        assert not hasattr(record, "reward")
        assert not hasattr(record, "authority_grant")
        assert not hasattr(record, "execute_action")


def test_record_json_round_trips_preserve_closed_concrete_types() -> None:
    records = _records()

    restored = tuple(experimental_record_from_dict(record.to_dict()) for record in records)

    assert restored == records
    assert tuple(type(item) for item in restored) == tuple(type(item) for item in records)
    assert tuple(item.to_dict()["record_type"] for item in restored) == (
        "hypothesis_set",
        "information_gain_question",
        "topology_proposal",
        "orchestration_scope_declaration",
        "mechanistic_audit_reference",
    )


@pytest.mark.parametrize(
    ("record_index", "mutation", "message"),
    [
        (0, lambda payload: payload.update(source_case_id=18), "source_case_id"),
        (0, lambda payload: payload.update(uncertainty=-0.1), "uncertainty"),
        (
            0,
            lambda payload: payload.update(feature_flag="mechanistic_circuit_audit"),
            "feature_flag",
        ),
        (0, lambda payload: payload.update(maturity="stable"), "maturity"),
        (0, lambda payload: payload.update(diagnostic_status="authoritative"), "diagnostic_status"),
        (0, lambda payload: payload.update(hypotheses=["only one"]), "at least two"),
        (1, lambda payload: payload.update(expected_information_gain=float("inf")), "finite"),
        (2, lambda payload: payload.update(agent_roles=["solo"]), "two to five"),
        (2, lambda payload: payload.update(directed_edges=[["planner", "unknown"]]), "known roles"),
        (3, lambda payload: payload.update(scope="workspace"), "scope"),
        (4, lambda payload: payload.update(audit_refs=[]), "audit_refs"),
    ],
)
def test_records_reject_invalid_identity_uncertainty_and_payloads(
    record_index: int,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    payload = _records()[record_index].to_dict()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        experimental_record_from_dict(payload)


@pytest.mark.parametrize(
    "prohibited_field",
    [
        "execute_action",
        "authority_grant",
        "canonical_case_ids",
        "reward",
        "reward_component",
        "optimizer_fitness",
    ],
)
def test_deserialization_rejects_prohibited_effect_and_reward_fields(
    prohibited_field: str,
) -> None:
    payload = _records()[0].to_dict()
    payload[prohibited_field] = True

    with pytest.raises(ValueError, match="unknown fields"):
        experimental_record_from_dict(payload)


def test_records_are_frozen() -> None:
    record = _records()[0]

    with pytest.raises(FrozenInstanceError):
        record.uncertainty = 0.0  # type: ignore[attr-defined,misc]
