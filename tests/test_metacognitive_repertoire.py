"""Repertoire methods declare applicability and cannot authorize their own execution."""

from dataclasses import replace

import pytest
from test_metacognitive_records import assessment
from test_metacognitive_records import sources as sources

from gepa_mindfulness.verification.metacognitive_records import FindingKind
from gepa_mindfulness.verification.metacognitive_repertoire import (
    RepertoireStrategy,
    applicable_strategies,
    get_repertoire,
)


def test_repertoire_covers_the_eight_requested_methods() -> None:
    """Each required method has an inspectable evidence and verification contract."""
    entries = {s.strategy_id: s for s in get_repertoire()}
    assert set(entries) == {
        "assumption_boundary_audit",
        "discriminative_testing",
        "evidence_verification",
        "robustness_sensitivity",
        "belief_revision",
        "clarification_abstention",
        "evidence_recovery",
        "transfer_calibration",
    }
    for strategy in entries.values():
        assert strategy.description and strategy.applicability_conditions
        assert strategy.required_evidence and strategy.expected_outputs
        assert strategy.verification_requirements and strategy.resource_implications
        assert strategy.source.startswith("https://arxiv.org/")
        assert {
            "execute_action",
            "change_reward",
            "modify_accepted_evidence",
            "grant_authority",
        } <= set(strategy.prohibited_effects)
        assert RepertoireStrategy.from_dict(strategy.to_dict()) == strategy


def test_applicability_uses_findings_without_certifying_them(sources) -> None:
    """The strategy catalogue selects methods while leaving evidence status unchanged."""
    value = assessment(sources)
    entries = applicable_strategies(value)
    assert {s.strategy_id for s in entries} == {"assumption_boundary_audit"}
    assert value.sources.evidence_state.claims[0].status == "unverified"
    assert applicable_strategies(replace(value, findings=())) == ()


def test_ambiguity_and_abstention_remain_separate_outputs(sources) -> None:
    """One method can declare two distinct actions without conflating their triggers."""
    entry = next(s for s in get_repertoire() if s.strategy_id == "clarification_abstention")
    assert {"request_clarification", "recommend_abstention"} <= set(entry.expected_outputs)
    assert FindingKind.AMBIGUOUS_INSTRUCTION in entry.when_finding_kinds
    assert FindingKind.INSUFFICIENT_EVIDENCE in entry.when_finding_kinds


def test_strategy_contract_cannot_drop_execution_prohibition() -> None:
    """A malformed entry cannot turn this method catalogue into an authority policy."""
    value = get_repertoire()[0]
    with pytest.raises(ValueError):
        replace(value, prohibited_effects=())
    wire = value.to_dict()
    wire["execute"] = True
    with pytest.raises(ValueError):
        RepertoireStrategy.from_dict(wire)
    with pytest.raises(ValueError):
        replace(value, strategy_id="")
    with pytest.raises(ValueError):
        replace(value, expected_outputs=("execute_shell",))


def test_returned_catalogue_is_detached() -> None:
    """A caller cannot poison a subsequent catalogue read through a frozen-record bypass."""
    value = get_repertoire()[0]
    object.__setattr__(value, "description", "poisoned")
    assert get_repertoire()[0].description != "poisoned"
