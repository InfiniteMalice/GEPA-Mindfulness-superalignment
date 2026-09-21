"""Matched pressure controls must separate omission from inferred motivation."""

from dataclasses import replace
from importlib import import_module

import pytest
from test_epistemic_continuity import assess, commitment, event_sequence

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind


def pressure():
    """A public later pressure observation tied to the current proposed action."""
    module = import_module("semantic_intent_robustness.motivated_forgetting")
    return module.DirectionalPressure(
        pressure_id="pressure",
        commitment_ids=("k",),
        pressure_type="reward_pressure",
        source_event_refs=("outcome-1",),
        evidence_refs=(EvidenceReference("evidence-1", EvidenceSourceKind.EXTERNAL_RECORD),),
        favored_action_id="action-2",
        omission_supports_action=True,
        support_evidence_refs=(
            EvidenceReference("evidence-2", EvidenceSourceKind.EXTERNAL_RECORD),
        ),
        support_event_refs=("prediction-2",),
        provenance=("independent-public-assessor",),
    )


@pytest.mark.parametrize(
    "active, has_pressure, expected",
    [
        ((), False, "no_signal"),
        ((), True, "possible"),
        (("k",), True, "no_signal"),
    ],
)
def test_pressure_omission_and_retention_controls(active, has_pressure, expected) -> None:
    """Neither pressure alone nor omission alone is a motivated-forgetting candidate."""
    module = import_module("semantic_intent_robustness.motivated_forgetting")
    result = module.assess_motivated_forgetting(
        continuity=assess(active=active),
        commitments=(commitment(),),
        events=event_sequence(),
        pressures=(pressure(),) if has_pressure else (),
    )
    assert result.status == expected
    assert not hasattr(result, "optimizer_score")


def test_incomplete_pressure_evidence_is_insufficient() -> None:
    """A pressure label without evidence of favored-action benefit does not establish motive."""
    module = import_module("semantic_intent_robustness.motivated_forgetting")
    result = module.assess_motivated_forgetting(
        continuity=assess(),
        commitments=(commitment(),),
        events=event_sequence(),
        pressures=(replace(pressure(), support_evidence_refs=()),),
    )
    assert result.status == "insufficient_evidence"


def test_pressure_before_commitment_does_not_establish_directional_sequence() -> None:
    """The pressure must be observed after the commitment's earlier evidence."""
    module = import_module("semantic_intent_robustness.motivated_forgetting")
    early = replace(
        pressure(),
        source_event_refs=("prediction-0",),
        evidence_refs=(EvidenceReference("evidence-0", EvidenceSourceKind.EXTERNAL_RECORD),),
    )
    result = module.assess_motivated_forgetting(
        continuity=assess(),
        commitments=(commitment(),),
        events=event_sequence(),
        pressures=(early,),
    )
    assert result.status == "insufficient_evidence"


def test_default_pipeline_does_not_execute_audit_or_adapter() -> None:
    """Installing the feature must not activate analysis or inspect supplied internal data."""
    module = import_module("semantic_intent_robustness.continuity_audit")
    from semantic_intent_robustness.modules import SemanticIntentPipeline

    assert (
        SemanticIntentPipeline().run_continuity_audit(None, config=module.ContinuityConfig())
        is None
    )


def test_assessment_cannot_be_replayed_against_another_event_window() -> None:
    """Reused event IDs in another run cannot launder a continuity assessment."""
    module = import_module("semantic_intent_robustness.motivated_forgetting")
    other = tuple(replace(e, run_id="other", conversation_id="other") for e in event_sequence())
    with pytest.raises(ValueError, match="window"):
        module.assess_motivated_forgetting(
            continuity=assess(),
            commitments=(commitment(),),
            events=other,
            pressures=(pressure(),),
        )
