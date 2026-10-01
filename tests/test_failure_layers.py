"""Failure layers preserve evidence scope and distinguish hypotheses from causes."""

from dataclasses import replace

import pytest
from test_epistemic_reconciliation import EVIDENCE, metadata, reconciliation, sequence

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_reconciliation import (
    EpistemicReconciliation,
    make_epistemic_reconciliation_event,
)
from gepa_mindfulness.verification.epistemic_state import MismatchStatus
from gepa_mindfulness.verification.failure_graph import (
    FailureGraph,
    FailureLocalization,
    FailureNode,
    FailureRole,
    FailureRoleEvidence,
)
from gepa_mindfulness.verification.failure_layers import (
    FailureLayer,
    LayerClaim,
    localize_failure_layers,
)
from gepa_mindfulness.verification.interfaces import (
    LocalVerificationResult,
    RelationalVerificationResult,
    VerificationEvidenceBinding,
    make_local_verification_event,
    make_relational_verification_event,
)


def graph(event=None):
    event = event or sequence()[-1]
    return FailureGraph(
        (
            FailureNode(
                "failure", event.event_id, "Observed discrepancy", event.timestamp, (EVIDENCE,)
            ),
        ),
        (),
        FailureLocalization(
            "failure",
            None,
            None,
            (),
            None,
            (FailureRoleEvidence(FailureRole.FIRST_ANOMALY, "failure", ("v",)),),
        ),
    )


def diagnose(events=None, failure_graph=None, claims=()):
    return localize_failure_layers(
        tuple(events or sequence()), failure_graph or graph(), "r", claims, enabled=True
    )


@pytest.mark.parametrize("layer", list(FailureLayer))
def test_each_layer_is_an_evidence_bound_hypothesis(layer):
    result = diagnose(claims=(LayerClaim("failure", layer, (EVIDENCE,)),))
    assert result.annotations[0].layer == layer
    assert result.annotations[0].status == "hypothesis"
    assert result.annotations[0].basis == "reported_claim"
    assert result.annotations[0].repair_target
    assert result.residuals == (("measurement", 2.0),)
    assert result.world_uncertainty == 0.6
    assert result.monitor_uncertainty is None
    assert result.to_dict()["confers_authority"] is False
    assert result.graph_digest


def test_residual_and_uncertainty_alone_do_not_prove_any_failure_layer():
    result = diagnose()
    assert result.annotations == ()
    assert result.unlocalized_failure_ids == ("failure",)


def test_explicit_mismatch_is_world_model_hypothesis():
    record = reconciliation()
    record = replace(
        record, update=replace(record.update, model_mismatch=MismatchStatus.MODEL_MISMATCH)
    )
    result = diagnose(sequence(record))
    assert [a.layer for a in result.annotations] == [FailureLayer.WORLD_MODEL]
    assert result.annotations[0].basis == "declared_model_mismatch"


def test_multiple_layers_preserve_ambiguity_and_do_not_mutate_graph():
    original = graph()
    before = original.to_dict()
    result = diagnose(
        failure_graph=original,
        claims=(
            LayerClaim("failure", FailureLayer.EXECUTION, (EVIDENCE,)),
            LayerClaim("failure", FailureLayer.REPORTING, (EVIDENCE,)),
        ),
    )
    assert len(result.annotations) == 2
    assert original.to_dict() == before
    assert original.root_cause_status is None


def test_adapter_requires_explicit_enable():
    with pytest.raises(ValueError, match="enabled"):
        localize_failure_layers(tuple(sequence()), graph(), "r")


@pytest.mark.parametrize(
    "reference",
    [
        EvidenceReference("unrecorded", EvidenceSourceKind.EXTERNAL_RECORD),
        EvidenceReference("sensor-log", EvidenceSourceKind.PRIVATE_REASONING),
    ],
)
def test_claim_cannot_launder_unrecorded_or_private_evidence(reference):
    with pytest.raises(ValueError):
        diagnose(claims=(LayerClaim("failure", FailureLayer.KNOWLEDGE_SKILL, (reference,)),))


def test_claim_requires_known_failure():
    with pytest.raises(ValueError, match="failure"):
        diagnose(claims=(LayerClaim("other", FailureLayer.ROUTING, (EVIDENCE,)),))


def test_invalid_peo_and_cross_unit_data_fail_closed():
    events = sequence()
    events[3] = replace(events[3], run_id="different")
    with pytest.raises(ValueError):
        diagnose(events)
    with pytest.raises(ValueError):
        diagnose(sequence()[1:])


def test_graph_node_must_belong_to_selected_reconciliation():
    node = replace(graph().nodes[0], event_id="unrelated")
    invalid = replace(graph(), nodes=(node,))
    with pytest.raises(ValueError, match="ancestry"):
        diagnose(failure_graph=invalid)


def test_graph_node_cannot_relabel_evidence_or_time():
    for change in (
        {"observed_at": "2026-10-01T12:00:00Z"},
        {"evidence_refs": (EvidenceReference("unknown", EvidenceSourceKind.EXTERNAL_RECORD),)},
    ):
        node = replace(graph().nodes[0], **change)
        with pytest.raises(ValueError):
            diagnose(failure_graph=replace(graph(), nodes=(node,)))


def negative_verifier(field, *, bind=True, evidence=EVIDENCE):
    bindings = (VerificationEvidenceBinding(field, (evidence,)),) if bind else ()
    if field in {
        "executed",
        "arguments_valid",
        "schema_valid",
        "authorization_valid",
        "intended_operation_observed",
        "irreversible_action_permitted",
    }:
        result = LocalVerificationResult(
            "action", False, False, False, False, False, False, (evidence,), bindings
        )
        event = make_local_verification_event(
            result, verifier_refs=("negative-log",), **metadata("negative", 4, ("o",))
        )
    else:
        result = RelationalVerificationResult(
            "action",
            False,
            False,
            "contradicted" if field == "contradiction_status" else "unknown",
            False,
            False,
            False,
            False,
            (evidence,),
            bindings,
        )
        event = make_relational_verification_event(
            result, verifier_refs=("negative-log",), **metadata("negative", 4, ("o",))
        )
    record = replace(reconciliation(), verification_event_ids=("v", "negative"))
    events = sequence(record)
    events.insert(-1, event)
    return events, event


@pytest.mark.parametrize(
    "field,layer",
    [
        ("executed", FailureLayer.EXECUTION),
        ("arguments_valid", FailureLayer.EXECUTION),
        ("schema_valid", FailureLayer.EXECUTION),
        ("intended_operation_observed", FailureLayer.EXECUTION),
        ("authorization_valid", FailureLayer.AUTHORITY),
        ("irreversible_action_permitted", FailureLayer.AUTHORITY),
        ("authorization_scope_valid", FailureLayer.AUTHORITY),
        ("provenance_intact", FailureLayer.EVIDENCE),
        ("contradiction_status", FailureLayer.EVIDENCE),
        ("claimed_outcome_supported", FailureLayer.REPORTING),
    ],
)
def test_typed_negative_findings_with_observable_bindings_localize(field, layer):
    events, event = negative_verifier(field)
    result = diagnose(events, graph(event))
    assert len(result.annotations) == 1
    assert result.annotations[0].layer == layer
    assert result.annotations[0].basis == f"verifier_finding:{field}"


def test_unbound_negative_verifier_does_not_establish_a_layer():
    events, event = negative_verifier("executed", bind=False)
    assert diagnose(events, graph(event)).annotations == ()


def test_graph_cannot_relabel_typed_private_verifier_evidence_as_observable():
    private = replace(EVIDENCE, source_kind=EvidenceSourceKind.PRIVATE_REASONING)
    events, event = negative_verifier("executed", evidence=private)
    with pytest.raises(ValueError, match="evidence"):
        diagnose(
            events, graph(event), (LayerClaim("failure", FailureLayer.EXECUTION, (EVIDENCE,)),)
        )


def test_reconciliation_cannot_launder_private_evidence_from_its_ancestry():
    private = EvidenceReference("private-only", EvidenceSourceKind.PRIVATE_REASONING)
    observable = replace(private, source_kind=EvidenceSourceKind.EXTERNAL_RECORD)
    events, _ = negative_verifier("executed", evidence=private)
    record = EpistemicReconciliation.from_dict(events[-1].payload)
    record = replace(
        record,
        update=replace(
            record.update,
            evidence_refs=record.update.evidence_refs + (observable,),
        ),
    )
    events[-1] = make_epistemic_reconciliation_event(
        record,
        event_id="r",
        timestamp=events[-1].timestamp,
    )
    failure_graph = graph()
    failure_graph = replace(
        failure_graph,
        nodes=(
            replace(
                failure_graph.nodes[0],
                evidence_refs=(observable,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="source kind"):
        diagnose(
            events,
            failure_graph,
            (LayerClaim("failure", FailureLayer.KNOWLEDGE_SKILL, (observable,)),),
        )


def test_per_binding_mismatch_is_not_hidden_by_aggregate_none():
    record = reconciliation()
    binding = replace(
        record.bindings[0],
        innovation=replace(
            record.bindings[0].innovation,
            mismatch_status=MismatchStatus.REGIME_SHIFT_SUSPECTED,
        ),
    )
    record = replace(
        record,
        bindings=(binding,),
        update=replace(record.update, model_mismatch=MismatchStatus.NONE),
    )
    assert diagnose(sequence(record)).annotations[0].layer == FailureLayer.WORLD_MODEL


@pytest.mark.parametrize("bad", [True, "routing", None])
def test_invalid_layer_is_rejected(bad):
    with pytest.raises(ValueError):
        LayerClaim("failure", bad, (EVIDENCE,))


def test_report_serialization_is_detached_and_inputs_are_not_mutated():
    events = sequence()
    original = [event.to_dict() for event in events]
    result = diagnose(events)
    data = result.to_dict()
    data["residuals"][0][1] = 900
    assert result.residuals[0][1] == 2
    assert [event.to_dict() for event in events] == original
