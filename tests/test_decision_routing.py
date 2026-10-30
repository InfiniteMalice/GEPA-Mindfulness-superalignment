"""Structured decisions cannot replace the existing evidence and authorization guard."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.improvement_records import record_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.factuality_observability.decision_routing import (
    DecisionBackend,
    DecisionReadout,
    LoopedBackendSpec,
    PublicDecisionContext,
    analyze_depth_trace,
    evaluate_decision_backend,
    route_decision,
)
from gepa_mindfulness.factuality_observability.epistemic_routing import prepare_routing
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from tests.test_epistemic_routing import ENABLED, request

REFS = (EvidenceReference("public:artifact", EvidenceSourceKind.EXTERNAL_RECORD),)
MODEL = "fixture-checkpoint-v1"


def context(req=None):
    features = prepare_routing(req or request(), ENABLED)
    return PublicDecisionContext(features.input_digest, "Check the result", "Public artifact", REFS)


def readout(packet, action=Action.ACCEPT, probability=0.9, depth=1):
    values = tuple(probability if a == action else (1 - probability) / 7 for a in Action)
    return DecisionReadout(record_digest(packet), MODEL, depth, values, 100, 0)


def backend(predict=None, kind="jev"):
    return DecisionBackend("fixture", "1", MODEL, kind, predict or readout)


def run(req=None, *, public=None, candidate=None, threshold=0.8):
    req = req or request()
    return route_decision(
        req,
        public=public or context(req),
        backend=candidate or backend(),
        confidence_threshold=threshold,
        policy=ENABLED,
    )


def test_default_off_does_not_inspect_inputs_or_call_backend():
    assert route_decision(None, public=None, backend=None, confidence_threshold=None) is None


def test_public_packet_binds_artifact_and_existing_state_without_granting_authority():
    seen = []

    def predict(packet):
        seen.append(packet.to_dict())
        return readout(packet)

    result = run(candidate=backend(predict))
    wire = result.to_dict()
    assert result.routing.action is Action.ACCEPT
    assert wire["confers_authority"] is False and wire["optimizer_input"] is False
    assert wire["training_eligibility"] == "DEVELOPMENT"
    assert seen[0]["context"]["artifact"] == "Public artifact"
    assert seen[0]["context"]["task"] == "Check the result"
    assert result.readout.packet_digest == record_digest(result.packet)
    assert result.raw_action is Action.ACCEPT
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.parametrize("values", [{"domain_risk": 0.95}, {"verification_required": True}])
def test_confident_pass_cannot_override_trusted_required_verification(values):
    req = request()
    req = replace(req, context=replace(req.context, **values))
    result = run(req)
    assert result.raw_action is Action.ACCEPT
    assert result.routing.action is Action.ROUTE_EXTERNAL
    assert result.routing.overridden is True


def test_low_confidence_uses_escalation_and_keeps_raw_prediction():
    result = run(candidate=backend(lambda p: readout(p, probability=0.6)))
    assert result.raw_action is Action.ACCEPT
    assert result.routing.action is Action.ESCALATE
    assert result.confidence_escalation is True


def test_stale_public_context_rejected_before_callback():
    public = replace(context(), state_digest="0" * 64)
    with pytest.raises(ValueError, match="state|binding"):
        run(public=public, candidate=backend(lambda p: pytest.fail("stale input reached backend")))


def test_full_live_report_capacity_precedes_callback():
    features = replace(prepare_routing(request(), ENABLED), decision_event_id="x" * 2_100_000)
    with pytest.raises(ValueError, match="limit|MiB|capacity"):
        evaluate_decision_backend(
            features,
            context(),
            backend(lambda p: pytest.fail("oversized live output reached backend")),
            confidence_threshold=0.8,
        )


@pytest.mark.parametrize("mismatch", ["packet", "model", "depth"])
def test_misbound_readout_fails_closed(mismatch):
    def predict(packet):
        result = readout(packet)
        if mismatch == "packet":
            return replace(result, packet_digest="0" * 64)
        if mismatch == "model":
            return replace(result, model_version="other-model")
        return replace(result, depth=2)

    result = run(candidate=backend(predict))
    assert result.routing.action is Action.ESCALATE
    assert result.error is not None
    assert result.raw_action is None


def test_mutation_of_detached_packet_is_detected_and_original_context_is_unchanged():
    public = context()

    def predict(packet):
        result = readout(packet)
        object.__setattr__(packet.context, "artifact", "changed artifact")
        return result

    result = run(public=public, candidate=backend(predict))
    assert public.artifact == "Public artifact"
    assert result.packet.context.artifact == "Public artifact"
    assert result.routing.action is Action.ESCALATE
    assert result.error == "mutated_packet"


@pytest.mark.parametrize("predict", [lambda p: True, lambda p: {"action": "accept"}])
def test_malformed_backend_outputs_escalate(predict):
    assert run(candidate=backend(predict)).routing.action is Action.ESCALATE


def test_callback_exception_has_no_untrusted_exception_text_in_report():
    def predict(packet):
        raise RuntimeError("UNTRUSTED_SECRET_OR_INSTRUCTION")

    result = run(candidate=backend(predict))
    assert result.routing.action is Action.ESCALATE
    assert "UNTRUSTED_SECRET_OR_INSTRUCTION" not in str(result.to_dict())


@pytest.mark.parametrize("bad", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_probability_and_threshold_validation(bad):
    packet = run().packet
    with pytest.raises(ValueError):
        replace(readout(packet), probabilities=(bad,) + (0.0,) * 7)
    with pytest.raises(ValueError):
        run(threshold=bad)


def test_invalid_vector_shape_or_sum_rejected():
    packet = run().packet
    for values in ((0.5, 0.5), (0.5,) * 8):
        with pytest.raises(ValueError):
            replace(readout(packet), probabilities=values)


def test_versioned_wire_roundtrip_and_non_authority_markers():
    result = run()
    for value in (result.packet.context, result.packet, result.readout):
        assert type(value).from_dict(value.to_dict()) == value
        with pytest.raises(ValueError):
            type(value).from_dict({**value.to_dict(), "confers_authority": True})
        with pytest.raises(ValueError):
            type(value).from_dict({**value.to_dict(), "schema_version": "unknown"})
        with pytest.raises(ValueError):
            type(value).from_dict({**value.to_dict(), "unknown_field": True})


def loop_inputs():
    packet = run().packet
    spec = LoopedBackendSpec(MODEL, "1" * 64, "native-fixture", True, True, True, 3, REFS)
    readings = (
        readout(packet, probability=0.55),
        readout(packet, probability=0.65, depth=2),
        readout(packet, probability=0.95, depth=3),
    )
    return packet, spec, readings


def test_depth_diagnostics_distinguish_agreement_from_correctness_and_executed_cost():
    packet, spec, readings = loop_inputs()
    report = analyze_depth_trace(
        packet,
        spec,
        readings,
        agreement_scores=(0.99, 0.99),
        agreement_threshold=0.98,
        confidence_threshold=0.8,
        expected_actions=(Action.CLARIFY,),
        enabled=True,
    )
    assert report["executed_depth"] == 3
    assert report["confidence_selected_depth"] == 3
    assert report["stopper_selected_depth"] == 1
    assert report["stopper_agrees_with_deepest"] is True
    assert report["stopper_label_agreement"] is False
    assert report["actual_compute_savings_established"] is False
    assert report["confers_authority"] is False


def test_depth_trace_analysis_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="enabled"):
        analyze_depth_trace(
            None,
            None,
            None,
            agreement_scores=None,
            agreement_threshold=None,
            confidence_threshold=None,
        )


@pytest.mark.parametrize(
    "field", ["native_loop_pretrained", "resumes_hidden_state", "shared_readout"]
)
def test_conventional_repeated_calls_do_not_qualify_as_native_loop(field):
    packet, spec, readings = loop_inputs()
    with pytest.raises(ValueError, match="native|compatible"):
        analyze_depth_trace(
            packet,
            replace(spec, **{field: False}),
            readings,
            agreement_scores=(0.99, 0.99),
            agreement_threshold=0.98,
            confidence_threshold=0.8,
            enabled=True,
        )


@pytest.mark.parametrize("kind", ["noncontiguous", "model", "packet", "incomplete"])
def test_depth_trace_rejects_unmatched_or_partial_readouts(kind):
    packet, spec, readings = loop_inputs()
    if kind == "noncontiguous":
        readings = (readings[0], replace(readings[1], depth=3), readings[2])
    elif kind == "model":
        readings = (replace(readings[0], model_version="other"),) + readings[1:]
    elif kind == "packet":
        readings = (replace(readings[0], packet_digest="0" * 64),) + readings[1:]
    else:
        readings = readings[:-1]
    with pytest.raises(ValueError):
        analyze_depth_trace(
            packet,
            spec,
            readings,
            agreement_scores=(0.99, 0.99),
            agreement_threshold=0.98,
            confidence_threshold=0.8,
            enabled=True,
        )
