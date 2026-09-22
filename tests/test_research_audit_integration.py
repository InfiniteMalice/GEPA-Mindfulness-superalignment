"""Research overlays compose through references without expanding V5 authority."""

from dataclasses import replace

import pytest

from evaluation.cases.registry import load_case_manifest, load_stripe_registry
from evaluation.research_audits import (
    ResearchAuditReferences,
    bind_research_audits,
    make_research_event,
)
from evaluation.serialization_roundtrip import (
    EquivalenceResult,
    EquivalenceStatus,
    Expression,
    FaultStatus,
    JsonTreeCodec,
    PropositionalTreeVerifier,
    StructuredSource,
    audit_roundtrip,
)
from evaluation.v5_records import V5EvaluationRecord
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.formal_reasoning import (
    FormalClaim,
    PremiseGrounding,
    PremiseGroundingStatus,
    audit_formal_claim,
)
from gepa_mindfulness.verification.runtime_governance import AuthorityGrantRegistry
from mindful_trace_gepa.logging_schema import StructuredEventType, make_event_envelope
from semantic_intent_robustness.internal_state_trajectory import MeasurementStatus, SoTStateSnapshot
from semantic_intent_robustness.latent_language_transition import (
    PublicDelta,
    PublicMeasurementOrigin,
    audit_latent_language_transition,
)
from test_v5_failure_atlas import failed, failure_events


def test_existing_v5_payload_and_seventeen_cases_stay_stable():
    record = failed()
    payload = record.to_dict()
    bundle = bind_research_audits(
        record,
        failure_events(),
        evaluation_ref="evaluation:1",
        trace_id="trace:1",
        semantic_equivalence=(
            "equivalence:1",
            EquivalenceResult(EquivalenceStatus.EXACT_EQUIVALENCE, "independent:1", ("raw:1",)),
        ),
        judgment_changed=True,
    )
    assert V5EvaluationRecord.from_dict(payload).to_dict() == payload == record.to_dict()
    assert tuple(c.id for c in load_case_manifest().cases) == tuple(range(1, 18))
    assert bundle.attributions == (
        "MEANING_PRESERVED_JUDGMENT_CHANGED",
        "FINAL_OUTPUT_FAILURE",
    )
    assert bundle.references.semantic_equivalence_ref == "equivalence:1"
    assert "serialized_text" not in bundle.to_event().payload
    assert bundle.to_event().model_version == record.system.model_version


def test_unknown_attribution_stays_unknown_even_when_outcome_failed():
    bundle = bind_research_audits(
        failed(),
        failure_events(),
        evaluation_ref="evaluation:1",
        trace_id="trace:1",
    )
    assert bundle.attributions == ("FINAL_OUTPUT_FAILURE", "UNKNOWN")


def test_communication_fidelity_is_an_orthogonal_subtype():
    stripes = load_stripe_registry().stripes
    stripe = next(s for s in stripes if s.id == "PARAPHRASE")
    assert "COMMUNICATION_FIDELITY" in stripe.allowed_subtypes
    assert len(stripes) == 11


def test_audit_reference_cannot_replace_external_v5_verification():
    with pytest.raises(ValueError):
        bind_research_audits(failed(), (), evaluation_ref="eval:1", trace_id="trace:1")


def test_research_event_requires_context_and_is_deeply_immutable():
    event = make_research_event(
        StructuredEventType.SEMANTIC_STRATEGY_CANDIDATE,
        "strategy:1",
        run_id="run:1",
        trace_id="trace:1",
        evaluation_ref="eval:1",
        model_version="model:1",
        harness_version="harness:1",
        parent_event_ids=("event:1",),
        evidence_refs=("raw:1",),
        provenance_refs=("source:1",),
    )
    with pytest.raises(TypeError):
        event.payload["record_ref"] = "changed"
    with pytest.raises(ValueError):
        replace(event, run_id=None)
    with pytest.raises(ValueError):
        replace(event, evidence_refs=())
    assert event.event_type != StructuredEventType.VERIFICATION_RESULT.value


def bundle(**changes):
    return bind_research_audits(
        failed(), failure_events(), evaluation_ref="evaluation:1", trace_id="trace:1", **changes
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"attributions": ["UNKNOWN"]},
        {"attributions": ("CAUSATION_PROVEN",)},
        {"attributions": ()},
        {"attributions": ("UNKNOWN", "UNKNOWN")},
        {"references": {}},
        {"evidence_refs": ["raw:mutable"]},
        {"evidence_refs": ({"nested": []},)},
        {"evidence_refs": ()},
        {"parent_event_ids": ()},
        {"parent_event_ids": tuple(f"parent:{i}" for i in range(129))},
        {"record_digest": "not-a-digest"},
        {"run_id": ""},
        {"strategy_generation": float("nan")},
        {"strategy_id": "strategy:1", "strategy_generation": True},
        {"strategy_id": "strategy:1", "strategy_generation": -1},
        {"strategy_id": "strategy:1"},
        {"strategy_parent_ids": ("parent:1",)},
        {
            "strategy_id": "strategy:1",
            "strategy_generation": 1,
            "strategy_parent_ids": ("strategy:1",),
        },
        {"attributions": ("INVALID_PUBLIC_INFERENCE",)},
        {
            "references": ResearchAuditReferences(formal_audit_ref="formal:1"),
            "attributions": ("UNKNOWN", "INVALID_PUBLIC_INFERENCE"),
        },
    ],
)
def test_bundle_rejects_mutable_malformed_or_incoherent_construction(changes):
    with pytest.raises(ValueError):
        replace(bundle(), **changes)


@pytest.mark.parametrize(
    "findings",
    [
        ("MEANING_CHANGED", "MEANING_PRESERVED_JUDGMENT_CHANGED"),
        ("SERIALIZATION_FAULT", "MEANING_PRESERVED_JUDGMENT_CHANGED"),
        ("COMMUNICATION_FAILURE_UNATTRIBUTED", "EXTRACTION_FAULT"),
        ("INTERNAL_STATE_DRIFT", "STABLE_MEASURED_STATE_OUTPUT_DIVERGED"),
        ("LATENT_LANGUAGE_DECOUPLING",),
    ],
)
def test_bundle_rejects_contradictory_findings_for_single_artifact_pair(findings):
    refs = ResearchAuditReferences("equivalence:1", "roundtrip:1", None, "transition:1")
    with pytest.raises(ValueError):
        replace(bundle(), references=refs, attributions=findings)


def test_unknown_roundtrip_does_not_mask_confirmed_semantic_preservation():
    class UnknownVerifier:
        def verify(self, source, reconstructed):
            return EquivalenceResult(EquivalenceStatus.UNKNOWN, "unknown-verifier", ())

    source = StructuredSource("source:1", Expression("atom", atom="scope"), ("raw:source",))
    result = audit_roundtrip(
        source, JsonTreeCodec(), JsonTreeCodec(), UnknownVerifier(), enabled=True
    )
    assert result.roundtrip_failure is None
    record = bundle(
        semantic_equivalence=(
            "equivalence:1",
            EquivalenceResult(EquivalenceStatus.EXACT_EQUIVALENCE, "exact-verifier", ()),
        ),
        roundtrip=("roundtrip:1", result),
        judgment_changed=True,
    )
    assert "MEANING_PRESERVED_JUDGMENT_CHANGED" in record.attributions


def test_total_channel_failure_blocks_preserved_semantics_without_stage_blame():
    class ChangedExtractor:
        def extract(self, text):
            return Expression("atom", atom="other")

    source = StructuredSource("source:1", Expression("atom", atom="scope"), ("raw:source",))
    result = audit_roundtrip(
        source, JsonTreeCodec(), ChangedExtractor(), PropositionalTreeVerifier(), enabled=True
    )
    assert result.serialization_fault is FaultStatus.UNKNOWN
    assert result.extraction_fault is FaultStatus.UNKNOWN
    record = bundle(
        semantic_equivalence=(
            "equivalence:1",
            EquivalenceResult(EquivalenceStatus.EXACT_EQUIVALENCE, "exact-verifier", ()),
        ),
        roundtrip=("roundtrip:1", result),
        judgment_changed=True,
    )
    assert record.attributions == ("COMMUNICATION_FAILURE_UNATTRIBUTED", "FINAL_OUTPUT_FAILURE")


def transition(kind="measured", enabled=True):
    before = SoTStateSnapshot(
        "state:before",
        "conversation:1",
        0,
        "adapter:1",
        "model:1",
        "backend:1",
        ("layer:1",),
        "features:v1",
        ("raw:state:before",),
        MeasurementStatus.MEASURED_INTERNAL,
        "observed",
        "internal",
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if kind == "proxy":
        before = replace(
            before, measurement_status=MeasurementStatus.TRANSCRIPT_PROXY, source_kind="transcript"
        )
    after = replace(
        before,
        snapshot_id="state:after",
        turn_index=1,
        provenance=("raw:state:after",),
        local_organization=1.0,
        progress_magnitude=1.0,
        directional_consistency=1.0,
        predictive_uncertainty=1.0,
    )
    if kind == "unavailable":
        after = None
    output = PublicDelta(
        0.0,
        True,
        PublicMeasurementOrigin.OBSERVED,
        "output:v1",
        "unit:v1",
        ("raw:output:before", "raw:output:after"),
        ("public:output",),
    )
    action = replace(
        output,
        endpoint_refs=("raw:action:before", "raw:action:after"),
        provenance=("public:action",),
    )
    return audit_latent_language_transition(
        assessment_id="transition:1",
        before=before,
        after=after,
        output=output,
        action=action,
        provenance=("paired:1",),
        enabled=enabled,
    )


@pytest.mark.parametrize(
    "kind,enabled", [("proxy", True), ("unavailable", True), ("measured", False)]
)
def test_proxy_unavailable_and_disabled_transition_cannot_claim_measured_drift(kind, enabled):
    result = bundle(transition=("transition:1", transition(kind, enabled)))
    assert result.attributions == ("FINAL_OUTPUT_FAILURE", "UNKNOWN")


def test_latent_attribution_retains_snapshot_and_public_raw_references():
    result = bundle(transition=("transition:1", transition()))
    assert result.attributions == (
        "INTERNAL_STATE_DRIFT",
        "LATENT_LANGUAGE_DECOUPLING",
        "FINAL_OUTPUT_FAILURE",
    )
    assert {
        "raw:state:before",
        "raw:state:after",
        "raw:output:before",
        "raw:output:after",
        "raw:action:before",
        "raw:action:after",
        "public:output",
        "public:action",
        "paired:1",
    } <= set(result.to_event().evidence_refs)


def test_invalid_inference_and_unsupported_premises_remain_separate_and_provenance_complete():
    def ref(name):
        return EvidenceReference(name, EvidenceSourceKind.EXTERNAL_RECORD)

    claim = FormalClaim(
        "claim:1",
        ("p", "r"),
        "q",
        EvidenceReference("raw:public-claim", EvidenceSourceKind.OBSERVABLE_OUTPUT),
        evidence_refs=(ref("raw:claim-evidence"),),
        provenance_refs=(ref("raw:claim-provenance"),),
    )
    logic = audit_formal_claim(
        claim,
        audit_id="formal:1",
        enabled=True,
        premise_grounding=(
            PremiseGrounding("p", PremiseGroundingStatus.UNSUPPORTED),
            PremiseGrounding(
                "r",
                PremiseGroundingStatus.GROUNDED,
                (ref("raw:grounding"),),
                (ref("verifier:grounding"),),
            ),
        ),
    )
    result = bundle(formal=("formal:1", logic))
    assert result.attributions == (
        "INVALID_PUBLIC_INFERENCE",
        "PREMISES_UNSUPPORTED",
        "FINAL_OUTPUT_FAILURE",
    )
    assert {
        "raw:public-claim",
        "raw:claim-evidence",
        "raw:claim-provenance",
        "raw:grounding",
        "verifier:grounding",
        "verifier:bounded-propositional-v1",
    } <= set(result.to_event().evidence_refs)


def test_diagnostic_event_cannot_replace_v5_verification_or_runtime_authority():
    event = bundle().to_event()
    with pytest.raises(ValueError):
        AuthorityGrantRegistry.enroll((event,))
    required = next(item for item in failure_events() if item.event_type == "verification_result")
    substituted = tuple(
        replace(event, event_id=item.event_id) if item.event_id == required.event_id else item
        for item in failure_events()
    )
    with pytest.raises(ValueError):
        bind_research_audits(
            failed(), substituted, evaluation_ref="evaluation:1", trace_id="trace:1"
        )


def test_research_payload_detaches_nested_inputs_and_serialized_output():
    provenance = ["source:1"]
    nested = {"references": ["audit:1"]}
    event = make_event_envelope(
        StructuredEventType.RESEARCH_FAILURE_ATTRIBUTION,
        {
            "record_ref": "bundle:1",
            "evaluation_ref": "evaluation:1",
            "provenance_refs": provenance,
            "nested": nested,
        },
        run_id="run:1",
        trace_id="trace:1",
        model_version="model:1",
        harness_version="harness:1",
        parent_event_ids=("parent:1",),
        evidence_refs=("raw:1",),
    )
    provenance.clear()
    nested["references"].append("audit:mutated")
    serialized = event.to_dict()
    serialized["payload"]["nested"]["references"].append("audit:serialized-mutation")
    assert event.to_dict()["payload"]["provenance_refs"] == ["source:1"]
    assert event.to_dict()["payload"]["nested"]["references"] == ["audit:1"]


@pytest.mark.parametrize("payload", [[], None, "not a mapping"])
def test_research_logging_rejects_nonmapping_payload_with_value_error(payload):
    with pytest.raises(ValueError, match="payload"):
        make_event_envelope(
            StructuredEventType.RESEARCH_FAILURE_ATTRIBUTION,
            payload,
            run_id="run:1",
            trace_id="trace:1",
            model_version="model:1",
            harness_version="harness:1",
            parent_event_ids=("parent:1",),
            evidence_refs=("raw:1",),
        )


def test_evidence_aggregation_rejects_overflow_instead_of_dropping_raw_references():
    result = EquivalenceResult(
        EquivalenceStatus.EXACT_EQUIVALENCE, "exact-verifier", tuple(f"raw:{i}" for i in range(128))
    )
    with pytest.raises(ValueError, match="128"):
        bundle(semantic_equivalence=("equivalence:1", result))


@pytest.mark.parametrize("field", ["parent_event_ids", "evidence_refs", "provenance_refs"])
def test_research_event_helper_bounds_each_reference_collection(field):
    args = dict(
        run_id="run:1",
        trace_id="trace:1",
        evaluation_ref="eval:1",
        model_version="model:1",
        harness_version="harness:1",
        parent_event_ids=("parent:1",),
        evidence_refs=("raw:1",),
        provenance_refs=("source:1",),
    )
    args[field] = tuple(f"ref:{i}" for i in range(129))
    with pytest.raises(ValueError, match="128"):
        make_research_event(StructuredEventType.SEMANTIC_STRATEGY_CANDIDATE, "strategy:1", **args)
