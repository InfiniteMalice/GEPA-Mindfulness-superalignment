"""Improvement input boundaries must reject ambiguous or mutable provenance."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_records import (
    AttemptEvent,
    AttemptJournal,
    AuthenticationRequest,
    CandidateSpec,
    CostEntry,
    DatasetCase,
    DatasetManifest,
    DiagnosticEvidence,
    EvaluationSlot,
    ExposureRecord,
    FinalTestAuthorization,
    ImprovementProtocol,
    MetricSpec,
    ResamplingPolicy,
    SystemConfig,
    record_digest,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract

REF = EvidenceReference("host-log", EvidenceSourceKind.EXTERNAL_RECORD)
EVALUATOR = TrustedEvaluatorContract("external", "1", "rubric-v1")
DIGEST = "a" * 64


def case(case_id="case", purpose="optimizer_selection", **kwargs):
    return DatasetCase(
        case_id,
        kwargs.pop("content_digest", content_digest(case_id)),
        kwargs.pop("family_id", case_id),
        purpose,
        **kwargs,
    )


def candidate(candidate_id="candidate", **kwargs):
    return CandidateSpec(
        candidate_id,
        SystemConfig("baseline", "harness", DIGEST),
        SystemConfig(candidate_id, "harness", DIGEST),
        DIGEST,
        0,
        **kwargs,
    )


def protocol(cases=None, candidates=None, slots=()):
    manifest = DatasetManifest("dataset", cases or (case(),), (REF,))
    spec = ImprovementProtocol(
        "protocol",
        record_digest(manifest),
        candidates or (candidate(),),
        (MetricSpec("accuracy", "evaluation-ladder-v1", "representation_accuracy", DIGEST),),
        slots,
        EVALUATOR,
        (REF,),
        ResamplingPolicy(seed=7),
    )
    return spec, manifest


def test_round_trips_preserve_exact_identities_and_detach_exports():
    spec, manifest = protocol()
    slot = EvaluationSlot("slot", "candidate", "case", "accuracy", "baseline", 0, DIGEST, "base")
    event = AttemptEvent("proposal", 0, "candidate", "round", "proposal")
    records = (
        manifest,
        spec,
        slot,
        event,
        AttemptJournal("journal", record_digest(spec), (event,), (REF,)),
        CostEntry(0.2, "USD", "billed", "USD"),
        ExposureRecord(
            1, DIGEST, "independent_audit", ("candidate",), (), "evaluation_only", (REF,)
        ),
        FinalTestAuthorization("candidate", DIGEST, DIGEST, DIGEST, "event", (REF,)),
        AuthenticationRequest("protocol", DIGEST, EVALUATOR, (REF,)),
    )
    for original in records:
        restored = type(original).from_dict(original.to_dict())
        assert record_digest(restored) == record_digest(original)
    export = spec.to_dict()
    export["candidates"][0]["baseline"]["model_version"] = "modified"
    assert spec.candidates[0].baseline.model_version == "baseline"


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("resamples", 0),
        ("resamples", 1.5),
        ("confidence_level", float("nan")),
        ("confidence_level", 1),
        ("minimum_clusters", 1),
    ],
)
def test_invalid_resampling_policy_is_rejected(field, value):
    with pytest.raises(ValueError):
        ResamplingPolicy(**{"seed": 7, field: value})


def test_unknown_seed_stays_unknown_and_defaults_survive_serialization():
    slot = EvaluationSlot("slot", "candidate", "case", "accuracy", "baseline", 0, DIGEST, "base")
    assert EvaluationSlot.from_dict(slot.to_dict()).seed is None
    assert slot.seed_policy == "unknown"
    policy = ResamplingPolicy.from_dict(ResamplingPolicy(seed=7).to_dict())
    assert (policy.resamples, policy.minimum_clusters, policy.confidence_level) == (10000, 20, 0.95)


@pytest.mark.parametrize("value", [-1, float("inf"), True])
def test_bad_cost_is_rejected(value):
    with pytest.raises(ValueError):
        CostEntry(value, "tokens", "estimated")


def test_unknown_fields_private_evidence_and_mutable_inputs_are_rejected():
    spec, manifest = protocol()
    data = manifest.to_dict()
    data["verified"] = True
    with pytest.raises(ValueError):
        DatasetManifest.from_dict(data)
    with pytest.raises(ValueError):
        replace(
            spec, evidence_refs=(EvidenceReference("private", EvidenceSourceKind.LATENT_STATE),)
        )
    with pytest.raises(ValueError):
        replace(manifest, cases=list(manifest.cases))
    with pytest.raises(ValueError):
        replace(manifest.cases[0], content_digest="wrong")


def test_source_json_is_canonical_detached_and_retains_restrictions():
    from evaluation.causal_records import canonical_json

    source = {"schema_version": "evaluation-ladder-v1", "training_eligibility": "HIDDEN_EVAL"}
    evidence = DiagnosticEvidence("slot", canonical_json(source), "rows", "probe_id", "probe")
    restored = DiagnosticEvidence.from_dict(evidence.to_dict())
    assert '"HIDDEN_EVAL"' in restored.source_json
    source["training_eligibility"] = "TRAIN"
    assert '"HIDDEN_EVAL"' in evidence.source_json
    with pytest.raises(ValueError):
        DiagnosticEvidence("slot", canonical_json(source), "rows", "probe_id", "probe")


@pytest.mark.parametrize(
    "kind", ["latent_state", "private_reasoning", "attention_data", "cache_data"]
)
def test_nested_source_private_evidence_is_rejected_before_authentication(kind):
    from evaluation.causal_records import canonical_json

    source = {
        "schema_version": "evaluation-ladder-v1",
        "training_eligibility": "HIDDEN_EVAL",
        "rows": [{"evidence_refs": [{"reference_id": "private", "source_kind": kind}]}],
    }
    with pytest.raises(ValueError, match="private"):
        DiagnosticEvidence("slot", canonical_json(source), "rows", "probe_id", "probe")
