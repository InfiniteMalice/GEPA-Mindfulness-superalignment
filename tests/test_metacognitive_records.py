"""Derived diagnostics cannot promote evidence or lose source identity/history."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from gepa_mindfulness.verification.hypothesis_records import (
    Hypothesis,
    HypothesisAssessment,
    HypothesisScores,
    HypothesisState,
    HypothesisTrigger,
)
from gepa_mindfulness.verification.metacognitive_records import (
    AssessmentStatus,
    AssumptionBinding,
    FindingKind,
    MetacognitiveAssessment,
    MonitoringFinding,
    SharedDependency,
    SourceSnapshot,
    UncertaintyDimensions,
)
from gepa_mindfulness.verification.state import EvidenceClaim, EvidenceState
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion


@pytest.fixture
def sources() -> SourceSnapshot:
    """Two alternatives depend on one unverified premise with observable provenance."""
    ref = EvidenceReference("observation-1", EvidenceSourceKind.EXTERNAL_RECORD)
    context = EpistemicContext("run-1", 0, EvaluatedSystemVersion("model-1", "harness-1"))
    hypotheses = HypothesisState(
        "hypotheses-1",
        context,
        1,
        "protocol-1",
        "nodes",
        "tokens",
        (Hypothesis("h1", "Explanation A", (ref,)), Hypothesis("h2", "Explanation B", (ref,))),
        (
            HypothesisAssessment("a1", "h1", "judge-1", "supported", HypothesisScores(), (ref,)),
            HypothesisAssessment("a2", "h1", "judge-2", "challenged", HypothesisScores(), (ref,)),
        ),
        (HypothesisTrigger("trigger-1", "structural_alternatives", (ref,)),),
    )
    cell = plan_v5_cells(
        case_ids=[1],
        stripe_ids=["NONE"],
        repeats=1,
        model_version="model-1",
        harness_version="harness-1",
    )[0]
    return SourceSnapshot(
        context=context,
        cell=cell,
        evidence_state=EvidenceState(
            (EvidenceClaim("premise", "Premise X", (ref,), "unverified"),)
        ),
        hypotheses=hypotheses,
        epistemic=None,
        assumptions=(AssumptionBinding("h1", "premise"), AssumptionBinding("h2", "premise")),
        source_event_ids=("event-1",),
        provenance=("recorded-episode-1",),
    )


def assessment(sources: SourceSnapshot) -> MetacognitiveAssessment:
    """Construct an inspectable view without changing the premise's truth status."""
    refs = sources.evidence_state.claims[0].evidence_refs
    return MetacognitiveAssessment(
        assessment_id="assessment-1",
        sources=sources,
        method_version="rules-v1",
        status=AssessmentStatus.COMPLETE,
        uncertainty=UncertaintyDimensions(),
        findings=(
            MonitoringFinding(
                finding_id="finding-1",
                kind=FindingKind.UNVERIFIED_ASSUMPTION,
                claim_ids=("premise",),
                hypothesis_ids=("h1", "h2"),
                evidence_refs=refs,
                source_event_ids=("event-1",),
                uncertainty=None,
                next_step="verify_assumption",
            ),
        ),
        shared_dependencies=(SharedDependency("premise", "assumption", ("h1", "h2"), refs),),
    )


def test_round_trip_retains_unknowns_conflicts_and_existing_status(sources: SourceSnapshot) -> None:
    """JSON restoration preserves every original source and unknown dimension."""
    value = assessment(sources)
    wire = json.loads(json.dumps(value.to_dict(), allow_nan=False, sort_keys=True))
    restored = MetacognitiveAssessment.from_dict(wire)
    assert restored == value
    assert restored.sources.evidence_state.claims[0].status == "unverified"
    assert len(restored.sources.hypotheses.assessments) == 2
    assert restored.uncertainty.model is None
    assert restored.findings[0].verification_status == "unverified"
    assert wire["diagnostic_status"] == "diagnostic"


@pytest.mark.parametrize("mutation", ["extra", "missing", "version", "record_type"])
def test_wire_schema_is_exact(sources: SourceSnapshot, mutation: str) -> None:
    """Malformed or unsupported wire data cannot silently drop information."""
    wire = assessment(sources).to_dict()
    if mutation == "extra":
        wire["authority_granted"] = True
    elif mutation == "missing":
        del wire["sources"]
    elif mutation == "version":
        wire["schema_version"] = "future-v2"
    else:
        wire["record_type"] = "verified_evidence"
    with pytest.raises(ValueError):
        MetacognitiveAssessment.from_dict(wire)


@pytest.mark.parametrize("dimension", ["model", "environmental", "missing_evidence"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "unknown"])
def test_uncertainty_rejects_coercion(dimension: str, value: object) -> None:
    """Invalid values cannot become plausible-looking uncertainty measurements."""
    with pytest.raises(ValueError):
        UncertaintyDimensions(**{dimension: value})


def test_nested_unknown_field_is_rejected(sources: SourceSnapshot) -> None:
    """Strict parsing applies recursively, not just to the enclosing record."""
    wire = assessment(sources).to_dict()
    wire["findings"][0]["hidden_reward"] = 1
    with pytest.raises(ValueError):
        MetacognitiveAssessment.from_dict(wire)


@pytest.mark.parametrize("field", ["repeat_id", "model_version", "harness_version"])
def test_source_context_must_match_cell(sources: SourceSnapshot, field: str) -> None:
    """Cross-repeat and cross-version records cannot be relabeled as this cell."""
    changed = replace(sources.cell, **{field: 1 if field == "repeat_id" else "other"})
    with pytest.raises(ValueError):
        replace(sources, cell=changed)


def test_hypothesis_case_must_match_cell(sources: SourceSnapshot) -> None:
    """Hypothesis source case cannot drift from the evaluation identity."""
    with pytest.raises(ValueError):
        replace(sources, hypotheses=replace(sources.hypotheses, source_case_id=2))


def test_missing_assumption_source_is_rejected(sources: SourceSnapshot) -> None:
    """A binding cannot invent a premise or an alternative."""
    for binding in (AssumptionBinding("h1", "missing"), AssumptionBinding("missing", "premise")):
        with pytest.raises(ValueError):
            replace(sources, assumptions=(binding,))


def test_findings_cannot_invent_references(sources: SourceSnapshot) -> None:
    """Evidence, hypothesis, claim and event links must resolve in retained inputs."""
    value = assessment(sources)
    for field, changed in (
        ("claim_ids", ("missing",)),
        ("hypothesis_ids", ("missing",)),
        ("source_event_ids", ("future-event",)),
        ("evidence_refs", (EvidenceReference("invented", EvidenceSourceKind.EXTERNAL_RECORD),)),
    ):
        with pytest.raises(ValueError):
            replace(value, findings=(replace(value.findings[0], **{field: changed}),))


def test_dependencies_need_multiple_known_hypotheses(sources: SourceSnapshot) -> None:
    """A shared-dependency report must actually link multiple retained alternatives."""
    value = assessment(sources)
    with pytest.raises(ValueError):
        SharedDependency("premise", "assumption", ("h1",), ())
    with pytest.raises(ValueError):
        replace(
            value,
            shared_dependencies=(SharedDependency("missing", "assumption", ("h1", "h2"), ()),),
        )


def test_source_history_is_detached_and_frozen(sources: SourceSnapshot) -> None:
    """Later caller mutation cannot rewrite an existing derived view."""
    value = assessment(sources)
    object.__setattr__(sources.evidence_state.claims[0], "proposition", "changed later")
    assert value.sources.evidence_state.claims[0].proposition == "Premise X"
    with pytest.raises(FrozenInstanceError):
        value.assessment_id = "changed"
    wire = value.to_dict()
    wire["sources"]["evidence_state"]["claims"][0]["status"] = "supported"
    assert value.sources.evidence_state.claims[0].status == "unverified"


def test_superseded_claims_retain_original_provenance(sources: SourceSnapshot) -> None:
    """A successor does not erase the premise and references originally considered."""
    old = sources.evidence_state.claims[0]
    state = EvidenceState(
        (
            replace(old, status="superseded", superseded_by="replacement"),
            EvidenceClaim("replacement", "Revised premise", old.evidence_refs, "contradicted"),
        )
    )
    value = assessment(replace(sources, evidence_state=state))
    assert value.sources.evidence_state.claims[0].proposition == "Premise X"
    assert value.sources.evidence_state.resolve("premise").status == "contradicted"
    assert MetacognitiveAssessment.from_dict(value.to_dict()) == value


def test_private_source_cannot_certify_monitoring(sources: SourceSnapshot) -> None:
    """A private reasoning reference cannot masquerade as external diagnostic support."""
    private = EvidenceReference("thought", EvidenceSourceKind.PRIVATE_REASONING)
    with pytest.raises(ValueError):
        replace(
            sources,
            evidence_state=EvidenceState(
                (EvidenceClaim("premise", "Premise X", (private,), "unverified"),)
            ),
        )


def test_diagnostic_cannot_be_promoted_by_wire_label(sources: SourceSnapshot) -> None:
    """Changing a record label cannot mint accepted evidence or a verified finding."""
    value = assessment(sources)
    with pytest.raises(ValueError):
        EvidenceState.from_dict(value.to_dict())
    with pytest.raises(ValueError):
        replace(value.findings[0], verification_status="supported")
    wire = value.to_dict()
    wire["diagnostic_status"] = "verified"
    with pytest.raises(ValueError):
        MetacognitiveAssessment.from_dict(wire)
    from gepa_mindfulness.core.epistemic_process import EpistemicProcessAssessment

    with pytest.raises(ValueError):
        EpistemicProcessAssessment(verified_components=(value,))


def test_revalidation_rejects_mutable_bypass(sources: SourceSnapshot) -> None:
    """Serialization revalidates values even after a frozen-record bypass."""
    value = assessment(sources)
    object.__setattr__(value.uncertainty, "model", float("nan"))
    with pytest.raises(ValueError):
        value.to_dict()


def test_unassessed_domains_survive_wire_round_trip(sources: SourceSnapshot) -> None:
    """Missing diagnostic coverage must not disappear from serialized assessments."""
    value = replace(
        assessment(sources),
        status=AssessmentStatus.PARTIAL,
        unassessed_domains=("calibration", "continuity"),
    )
    restored = MetacognitiveAssessment.from_dict(value.to_dict())
    assert restored.unassessed_domains == ("calibration", "continuity")
    with pytest.raises(ValueError):
        replace(value, unassessed_domains=("invented",))


def test_truncation_metadata_is_versioned_and_v1_remains_readable(sources) -> None:
    """New omission counts cannot silently alter the original assessment wire schema."""
    value = assessment(sources)
    wire = value.to_dict()
    assert wire["schema_version"] == "metacognitive-assessment-v2"
    legacy = dict(wire)
    del legacy["omitted_finding_count"]
    del legacy["omitted_dependency_count"]
    legacy["schema_version"] = "metacognitive-v1"
    restored = MetacognitiveAssessment.from_dict(legacy)
    assert restored == value
    for bad in (True, -1, "1"):
        with pytest.raises(ValueError):
            replace(value, omitted_finding_count=bad)


def test_source_ids_cannot_exceed_diagnostic_reference_bounds(sources) -> None:
    """Reject an oversized source ID before it can fail later finding construction."""
    with pytest.raises(ValueError):
        replace(
            sources,
            assumptions=(),
            evidence_state=EvidenceState(
                (
                    replace(
                        sources.evidence_state.claims[0], claim_id="x" * 513, status="contradicted"
                    ),
                )
            ),
        )
    oversized = EvidenceReference("r" * 513, EvidenceSourceKind.EXTERNAL_RECORD)
    with pytest.raises(ValueError):
        replace(
            sources,
            evidence_state=EvidenceState(
                (replace(sources.evidence_state.claims[0], evidence_refs=(oversized,)),)
            ),
        )
