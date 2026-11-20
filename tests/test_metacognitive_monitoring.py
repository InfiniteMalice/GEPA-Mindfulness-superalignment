"""Monitoring preserves uncertainty and derives bounded, source-linked findings."""

from dataclasses import replace

import pytest
from test_metacognitive_records import sources as sources

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.metacognitive_monitoring import MonitoringInput, monitor_episode
from gepa_mindfulness.verification.metacognitive_records import (
    AssumptionBinding,
    FindingKind,
    SourceSnapshot,
)
from gepa_mindfulness.verification.state import EvidenceClaim, EvidenceState


def kinds(result):
    """Inspect substantive diagnostic kinds without deriving expected values from production."""
    return {f.kind for f in result.findings}


def test_disabled_never_reads_input() -> None:
    """Default-off computation must tolerate even an unreadable input."""
    assert monitor_episode(object()) is None
    with pytest.raises(ValueError):
        monitor_episode(object(), enabled=1)


def test_shared_unverified_premise_is_not_false_consensus(sources: SourceSnapshot) -> None:
    """Common provenance cannot become independent confirmation or proof of falsehood."""
    result = monitor_episode(MonitoringInput(sources, "audit-1"), enabled=True)
    assert FindingKind.UNVERIFIED_ASSUMPTION in kinds(result)
    assert FindingKind.SHARED_DEPENDENCY in kinds(result)
    assert FindingKind.CONTRADICTION in kinds(result)
    assert result.sources.evidence_state.claims[0].status == "unverified"
    assert result.uncertainty.model is None
    assert any(
        d.kind == "assumption" and d.hypothesis_ids == ("h1", "h2")
        for d in result.shared_dependencies
    )


def test_supported_independent_inputs_need_no_intervention(sources: SourceSnapshot) -> None:
    """Supported premises and independent alternatives must not cause blanket warnings."""
    other = EvidenceReference("other-observation", EvidenceSourceKind.EXTERNAL_RECORD)
    h = sources.hypotheses
    changed = replace(
        sources,
        evidence_state=EvidenceState(
            (replace(sources.evidence_state.claims[0], status="supported"),)
        ),
        hypotheses=replace(
            h,
            assessments=(),
            hypotheses=(h.hypotheses[0], replace(h.hypotheses[1], evidence_refs=(other,))),
        ),
        assumptions=(sources.assumptions[0],),
    )
    result = monitor_episode(MonitoringInput(changed, "healthy"), enabled=True)
    assert result.findings == ()
    assert result.shared_dependencies == ()


@pytest.mark.parametrize(
    "status,expected",
    [
        ("unverified", FindingKind.UNVERIFIED_ASSUMPTION),
        ("unavailable", FindingKind.UNVERIFIED_ASSUMPTION),
        ("contradicted", FindingKind.CONTRADICTED_ASSUMPTION),
    ],
)
def test_assumption_status_remains_distinct(sources, status, expected) -> None:
    """Unavailable evidence is not proof of a contradicted premise."""
    changed = replace(
        sources,
        evidence_state=EvidenceState((replace(sources.evidence_state.claims[0], status=status),)),
    )
    result = monitor_episode(MonitoringInput(changed, "audit"), enabled=True)
    assert expected in kinds(result)
    assert result.sources.evidence_state.claims[0].status == status


def test_applicability_failure_does_not_rewrite_premise(sources) -> None:
    """A contradicted scope condition warns about applicability, not universal falsehood."""
    ref = sources.evidence_refs[0]
    changed = replace(
        sources,
        evidence_state=EvidenceState(
            (
                replace(sources.evidence_state.claims[0], status="supported"),
                EvidenceClaim("condition", "This is the tested domain", (ref,), "contradicted"),
            )
        ),
        assumptions=(replace(sources.assumptions[0], applicability_claim_ids=("condition",)),),
    )
    result = monitor_episode(MonitoringInput(changed, "audit"), enabled=True)
    assert FindingKind.OUT_OF_SCOPE_ASSUMPTION in kinds(result)
    assert result.sources.evidence_state.claims[0].status == "supported"


def test_legitimate_supersession_keeps_old_contradiction(sources) -> None:
    """The active successor governs current use while historical evidence stays inspectable."""
    old = sources.evidence_state.claims[0]
    changed = replace(
        sources,
        evidence_state=EvidenceState(
            (
                replace(old, status="superseded", superseded_by="revised"),
                EvidenceClaim("revised", "Revised premise", old.evidence_refs, "supported"),
            )
        ),
    )
    result = monitor_episode(MonitoringInput(changed, "audit"), enabled=True)
    assert FindingKind.CONTRADICTED_ASSUMPTION not in kinds(result)
    assert result.sources.evidence_state.claims[0].status == "superseded"


def test_shared_analysis_can_be_ablated_without_changing_sources(sources) -> None:
    """Ablation removes the diagnostic mechanism without altering accepted evidence."""
    original = sources.to_dict()
    result = monitor_episode(
        MonitoringInput(sources, "ablation", shared_analysis=False), enabled=True
    )
    assert result.shared_dependencies == ()
    assert FindingKind.SHARED_DEPENDENCY not in kinds(result)
    assert sources.to_dict() == original


def test_high_stakes_ambiguity_and_missing_evidence_are_distinct(sources) -> None:
    """Instruction ambiguity cannot be silently converted to epistemic abstention."""
    ref = sources.evidence_refs[0]
    changed = replace(
        sources,
        evidence_state=EvidenceState(
            sources.evidence_state.claims
            + (
                EvidenceClaim(
                    "ambiguity", "Instruction target is materially ambiguous", (ref,), "supported"
                ),
                EvidenceClaim("stakes", "Target has consequential effects", (ref,), "supported"),
                EvidenceClaim("answer", "Requested factual answer", (), "unavailable"),
            )
        ),
    )
    result = monitor_episode(
        MonitoringInput(
            changed,
            "audit",
            ambiguity_claim_id="ambiguity",
            stakes_claim_id="stakes",
            answer_claim_ids=("answer",),
        ),
        enabled=True,
    )
    assert FindingKind.AMBIGUOUS_INSTRUCTION in kinds(result)
    assert FindingKind.INSUFFICIENT_EVIDENCE in kinds(result)
    assert {f.next_step for f in result.findings} >= {"request_clarification", "request_evidence"}


def test_unknown_diagnostics_are_disclosed(sources) -> None:
    """Absent telemetry is explicit missing coverage rather than a successful check."""
    result = monitor_episode(MonitoringInput(sources, "audit"), enabled=True)
    assert result.status.value == "partial"
    assert set(result.unassessed_domains) >= {"continuity", "calibration", "repetition", "strategy"}


def test_monitoring_inputs_are_detached(sources) -> None:
    """Caller mutation after input construction cannot change the later assessment."""
    inputs = MonitoringInput(sources, "audit")
    object.__setattr__(sources.evidence_state.claims[0], "status", "supported")
    result = monitor_episode(inputs, enabled=True)
    assert FindingKind.UNVERIFIED_ASSUMPTION in kinds(result)


def test_long_source_ids_have_bounded_deterministic_finding_ids(sources) -> None:
    """A valid long premise ID must not overflow the diagnostic ID contract."""
    long_id = "x" * 512
    changed = replace(
        sources,
        evidence_state=EvidenceState(
            (replace(sources.evidence_state.claims[0], claim_id=long_id),)
        ),
        assumptions=tuple(replace(b, claim_id=long_id) for b in sources.assumptions),
    )
    result = monitor_episode(MonitoringInput(changed, "audit"), enabled=True)
    assert all(len(f.finding_id) <= 128 for f in result.findings)
    assert result == monitor_episode(MonitoringInput(changed, "audit"), enabled=True)


def test_dense_valid_inputs_disclose_omitted_findings(sources) -> None:
    """Nine assumptions with 64 conditions must return partial diagnostics, not crash."""
    ref = sources.evidence_refs[0]
    condition_ids = tuple(f"condition-{i}" for i in range(64))
    claims = tuple(
        EvidenceClaim(f"premise-{i}", "A premise", (ref,), "unverified") for i in range(9)
    ) + tuple(EvidenceClaim(c, "A condition", (ref,), "unverified") for c in condition_ids)
    changed = replace(
        sources,
        evidence_state=EvidenceState(claims),
        hypotheses=replace(sources.hypotheses, assessments=()),
        assumptions=tuple(AssumptionBinding("h1", f"premise-{i}", condition_ids) for i in range(9)),
    )
    result = monitor_episode(MonitoringInput(changed, "dense"), enabled=True)
    assert len(result.findings) == 512
    assert result.omitted_finding_count == 74
    assert result.status.value == "partial"
    assert "assumptions" in result.unassessed_domains
    assert len(result.sources.evidence_state.claims) == 73
    assert type(result).from_dict(result.to_dict()) == result


def test_many_shared_sources_disclose_omitted_dependencies(sources) -> None:
    """Shared provenance can exceed the view limit without erasing the original records."""
    refs = tuple(
        EvidenceReference(f"shared-{i}", EvidenceSourceKind.EXTERNAL_RECORD) for i in range(513)
    )
    h = sources.hypotheses
    changed = replace(
        sources,
        hypotheses=replace(
            h, hypotheses=tuple(replace(item, evidence_refs=refs) for item in h.hypotheses)
        ),
    )
    result = monitor_episode(MonitoringInput(changed, "dense"), enabled=True)
    assert len(result.shared_dependencies) == 512
    assert result.omitted_dependency_count == 2
    assert "dependencies" in result.unassessed_domains
    assert len(result.sources.hypotheses.hypotheses[0].evidence_refs) == 513


def test_one_claim_can_fill_both_ambiguity_and_stakes_roles(sources) -> None:
    """Shared task-role provenance must not create duplicate invalid finding references."""
    changed = replace(
        sources,
        evidence_state=EvidenceState(
            (replace(sources.evidence_state.claims[0], status="supported"),)
        ),
    )
    result = monitor_episode(
        MonitoringInput(
            changed, "shared-role", ambiguity_claim_id="premise", stakes_claim_id="premise"
        ),
        enabled=True,
    )
    finding = next(f for f in result.findings if f.kind is FindingKind.AMBIGUOUS_INSTRUCTION)
    assert finding.claim_ids == ("premise",)


def test_same_claim_can_be_premise_and_scope_condition(sources) -> None:
    """Distinct dependency roles may legitimately cite one retained claim."""
    changed = replace(
        sources,
        evidence_state=EvidenceState(
            (replace(sources.evidence_state.claims[0], status="contradicted"),)
        ),
        assumptions=(replace(sources.assumptions[0], applicability_claim_ids=("premise",)),),
    )
    result = monitor_episode(MonitoringInput(changed, "same-condition"), enabled=True)
    assert FindingKind.OUT_OF_SCOPE_ASSUMPTION in kinds(result)
