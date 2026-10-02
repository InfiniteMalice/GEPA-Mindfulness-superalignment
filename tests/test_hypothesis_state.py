"""Retention, conservative Pareto comparison and actor projection contracts."""

import json
from dataclasses import replace

import pytest

from evaluation.experimental_overlays import ExperimentalOverlayConfig
from evaluation.experimental_records import experimental_record_from_dict
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from gepa_mindfulness.verification.hypothesis_records import (
    Hypothesis,
    HypothesisAssessment,
    HypothesisScores,
    HypothesisState,
    HypothesisTrigger,
)
from gepa_mindfulness.verification.hypothesis_state import (
    append_hypotheses,
    pareto_hypotheses,
    project_hypotheses,
    validate_extension,
)
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

ON = ExperimentalOverlayConfig(competing_hypotheses=True)


def refs(name="evidence"):
    """Make observable evidence that never points at private model internals."""
    return (EvidenceReference(name, EvidenceSourceKind.EXTERNAL_RECORD),)


def candidate(name):
    """Give each public explanation an immutable identity."""
    return Hypothesis(name, f"Explanation {name}", refs())


def scores(**changes):
    """Supply complete normalized diagnostics with common complexity/compute units."""
    return replace(HypothesisScores(0.8, 0.2, 2, 0.2, 0.8, 2, 0.8), **changes)


def assessment(name, hypothesis="a", assessor="v1", **changes):
    """Declare one host assessment with explicit evidence and optional supersession."""
    return replace(
        HypothesisAssessment(name, hypothesis, assessor, "unresolved", scores(), refs()), **changes
    )


def state(*, count=3, assessments=()):
    """Create an inert external history with no default runtime effect."""
    return HypothesisState(
        "state",
        EpistemicContext("run", 0, EvaluatedSystemVersion("model", "harness")),
        1,
        "measurement-protocol-v1",
        "nodes",
        "tokens",
        tuple(candidate(chr(97 + i)) for i in range(count)),
        assessments,
        (HypothesisTrigger("t1", "structural_alternatives", refs()),),
    )


def test_pareto_dominance_ties_and_tradeoffs_never_prune():
    """Complete vectors preserve ties/tradeoffs and only diagnose strict domination."""
    prior = state(
        count=4,
        assessments=(
            assessment("a1"),
            assessment("b1", "b", scores=scores(evidence_fit=0.6)),
            assessment("c1", "c", scores=scores(complexity=1, risk=0.4)),
            assessment("d1", "d"),
        ),
    )
    report = pareto_hypotheses(prior, config=ON)
    assert report["frontier"] == ["a", "c", "d"]
    assert report["dominated"] == ["b"]
    assert report["incomparable"] == []
    assert len(prior.hypotheses) == 4
    assert report["authority_granted"] is False


def test_unknown_and_conflicting_assessments_stay_incomparable():
    """No optimistic scalar collapse or convenient verifier selection is permitted."""
    prior = state(
        assessments=(
            assessment("a1"),
            assessment("a2", assessor="v2", status="challenged"),
            assessment("b1", "b", scores=scores(risk=None)),
        )
    )
    result = pareto_hypotheses(prior, config=ON)
    assert result["frontier"] == ["a", "b", "c"]
    assert result["incomparable"] == ["a", "b", "c"]
    assert result["conflicted"] == ["a"]


def test_explicit_supersession_retains_history_and_other_verifiers():
    """A verifier can revise its own live record without erasing another verdict."""
    prior = state(assessments=(assessment("a1"), assessment("a2", assessor="v2")))
    revised = append_hypotheses(
        prior, assessments=(assessment("a3", status="challenged", supersedes="a1"),), config=ON
    )
    validate_extension(prior, revised)
    assert revised.assessments[:2] == prior.assessments
    assert revised.revision == prior.revision + 1
    assert pareto_hypotheses(revised, config=ON)["conflicted"] == ["a"]
    assert len(prior.assessments) == 2


@pytest.mark.parametrize("change", ["foreign", "missing", "already", "future", "wrong_hyp"])
def test_bad_supersession_graph_is_rejected(change):
    """Only earlier live records for the same assessor/hypothesis can be superseded."""
    records = [assessment("a1")]
    new = assessment("a2", supersedes="a1")
    if change == "foreign":
        new = replace(new, assessor_id="v2")
    elif change == "missing":
        new = replace(new, supersedes="unknown")
    elif change == "already":
        records.append(assessment("a0", supersedes="a1"))
    elif change == "future":
        records = [replace(records[0], supersedes="a2")]
        new = replace(new, supersedes=None)
    else:
        new = replace(new, hypothesis_id="b")
    with pytest.raises(ValueError):
        state(assessments=tuple(records) + (new,))


@pytest.mark.parametrize("change", ["delete", "edit", "units", "case", "context", "revision"])
def test_extension_rejects_forged_successor(change):
    """Restoring JSON cannot authorize editing or omitting prior records."""
    prior = state()
    successor = append_hypotheses(prior, hypotheses=(candidate("z"),), config=ON)
    if change == "delete":
        successor = replace(successor, hypotheses=successor.hypotheses[1:])
    elif change == "edit":
        successor = replace(
            successor,
            hypotheses=(replace(candidate("a"), statement="new"),) + successor.hypotheses[1:],
        )
    elif change == "units":
        successor = replace(successor, compute_unit="seconds")
    elif change == "case":
        successor = replace(successor, source_case_id=2)
    elif change == "context":
        successor = replace(successor, context=replace(successor.context, run_id="other"))
    else:
        successor = replace(successor, revision=0)
    with pytest.raises(ValueError):
        validate_extension(prior, HypothesisState.from_dict(successor.to_dict()))


def test_projection_pages_reach_every_candidate_and_preserve_state():
    """Bounds never silently turn omissions into deletion or a single incumbent."""
    original = state(
        count=5,
        assessments=(
            assessment("a1"),
            assessment("b1", "b", scores=scores(evidence_fit=0.1)),
        ),
    )
    before = original.to_dict()
    seen = set()
    offset = 0
    while offset is not None:
        page = project_hypotheses(
            original, diagnostic_uncertainty=0.7, offset=offset, limit=2, config=ON
        )
        assert len(page["candidates"]) == 2
        assert page["total_hypotheses"] == 5 and page["omitted_count"] == 3
        assert experimental_record_from_dict(page["diagnostic"]).hypotheses
        seen.update(item["id"] for item in page["candidates"])
        offset = page["next_offset"]
        page["candidates"].clear()
    assert seen == {h.id for h in original.hypotheses}
    assert original.to_dict() == before


def test_actor_projection_excludes_external_provenance_and_marks_unknowns():
    """Only bounded public statements and score/status diagnostics reach the actor."""
    original = state(
        assessments=(
            assessment(
                "SECRET_ASSESSMENT",
                assessor="SECRET_ASSESSOR",
                evidence_refs=refs("SECRET_EVIDENCE"),
            ),
        )
    )
    original = replace(
        original,
        state_id="SECRET_STATE",
        context=replace(original.context, run_id="SECRET_RUN"),
        triggers=(HypothesisTrigger("SECRET_TRIGGER", "regime_shift", refs("SECRET_TRIGGER_REF")),),
    )
    page = project_hypotheses(original, diagnostic_uncertainty=0.8, config=ON)
    assert "SECRET" not in json.dumps(page)
    assert page["candidates"][1]["scores"] is None
    assert page["candidates"][1]["incomparable"] is True
    assert page["diagnostic"]["uncertainty"] == 0.8


def test_roundtrip_detaches_nested_records_and_preserves_nontrain():
    """External state survives JSON exactly and remains optimizer-ineligible."""
    original = replace(
        state(assessments=(assessment("a1"),)), training_eligibility=TrainingEligibility.HIDDEN_EVAL
    )
    encoded = original.to_dict()
    assert HypothesisState.from_dict(json.loads(json.dumps(encoded))).to_dict() == encoded
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(encoded)
    encoded["hypotheses"][0]["statement"] = "changed"
    assert original.hypotheses[0].statement == "Explanation a"


def test_legacy_evidence_method_overrides_are_not_executed():
    """Caller-owned EvidenceReference instance methods cannot forge private provenance."""
    ref = refs()[0]
    ref.__dict__["to_dict"] = lambda: pytest.fail("untrusted method invoked")
    ref.__dict__["__dataclass_fields__"] = {}
    prior = replace(state(), hypotheses=(Hypothesis("a", "public a", (ref,)), candidate("b")))
    assert prior.to_dict()["hypotheses"][0]["evidence_refs"][0]["reference_id"] == "evidence"
    object.__setattr__(ref, "reference_id", "mutated")
    assert prior.hypotheses[0].evidence_refs[0].reference_id == "evidence"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_unit_dimensions_reject_invalid_numbers(value):
    """Scores never coerce booleans/nonfinite/out-of-range inputs."""
    with pytest.raises(ValueError):
        scores(risk=value)


@pytest.mark.parametrize("value", [2**53 + 1, -1, float("inf"), True])
def test_cost_dimensions_reject_lossy_or_invalid_numbers(value):
    """Cost inputs retain numeric precision rather than silently rounding integers."""
    with pytest.raises(ValueError):
        scores(compute_cost=value)


def test_missing_numbers_are_not_zero_and_large_exact_costs_work():
    """Unavailable dimensions differ from legitimate zero and exact large measurements."""
    assert HypothesisScores().risk is None
    assert scores(risk=0, complexity=2**60).complexity == 2**60


@pytest.mark.parametrize("limit,offset", [(1, 0), (33, 0), (True, 0), (2, -1), (2, 2), (2, True)])
def test_projection_bounds_fail_closed(limit, offset):
    """A projection always has two alternatives and bounded, exact integer coordinates."""
    with pytest.raises(ValueError):
        project_hypotheses(
            state(), diagnostic_uncertainty=0.5, limit=limit, offset=offset, config=ON
        )


def test_flags_are_explicit_and_all_changes_are_additive():
    """Default operations are disabled; enabled operations cannot silently do nothing."""
    prior = state()
    for operation in (
        lambda: append_hypotheses(prior, hypotheses=(candidate("z"),)),
        lambda: pareto_hypotheses(prior),
        lambda: project_hypotheses(prior, diagnostic_uncertainty=0.5),
    ):
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ValueError):
        append_hypotheses(prior, config=ON)
    with pytest.raises(ValueError):
        append_hypotheses(prior, hypotheses=(candidate("a"),), config=ON)


@pytest.mark.parametrize(
    "kind",
    [
        "persistent_innovation",
        "multimodal_evidence",
        "verifier_conflict",
        "regime_shift",
        "structural_alternatives",
    ],
)
def test_each_declared_trigger_is_retained(kind):
    """Trigger labels preserve evidence without inferring runtime policy."""
    prior = state()
    new = append_hypotheses(prior, triggers=(HypothesisTrigger("t2", kind, refs()),), config=ON)
    assert new.triggers[0] == prior.triggers[0]
    assert new.triggers[1].kind == kind


def test_unknown_fields_bad_provenance_and_text_limits_are_rejected():
    """JSON import stays strict and byte bounds include multibyte text."""
    raw = state().to_dict()
    raw["actor_may_delete"] = True
    with pytest.raises(ValueError):
        HypothesisState.from_dict(raw)
    with pytest.raises(ValueError):
        Hypothesis("h", "public", (EvidenceReference("x", EvidenceSourceKind.PRIVATE_REASONING),))
    with pytest.raises(ValueError):
        Hypothesis("h", "😀" * 129, refs())
    assert Hypothesis("h", "😀" * 128, refs()).statement
    with pytest.raises(ValueError):
        replace(state(), training_eligibility=TrainingEligibility.TRAIN)


def test_projection_labels_do_not_collide_when_ids_contain_separators():
    """Distinct valid alternatives must not collapse into the same diagnostic label."""
    prior = replace(
        state(), hypotheses=(Hypothesis("a", "b: c", refs()), Hypothesis("a: b", "c", refs()))
    )
    page = project_hypotheses(prior, diagnostic_uncertainty=0.5, config=ON)
    assert len(set(page["diagnostic"]["hypotheses"])) == 2


def test_all_history_prefixes_are_protected_and_empty_successor_fails():
    """Revision increments alone cannot authorize missing assessment/trigger history."""
    prior = state(assessments=(assessment("a1"),))
    with pytest.raises(ValueError):
        validate_extension(prior, replace(prior, revision=1))
    new = append_hypotheses(
        prior, triggers=(HypothesisTrigger("t2", "regime_shift", refs()),), config=ON
    )
    for forged in (replace(new, assessments=()), replace(new, triggers=new.triggers[1:])):
        with pytest.raises(ValueError):
            validate_extension(prior, forged)


def test_disagreeing_scores_remain_conflicted_until_each_assessor_revises():
    """A newer conflicting record cannot win by insertion order."""
    prior = state(
        assessments=(assessment("a1"), assessment("a2", assessor="v2", scores=scores(risk=0.9)))
    )
    assert pareto_hypotheses(prior, config=ON)["conflicted"] == ["a"]
    agreed = append_hypotheses(
        prior, assessments=(assessment("a3", assessor="v2", supersedes="a2"),), config=ON
    )
    result = pareto_hypotheses(agreed, config=ON)
    assert result["conflicted"] == [] and "a" not in result["incomparable"]
    assert len(agreed.assessments) == 3


def test_subclasses_and_mutated_nested_records_fail_without_callbacks():
    """Exact strings and canonical enum checks precede nested legacy validators."""

    class BadString(str):
        def strip(self):
            pytest.fail("subclass method invoked")

    ref = refs()[0]
    object.__setattr__(ref, "reference_id", BadString(""))
    with pytest.raises(ValueError):
        Hypothesis("a", "public", (ref,))
    original = state()
    object.__setattr__(original.context.system, "model_version", BadString(""))
    with pytest.raises(ValueError):
        original.to_dict()
    config = ExperimentalOverlayConfig(competing_hypotheses=True)
    object.__setattr__(config, "competing_hypotheses", 1)
    with pytest.raises(ValueError):
        pareto_hypotheses(state(), config=config)


@pytest.mark.parametrize(
    "field,value",
    [("id", ""), ("hypothesis_id", "absent"), ("status", "retired"), ("supersedes", "self")],
)
def test_invalid_assessment_records_are_rejected(field, value):
    """No removal status or dangling identity can enter a history."""
    with pytest.raises(ValueError):
        state(assessments=(replace(assessment("self"), **{field: value}),))


def test_extra_json_record_fields_are_rejected():
    """Nested restored records have exactly the declared schema."""
    raw = state().to_dict()
    raw["hypotheses"][0]["can_delete"] = True
    with pytest.raises(ValueError):
        HypothesisState.from_dict(raw)


def test_json_restore_rejects_container_subclasses_without_invoking_them():
    """Only inert JSON containers can cross the external snapshot boundary."""

    class CallbackDict(dict):
        def __iter__(self):
            pytest.fail("untrusted mapping invoked")

    raw = state().to_dict()
    raw["context"] = CallbackDict(raw["context"])
    with pytest.raises(ValueError):
        HypothesisState.from_dict(raw)


def test_json_restore_rejects_cycles_and_malformed_snapshots():
    """Cyclic containers, non-array history and unknown schemas fail closed."""
    raw = state().to_dict()
    raw["cycle"] = raw
    with pytest.raises(ValueError):
        HypothesisState.from_dict(raw)
    raw = state().to_dict()
    raw["assessments"] = {}
    with pytest.raises(ValueError):
        HypothesisState.from_dict(raw)
    raw = state().to_dict()
    raw["schema_version"] = "future"
    with pytest.raises(ValueError):
        HypothesisState.from_dict(raw)
