"""Candidate families cannot bypass existing evidence, budget or authority contracts."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from evaluation.family_search import prepare_family_search, summarize_family_search
from evaluation.family_search_records import (
    FamilyMembership,
    FamilySearchConfig,
    FamilySearchPlan,
    SearchFamily,
)
from evaluation.improvement_records import (
    AttemptEvent,
    AttemptJournal,
    CandidateSpec,
    CostEntry,
    DatasetCase,
    DatasetManifest,
    EvaluationSlot,
    ImprovementProtocol,
    MetricSpec,
    ResamplingPolicy,
    SystemConfig,
    record_digest,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import require_training_eligible

DIGEST = "a" * 64
BUDGET = "b" * 64
REFS = (EvidenceReference("host:source", EvidenceSourceKind.EXTERNAL_RECORD),)
EVALUATOR = TrustedEvaluatorContract("independent", "v1", "rubric-v1")
ENABLED = FamilySearchConfig(enabled=True)


def fixture():
    """Two independent roots and a later donor transfer share a hand-checked paired roster."""
    manifest = DatasetManifest(
        "tasks",
        tuple(
            DatasetCase(c, content_digest(c), c, purpose)
            for c, purpose in (
                ("selection", "optimizer_selection"),
                ("audit", "independent_audit"),
            )
        ),
        REFS,
    )
    baseline = SystemConfig("base-model", "base-harness", DIGEST)
    candidates = tuple(
        CandidateSpec(
            cid,
            baseline,
            SystemConfig(f"model:{cid}", "harness-v1", content_digest(cid)),
            content_digest(f"proposal:{cid}"),
            0 if cid.endswith("0") else 10,
            None if cid.endswith("0") else "a0",
        )
        for cid in ("a0", "b0", "a1", "b1")
    )
    metric = MetricSpec(
        "quality",
        "evaluation-ladder-v1",
        "representation_accuracy",
        content_digest(
            {"evaluator_id": "independent", "evaluator_version": "v1", "contract_id": "rubric-v1"}
        ),
        "fraction",
    )
    slots = tuple(
        EvaluationSlot(
            f"{c.candidate_id}:{case.case_id}:{seed}:{arm}",
            c.candidate_id,
            case.case_id,
            "quality",
            arm,
            0,
            BUDGET,
            f"{c.candidate_id}:seed:{seed}",
            seed,
            "shared",
        )
        for c in candidates
        for case in manifest.cases
        for seed in (0, 1)
        for arm in ("baseline", "candidate")
    )
    protocol = ImprovementProtocol(
        "protocol",
        record_digest(manifest),
        candidates,
        (metric,),
        slots,
        EVALUATOR,
        REFS,
        ResamplingPolicy(7, resamples=20, minimum_clusters=2),
    )
    families = (
        SearchFamily("model-family", "model", "Localized policy correction", REFS),
        SearchFamily("harness-family", "harness", "Evidence retrieval sequencing", REFS),
    )
    members = tuple(
        FamilyMembership(
            cid,
            "model-family" if cid.startswith("a") else "harness-family",
            int(cid[-1]),
            f"r{cid[-1]}",
        )
        for cid in ("a0", "b0", "a1", "b1")
    )
    plan = FamilySearchPlan(
        protocol,
        content_digest("fixed task"),
        DIGEST,
        DIGEST,
        BUDGET,
        DIGEST,
        families,
        members,
        REFS,
    )
    return plan, manifest


def prepare(plan=None, manifest=None):
    """Run the real adapter over the controlled protocol rather than a mock evaluator."""
    p, m = fixture()
    return prepare_family_search(plan or p, manifest or m, config=ENABLED)


def test_disabled_adapter_does_not_inspect_inputs():
    assert prepare_family_search(None, None) is None


def test_exact_records_roundtrip_without_authority_or_type_coercion():
    plan, _ = fixture()
    for value in (FamilySearchConfig(), plan, plan.families[0], plan.members[0]):
        wire = value.to_dict()
        assert type(value).from_dict(wire) == value
        for change in (
            {"optimizer_input": True},
            {"confers_authority": True},
            {"training_eligibility": "TRAIN"},
            {"schema_version": "unknown"},
            {"approved": True},
        ):
            with pytest.raises(ValueError):
                type(value).from_dict({**wire, **change})
    for field in ("families", "members"):
        for bad in ({"a": {}}, "ab"):
            with pytest.raises(ValueError):
                FamilySearchPlan.from_dict({**plan.to_dict(), field: bad})
    with pytest.raises(ValueError):
        FamilySearchConfig(enabled=1)


def test_preparation_retains_roots_transfers_and_matching_budgeted_pairs():
    plan, manifest = fixture()
    before = plan.to_dict(), manifest.to_dict()
    result = prepare(plan, manifest)
    assert result["paired_cells_per_candidate"] == 4
    assert [r["planned_evaluations"] for r in result["families"]] == [16, 16]
    assert result["transfers"] == [
        {
            "candidate_id": "b1",
            "donor_candidate_id": "a0",
            "from_family": "model-family",
            "to_family": "harness-family",
            "round_index": 1,
            "round_id": "r1",
        }
    ]
    assert result["optimizer_input"] is False and result["confers_authority"] is False
    assert result["promotion_eligible"] is False
    assert result["model_effectiveness_established"] is False
    assert (plan.to_dict(), manifest.to_dict()) == before
    with pytest.raises(ValueError):
        require_training_eligible(result)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_family", "alias"])
def test_ambiguous_candidate_membership_is_rejected(mutation):
    p, _ = fixture()
    members = p.members
    if mutation == "missing":
        members = members[:-1]
    elif mutation == "duplicate":
        members = members[:-1] + (members[0],)
    elif mutation == "unknown_family":
        members = (replace(members[0], family_id="unknown"),) + members[1:]
    else:
        members = (replace(members[0], candidate_id="A0"),) + members[1:]
    with pytest.raises(ValueError):
        prepare(replace(p, members=members))


def test_families_require_unique_identities_and_cannot_claim_normative_authority():
    p, _ = fixture()
    with pytest.raises(ValueError):
        replace(p, families=(p.families[0], replace(p.families[1], family_id="MODEL-FAMILY")))
    with pytest.raises(ValueError):
        replace(p.families[0], surface="constitution")


@pytest.mark.parametrize("mutation", ["missing_root", "parent_round", "parent_freeze", "cycle"])
def test_lineage_preserves_independent_roots_and_supported_chronology(mutation):
    p, _ = fixture()
    members, candidates = p.members, p.protocol.candidates
    if mutation == "missing_root":
        members = (members[0], replace(members[1], round_index=1, round_id="r1"), *members[2:])
    elif mutation == "parent_round":
        members = (
            members[0],
            members[1],
            replace(members[2], round_index=0, round_id="r0"),
            members[3],
        )
    elif mutation == "parent_freeze":
        candidates = (replace(candidates[0], freeze_ordinal=10), *candidates[1:])
    else:
        candidates = (replace(candidates[0], parent_candidate_id="a1"), *candidates[1:])
    with pytest.raises(ValueError):
        prepare(replace(p, members=members, protocol=replace(p.protocol, candidates=candidates)))


def test_donor_transfers_obey_initial_separation_and_explicit_sharing_schedule():
    p, _ = fixture()
    with pytest.raises(ValueError):
        prepare(replace(p, independent_rounds=2))
    members = p.members[:-1] + (replace(p.members[-1], round_index=2, round_id="r2"),)
    with pytest.raises(ValueError):
        prepare(replace(p, members=members, sharing_interval=2))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_arm",
        "different_roster",
        "budget",
        "seed",
        "unknown_seed",
        "single_seed",
        "baseline",
    ],
)
def test_unmatched_task_seed_budget_or_system_rosters_are_rejected(mutation):
    p, _ = fixture()
    slots, candidates = p.protocol.slots, p.protocol.candidates
    if mutation == "missing_arm":
        slots = slots[1:]
    elif mutation == "different_roster":
        slots = slots[2:]
    elif mutation == "budget":
        slots = (replace(slots[0], budget_digest="c" * 64),) + slots[1:]
    elif mutation == "seed":
        slots = (replace(slots[0], seed=3),) + slots[1:]
    elif mutation == "unknown_seed":
        slots = tuple(replace(s, seed=None, seed_policy="unknown") for s in slots)
    elif mutation == "single_seed":
        slots = tuple(s for s in slots if s.seed == 0)
    else:
        candidates = (
            replace(candidates[0], baseline=SystemConfig("other", "h", DIGEST)),
            *candidates[1:],
        )
    with pytest.raises(ValueError):
        prepare(replace(p, protocol=replace(p.protocol, slots=slots, candidates=candidates)))


def test_training_tasks_and_manifest_substitution_are_rejected():
    p, manifest = fixture()
    changed = replace(
        manifest,
        cases=(replace(manifest.cases[0], purpose="synthetic_training"), *manifest.cases[1:]),
    )
    with pytest.raises(ValueError):
        prepare(p, changed)
    p = replace(p, protocol=replace(p.protocol, manifest_digest=record_digest(changed)))
    with pytest.raises(ValueError):
        prepare(p, changed)


@pytest.mark.parametrize(
    "change", [{"max_candidates_per_family": 1}, {"max_evaluations_per_family": 8}]
)
def test_each_family_gets_the_same_predeclared_candidate_and_evaluation_caps(change):
    p, _ = fixture()
    with pytest.raises(ValueError):
        prepare(replace(p, **change))


def journal_fixture(plan):
    """Keep failed, rejected, pending and declared-selected attempts with unknown costs."""
    slots = {
        cid: next(s for s in plan.protocol.slots if s.candidate_id == cid).slot_id
        for cid in ("a0", "b0", "a1", "b1")
    }
    events = (
        AttemptEvent("p-a0", 0, "a0", "r0", "proposal"),
        AttemptEvent("p-b0", 1, "b0", "r0", "proposal"),
        AttemptEvent("s-a0", 2, "a0", "capture", "evaluation_started", slots["a0"]),
        AttemptEvent(
            "f-a0",
            3,
            "a0",
            "capture",
            "evaluation_finished",
            slots["a0"],
            costs=(CostEntry(2, "tokens", "estimated"),),
        ),
        AttemptEvent("s-b0", 4, "b0", "capture", "evaluation_started", slots["b0"]),
        AttemptEvent(
            "f-b0",
            5,
            "b0",
            "capture",
            "evaluation_finished",
            slots["b0"],
            costs=(CostEntry(None, "tokens", "estimated"),),
        ),
        AttemptEvent("d-b0", 6, "b0", "r0", "decision", terminal_status="failed"),
        AttemptEvent("p-a1", 20, "a1", "r1", "proposal"),
        AttemptEvent("p-b1", 21, "b1", "r1", "proposal"),
        AttemptEvent("s-a1", 22, "a1", "capture", "evaluation_started", slots["a1"]),
        AttemptEvent(
            "f-a1",
            23,
            "a1",
            "capture",
            "evaluation_finished",
            slots["a1"],
            costs=(CostEntry(3, "tokens", "estimated"),),
        ),
        AttemptEvent("d-a1", 24, "a1", "r1", "decision", terminal_status="rejected"),
        AttemptEvent("s-b1", 25, "b1", "capture", "evaluation_started", slots["b1"]),
        AttemptEvent("d-b1", 26, "b1", "r1", "decision", terminal_status="selected"),
    )
    return AttemptJournal("journal", record_digest(plan.protocol), events, REFS)


def test_disabled_report_does_not_inspect_inputs():
    assert summarize_family_search(None, None, None) is None


def test_family_report_retains_failures_incomplete_attempts_and_partial_costs():
    p, manifest = fixture()
    journal = journal_fixture(p)
    result = summarize_family_search(p, manifest, journal, config=ENABLED)
    a, b = result["families"]
    assert a["statuses"] == {"a0": "pending", "a1": "rejected"}
    assert b["statuses"] == {"b0": "failed", "b1": "selected"}
    assert a["evaluation_attempt_count"] == 2 and b["evaluation_attempt_count"] == 2
    assert a["completed_evaluations"] == 2 and b["incomplete_evaluations"] == 1
    assert a["cost"]["groups"][0]["known_total"] == 5
    assert a["cost"]["groups"][0]["partial"] is False
    assert b["cost"]["groups"][0]["known_total"] is None
    assert b["cost"]["groups"][0]["unknown_count"] == 1
    assert b["cost"]["unmeasured_attempts"] == 1
    assert len(result["journal"]["events"]) == len(journal.events)
    assert result["promotion_eligible"] is False and result["optimizer_input"] is False
    assert "selected_candidate_id" not in result
    with pytest.raises(ValueError):
        require_training_eligible(result)


def test_unattempted_candidates_and_empty_history_are_not_silently_dropped():
    p, manifest = fixture()
    journal = AttemptJournal("empty", record_digest(p.protocol), (), REFS)
    result = summarize_family_search(p, manifest, journal, config=ENABLED)
    assert [f["planned_unattempted"] for f in result["families"]] == [["a0", "a1"], ["b0", "b1"]]
    assert all(f["proposal_count"] == 0 for f in result["families"])
    assert result["journal"]["evaluation_attempt_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["binding", "proposal_round", "missing_parent", "backward_round", "parent_not_frozen"],
)
def test_journal_cannot_relabel_rounds_or_use_unrecorded_future_parents(mutation):
    p, manifest = fixture()
    events = [
        AttemptEvent("a0", 0, "a0", "r0", "proposal"),
        AttemptEvent("b0", 1, "b0", "r0", "proposal"),
        AttemptEvent("a1", 20, "a1", "r1", "proposal"),
    ]
    digest = record_digest(p.protocol)
    if mutation == "binding":
        digest = "c" * 64
    elif mutation == "proposal_round":
        events[0] = replace(events[0], round_id="other")
    elif mutation == "missing_parent":
        events = events[1:]
    elif mutation == "backward_round":
        events = [events[0], events[2], replace(events[1], ordinal=30)]
    else:
        candidates = (
            replace(p.protocol.candidates[0], freeze_ordinal=6),
            *p.protocol.candidates[1:],
        )
        p = replace(p, protocol=replace(p.protocol, candidates=candidates))
        digest = record_digest(p.protocol)
        events[-1] = replace(events[-1], ordinal=5)
    journal = AttemptJournal("journal", digest, tuple(events), REFS)
    with pytest.raises(ValueError):
        summarize_family_search(p, manifest, journal, config=ENABLED)


def test_cross_family_donor_exposure_uses_the_existing_transitive_independence_check():
    from evaluation.improvement_audit import independence_status, partition_digest
    from evaluation.improvement_records import ExposureRecord

    p, manifest = fixture()
    exposure = ExposureRecord(
        100,
        partition_digest(manifest, "independent_audit"),
        "independent_audit",
        ("a0",),
        (),
        "selection",
        REFS,
    )
    result = independence_status(p.protocol, "b1", "independent_audit", (exposure,))
    assert result["independent"] is False
    assert result["authentication"] == "not_assessed"
    assert (
        independence_status(p.protocol, "b0", "independent_audit", (exposure,))["independent"]
        is True
    )


def test_existing_independent_audit_retains_negative_results_and_donor_contamination():
    """Real ladder captures keep existing authentication and lineage authority unchanged."""
    from evaluation.causal_records import canonical_json
    from evaluation.improvement_audit import audit_improvement, partition_digest
    from evaluation.improvement_records import DiagnosticEvidence, ExposureRecord
    from evaluation.ladder import Metric, Observation, Probe, Severity, evaluate_ladder
    from evaluation.v5_records import SystemIdentity
    from gepa_mindfulness.training.eligibility import TrainingEligibility

    p, manifest = fixture()
    manifest = replace(manifest, dependencies_known=True)
    p = replace(p, protocol=replace(p.protocol, manifest_digest=record_digest(manifest)))
    prepare(p, manifest)
    events = [
        AttemptEvent(
            c.candidate_id,
            0 if i == 0 else 1 if i == 1 else 10 + i,
            c.candidate_id,
            "r0" if i < 2 else "r1",
            "proposal",
        )
        for i, c in enumerate(p.protocol.candidates)
    ]
    evidence = []
    candidates = {c.candidate_id: c for c in p.protocol.candidates}
    for i, slot in enumerate(p.protocol.slots):
        system = getattr(candidates[slot.candidate_id], slot.arm)
        source = evaluate_ladder(
            (
                Probe(
                    "p",
                    Metric.REPRESENTATION_ACCURACY,
                    "cohort",
                    Severity.CONSEQUENTIAL,
                    "fraction",
                ),
            ),
            (Observation("p", slot.arm == "baseline", REFS),),
            protocol_id=slot.slot_id,
            system=SystemIdentity(
                slot.repeat_id, slot.seed, system.model_version, system.harness_version
            ),
            evaluator=EVALUATOR,
            training_eligibility=TrainingEligibility.HIDDEN_EVAL,
            enabled=True,
        )
        evidence.append(
            DiagnosticEvidence(slot.slot_id, canonical_json(source), "rows", "probe_id", "p", REFS)
        )
        events.extend(
            (
                AttemptEvent(
                    f"start:{i}",
                    100 + 2 * i,
                    slot.candidate_id,
                    "capture",
                    "evaluation_started",
                    slot.slot_id,
                ),
                AttemptEvent(
                    f"finish:{i}",
                    101 + 2 * i,
                    slot.candidate_id,
                    "capture",
                    "evaluation_finished",
                    slot.slot_id,
                ),
            )
        )
    journal = AttemptJournal("all", record_digest(p.protocol), tuple(events), REFS)
    assert (
        summarize_family_search(p, manifest, journal, config=ENABLED)["promotion_eligible"] is False
    )
    unauthenticated = audit_improvement(
        p.protocol, manifest, journal, tuple(evidence), enabled=True
    )
    assert unauthenticated["verified_improvement_evidence"] == []
    assert all(
        c["estimate"]["raw_delta"] is None
        for c in unauthenticated["evaluated_behavior"]["comparisons"]
    )
    exposure = ExposureRecord(
        50,
        partition_digest(manifest, "independent_audit"),
        "independent_audit",
        ("a0",),
        (),
        "selection",
        REFS,
    )
    audited = audit_improvement(
        p.protocol,
        manifest,
        journal,
        tuple(evidence),
        exposures=(exposure,),
        authenticate=lambda request: True,
        enabled=True,
    )
    rows = {
        c["candidate_id"]: c
        for c in audited["evaluated_behavior"]["comparisons"]
        if c["purpose"] == "independent_audit"
    }
    assert rows["b1"]["estimate"]["raw_delta"] == -1
    assert rows["b1"]["independent_evidence_eligible"] is False
    assert rows["b0"]["independent_evidence_eligible"] is True
    assert audited["confers_authority"] is False
