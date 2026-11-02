"""Participation advice cannot certify expertise, eligibility or runtime authority."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.improvement_records import SystemConfig, record_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.factuality_observability.decision_routing import PublicDecisionContext
from gepa_mindfulness.factuality_observability.participation_records import (
    ParticipationConfig,
    ParticipationPacket,
    ParticipationReview,
    SpecialistBid,
    SpecialistCandidate,
)
from gepa_mindfulness.factuality_observability.specialist_participation import select_specialist
from gepa_mindfulness.training.eligibility import require_training_eligible

REFS = (EvidenceReference("public:record", EvidenceSourceKind.EXTERNAL_RECORD),)
EVALUATOR = TrustedEvaluatorContract("host-reviewer", "v1", "eligibility-v1")
STATE = "1" * 64


def candidate(name="a"):
    return SpecialistCandidate(
        name,
        f"principal:{name}",
        SystemConfig(f"model:{name}", "harness-v1", "2" * 64),
        "3" * 64,
        "Declared specialty, not proof of competence",
        REFS,
    )


def packet(candidates=None):
    return ParticipationPacket(
        "request-1",
        PublicDecisionContext(STATE, "Check the evidence", "Public artifact", REFS),
        "advisory-review",
        candidates if candidates is not None else (candidate("a"), candidate("b")),
    )


def review_value(p, c, status="eligible"):
    return ParticipationReview(
        record_digest(p),
        record_digest(c),
        EVALUATOR,
        "principal:reviewer",
        f"review:{c.specialist_id}",
        status,
        "Host supplied eligibility assessment",
        REFS,
    )


def bid_value(p, c, value=0.4):
    return SpecialistBid(record_digest(p), record_digest(c), c.specialist_id, c.system, value, REFS)


def test_versioned_records_roundtrip_and_refuse_authority_extensions():
    p = packet()
    values = (ParticipationConfig(), p, p.candidates[0], review_value(p, p.candidates[0]))
    for value in (*values, bid_value(p, p.candidates[0])):
        wire = value.to_dict()
        assert type(value).from_dict(wire) == value
        assert wire["training_eligibility"] == "DEVELOPMENT"
        assert wire["optimizer_input"] is False and wire["confers_authority"] is False
        for change in (
            {"confers_authority": True},
            {"optimizer_input": True},
            {"training_eligibility": "TRAIN"},
            {"schema_version": "future-v99"},
            {"authorization": "self-certified"},
        ):
            with pytest.raises(ValueError):
                type(value).from_dict({**wire, **change})


@pytest.mark.parametrize("bad", [True, -0.01, 1.01, float("nan"), float("inf"), 10**1000])
def test_invalid_bids_cannot_enter_selection(bad):
    p = packet()
    with pytest.raises(ValueError):
        bid_value(p, p.candidates[0], bad)


def test_packet_rejects_duplicate_and_case_alias_principals_and_specialists():
    a = candidate()
    for other in (a, replace(candidate("b"), principal_id="PRINCIPAL:A")):
        with pytest.raises(ValueError):
            packet((a, other))
    with pytest.raises(ValueError):
        packet((a, replace(candidate("b"), specialist_id="A")))
    with pytest.raises(ValueError):
        packet(())
    with pytest.raises(ValueError):
        packet(tuple(candidate(f"s{i}") for i in range(17)))


def test_nonpublic_evidence_and_unknown_review_status_are_rejected():
    p = packet()
    with pytest.raises(ValueError):
        replace(
            candidate(),
            evidence_refs=(
                EvidenceReference("private:thought", EvidenceSourceKind.PRIVATE_REASONING),
            ),
        )
    with pytest.raises(ValueError):
        review_value(p, p.candidates[0], "self-certified")


@pytest.mark.parametrize("field,value", [("candidates", {}), ("candidates", "ab")])
def test_serialized_arrays_are_not_coerced_from_other_types(field, value):
    wire = packet().to_dict()
    with pytest.raises(ValueError):
        ParticipationPacket.from_dict({**wire, field: value})


@pytest.mark.parametrize("change", [{"enabled": 1}, {"seed": True}, {"max_output_bytes": 4095}])
def test_config_requires_explicit_bounded_operator_settings(change):
    with pytest.raises(ValueError):
        replace(ParticipationConfig(), **change)


def run(*, p=None, review=review_value, bid=bid_value, config=None, state=STATE, principal=None):
    return select_specialist(
        p or packet(),
        current_state_digest=state,
        evaluator=EVALUATOR,
        reviewer_principal_id=principal or "principal:reviewer",
        review=review,
        propose_bid=bid,
        config=config or ParticipationConfig(enabled=True),
    )


def test_default_off_does_not_inspect_private_inputs_or_call_callbacks():
    assert (
        select_specialist(
            None,
            current_state_digest=None,
            evaluator=None,
            reviewer_principal_id=None,
            review=None,
            propose_bid=None,
        )
        is None
    )


def test_all_reviews_precede_bids_and_ineligible_highest_bidder_is_quarantined():
    seen = []

    def review(p, c):
        seen.append(f"review:{c.specialist_id}")
        return review_value(p, c, "rejected" if c.specialist_id == "a" else "eligible")

    def bid(p, c):
        seen.append(f"bid:{c.specialist_id}")
        return bid_value(p, c, 0.2)

    result = run(review=review, bid=bid)
    assert seen == ["review:a", "review:b", "bid:b"]
    assert result["selected_specialist_id"] == "b"
    assert result["rows"][0]["review"]["status"] == "rejected"
    assert result["rows"][0]["bid"] is None
    assert result["route"] == "suggest_specialist"
    assert result["authority_granted"] is False and result["execute_specialist"] is False
    assert result["optimizer_input"] is False and result["model_effectiveness_established"] is False
    with pytest.raises(ValueError):
        require_training_eligible(result)


@pytest.mark.parametrize(
    "status,route",
    [
        ("uncertain", "further_verification"),
        ("rejected", "no_participation"),
    ],
)
def test_unresolved_eligibility_is_not_scored_as_a_rule_violation(status, route):
    result = run(
        review=lambda p, c: review_value(p, c, status),
        bid=lambda p, c: pytest.fail("quarantined candidate reached bidder"),
    )
    assert result["route"] == route and result["selected_specialist_id"] is None
    assert all(row["review"]["status"] == status for row in result["rows"])


def test_all_zero_bids_abstain_without_floor_reopening():
    result = run(bid=lambda p, c: bid_value(p, c, 0))
    assert result["route"] == "no_participation"
    assert result["selected_specialist_id"] is None and result["tied_specialist_ids"] == []


def test_seeded_exact_ties_are_reproducible_and_independent_of_roster_order():
    for seed, expected in ((0, "b"), (1, "a")):
        cfg = ParticipationConfig(enabled=True, seed=seed)
        assert run(config=cfg)["selected_specialist_id"] == expected
        assert (
            run(config=cfg, p=packet((candidate("b"), candidate("a"))))["selected_specialist_id"]
            == expected
        )
    assert run()["tied_specialist_ids"] == ["a", "b"]


@pytest.mark.parametrize("low,high", [(0.4, 0.4000000000000001), (0, 5e-324)])
def test_near_ties_and_tiny_positive_bids_keep_exact_order(low, high):
    result = run(bid=lambda p, c: bid_value(p, c, low if c.specialist_id == "a" else high))
    assert result["selected_specialist_id"] == "b"
    assert result["tied_specialist_ids"] == ["b"]


@pytest.mark.parametrize("state,principal", [("0" * 64, None), (STATE, "PRINCIPAL:A")])
def test_stale_state_and_self_reviewer_fail_before_any_callback(state, principal):
    with pytest.raises(ValueError):
        run(
            state=state,
            principal=principal,
            review=lambda p, c: pytest.fail("invalid request reached reviewer"),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"packet_digest": "0" * 64},
        {"candidate_digest": "0" * 64},
        {"evaluator": TrustedEvaluatorContract("other", "v1", "eligibility-v1")},
        {"reviewer_principal_id": "principal:a"},
    ],
)
def test_review_substitution_aborts_selection_without_requesting_bids(change):
    result = run(
        review=lambda p, c: replace(review_value(p, c), **change),
        bid=lambda p, c: pytest.fail("misbound review reached bidder"),
    )
    assert result["route"] == "further_verification"
    assert result["selected_specialist_id"] is None
    assert result["failure"]["reason"] == "binding_mismatch"


@pytest.mark.parametrize(
    "change",
    [
        {"packet_digest": "0" * 64},
        {"candidate_digest": "0" * 64},
        {"specialist_id": "unknown"},
        {"system": SystemConfig("wrong", "h", "0" * 64)},
    ],
)
def test_changed_bid_identity_or_system_aborts_selection(change):
    result = run(bid=lambda p, c: replace(bid_value(p, c), **change))
    assert result["selected_specialist_id"] is None
    assert result["failure"]["reason"] == "binding_mismatch"


@pytest.mark.parametrize("stage", ["review", "bid"])
def test_mutated_detached_inputs_are_detected_without_changing_original(stage):
    original = packet()

    def mutate(p, c):
        value = review_value(p, c) if stage == "review" else bid_value(p, c)
        object.__setattr__(p.context, "artifact", "UNTRUSTED_REWRITE")
        object.__setattr__(c, "description", "UNTRUSTED_REWRITE")
        return value

    result = run(p=original, **{stage: mutate})
    assert result["selected_specialist_id"] is None
    assert result["failure"]["reason"] == "mutated_input"
    assert original.context.artifact == "Public artifact"
    assert result["packet"]["context"]["artifact"] == "Public artifact"
    assert "UNTRUSTED_REWRITE" not in str(result)


@pytest.mark.parametrize("stage", ["review", "bid"])
def test_callback_errors_and_malformed_results_fail_closed_without_exception_text(stage):
    def fail(p, c):
        raise RuntimeError("UNTRUSTED_SECRET_OR_INSTRUCTION")

    for callback, reason in ((fail, "callback_error"), (lambda p, c: True, "invalid_record")):
        result = run(**{stage: callback})
        assert result["selected_specialist_id"] is None
        assert result["failure"]["reason"] == reason
        assert "UNTRUSTED_SECRET_OR_INSTRUCTION" not in str(result)


def test_late_bid_failure_cannot_promote_an_earlier_candidate():
    def bid(p, c):
        if c.specialist_id == "b":
            raise RuntimeError("failure")
        return bid_value(p, c, 0.9)

    result = run(bid=bid)
    assert result["rows"][0]["bid"]["value"] == 0.9
    assert result["selected_specialist_id"] is None
    assert result["route"] == "further_verification"


def test_duplicate_review_ids_abort_before_bidding():
    result = run(review=lambda p, c: replace(review_value(p, c), review_id="same"))
    assert result["selected_specialist_id"] is None
    assert result["failure"]["reason"] == "duplicate_review_id"
    assert all(row["bid"] is None for row in result["rows"])


def test_complete_report_capacity_is_reserved_before_either_callback():
    with pytest.raises(ValueError, match="budget|capacity"):
        run(
            config=ParticipationConfig(enabled=True, max_output_bytes=4096),
            review=lambda p, c: pytest.fail("review ran before output reservation"),
            bid=lambda p, c: pytest.fail("bid ran before output reservation"),
        )


def test_returned_records_are_detached_before_later_callbacks():
    retained = []

    def bid(p, c):
        if retained:
            object.__setattr__(retained[0], "value", 0)
        value = bid_value(p, c, 0.9 if c.specialist_id == "a" else 0.2)
        retained.append(value)
        return value

    result = run(bid=bid)
    assert result["selected_specialist_id"] == "a"
    assert result["rows"][0]["bid"]["value"] == 0.9
