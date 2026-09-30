"""A peer packet is released only after the complete verified blind round is frozen."""

import json
from dataclasses import replace

import pytest
from test_temporal_estimator import config, initial, inputs

from gepa_mindfulness.verification.epistemic_reconciliation import (
    EpistemicReconciliation,
    make_epistemic_reconciliation_event,
)
from gepa_mindfulness.verification.epistemic_state import CorrelationTreatment
from gepa_mindfulness.verification.judgment_panel import VerifiedJudgmentPanel
from gepa_mindfulness.verification.temporal_estimator import ScalarTemporalEstimator


def trajectory(index, value=1):
    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, value, index)
    result = estimator.reconcile(events, **kwargs)
    return (*events, result.event), kwargs["measurement"]


def submit(panel, agent, index, value=1):
    events, measurement = trajectory(index, value)
    panel.add_judgment(
        agent,
        events,
        reconciliation_event_id=f"r{index}",
        measurement_id=measurement.measurement_id,
    )
    return events, measurement


def test_complete_verified_round_freezes_dissent_before_peer_release():
    panel = VerifiedJudgmentPanel("panel", ("alice", "bob"))
    submit(panel, "alice", 0, 0)
    with pytest.raises(ValueError, match="complete"):
        panel.open_discussion("2026-09-30T12:01:00Z")
    submit(panel, "bob", 1, 2)
    packet = panel.open_discussion("2026-09-30T12:01:00Z")
    assert packet.participant_ids == ("alice", "bob")
    assert tuple(m.value for m in packet.measurements) == (0, 2)
    assert packet.reconciliation_event_ids == ("r0", "r1")
    payload = json.loads(json.dumps(packet.to_dict()))
    assert payload["released_at"] == "2026-09-30T12:01:00Z"
    object.__setattr__(packet.measurements[0], "value", 100)
    result = panel.fuse_round_zero(estimate_id="round0")
    assert result.estimate.state.values == (1,)
    assert result.estimate.state.variances == (1,)
    with pytest.raises(ValueError, match="sealed"):
        submit(panel, "alice", 2)


@pytest.mark.parametrize("change", ["failed", "unbound", "missing", "wrong-measurement"])
def test_invalid_verification_cannot_fill_a_participant_slot(change):
    panel = VerifiedJudgmentPanel("panel", ("alice",))
    events, measurement = trajectory(0)
    original = events
    measurement_id = measurement.measurement_id
    if change == "failed":
        events = (
            *events[:-2],
            replace(events[-2], payload=dict(events[-2].payload) | {"verified": False}),
            events[-1],
        )
    elif change == "unbound":
        record = EpistemicReconciliation.from_dict(events[-1].payload)
        record = replace(record, bindings=(replace(record.bindings[0], verifier_event_id=None),))
        events = (
            *events[:-1],
            make_epistemic_reconciliation_event(
                record, event_id="r0", timestamp=events[-1].timestamp
            ),
        )
    elif change == "missing":
        events = events[1:]
    else:
        measurement_id = "not-present"
    with pytest.raises(ValueError):
        panel.add_judgment(
            "alice", events, reconciliation_event_id="r0", measurement_id=measurement_id
        )
    with pytest.raises(ValueError):
        panel.open_discussion("2026-09-30T12:01:00Z")
    panel.add_judgment(
        "alice", original, reconciliation_event_id="r0", measurement_id=measurement.measurement_id
    )
    assert panel.open_discussion("2026-09-30T12:01:00Z")


def test_release_chronology_is_checked_and_failure_does_not_seal():
    panel = VerifiedJudgmentPanel("panel", ("alice",))
    submit(panel, "alice", 0)
    with pytest.raises(ValueError, match="timestamp"):
        panel.open_discussion("2026-09-30T11:00:00Z")
    assert panel.open_discussion("2026-09-30T12:01:00Z")


def test_duplicates_and_unknown_participants_fail_without_replacing_prior_judgment():
    panel = VerifiedJudgmentPanel("panel", ("alice", "bob"))
    events, m = submit(panel, "alice", 0, 0)
    with pytest.raises(ValueError):
        submit(panel, "mallory", 1)
    with pytest.raises(ValueError):
        submit(panel, "alice", 1)
    with pytest.raises(ValueError):
        panel.add_judgment(
            "bob", events, reconciliation_event_id="r0", measurement_id=m.measurement_id
        )
    submit(panel, "bob", 1, 2)
    packet = panel.open_discussion("2026-09-30T12:01:00Z")
    assert tuple(m.value for m in packet.measurements) == (0, 2)


def test_shared_history_cannot_be_rewritten_between_submissions():
    panel = VerifiedJudgmentPanel("panel", ("alice", "bob"))
    first, _ = submit(panel, "alice", 0)
    second, m = trajectory(1)
    tampered = (replace(first[0], timestamp="2026-09-30T11:59:59Z"), *first[1:], *second)
    with pytest.raises(ValueError, match="history"):
        panel.add_judgment(
            "bob", tampered, reconciliation_event_id="r1", measurement_id=m.measurement_id
        )
    panel.add_judgment(
        "bob", (*first, *second), reconciliation_event_id="r1", measurement_id=m.measurement_id
    )
    assert panel.open_discussion("2026-09-30T12:01:00Z")


def test_post_discussion_consensus_is_correlated_and_round_zero_is_preserved():
    panel = VerifiedJudgmentPanel("panel", ("alice", "bob"))
    _, first = submit(panel, "alice", 0, 0)
    _, second = submit(panel, "bob", 1, 2)
    peers = (
        replace(first, measurement_id="post-a", value=2),
        replace(second, measurement_id="post-b", value=2),
    )
    with pytest.raises(ValueError):
        panel.fuse_post_discussion(peers, estimate_id="post")
    panel.open_discussion("2026-09-30T12:01:00Z")
    result = panel.fuse_post_discussion(peers, estimate_id="post")
    assert result.peer_exposed is True
    assert result.estimate.state.values == (2,)
    assert result.estimate.state.variances == (1,)
    assert panel.fuse_round_zero(estimate_id="blind").estimate.state.values == (1,)
    with pytest.raises(ValueError, match="dependence"):
        panel.fuse_post_discussion(
            peers,
            estimate_id="post",
            mode=CorrelationTreatment.KNOWN_COVARIANCE,
            covariance=((1, 0), (0, 1)),
            covariance_provenance=("declared",),
        )
    with pytest.raises(ValueError):
        panel.fuse_post_discussion(peers, estimate_id="post", peer_exposed=False)
    with pytest.raises(ValueError):
        panel.fuse_post_discussion((first, second), estimate_id="reused")
    with pytest.raises(ValueError):
        panel.fuse_post_discussion(peers[:1], estimate_id="partial")


@pytest.mark.parametrize(
    "participants", [(), ("a", "a"), ("",), "alice", tuple(map(str, range(17)))]
)
def test_invalid_cohorts_are_rejected(participants):
    with pytest.raises(ValueError):
        VerifiedJudgmentPanel("panel", participants)


@pytest.mark.parametrize("old,new", [("run", "other-run"), ("sensor-units-v1", "other-units")])
def test_incompatible_verified_judgments_cannot_fill_cohort(old, new):
    from mindful_trace_gepa.logging_schema import EventEnvelope

    panel = VerifiedJudgmentPanel("panel", ("alice", "bob"))
    submit(panel, "alice", 0)
    events, m = trajectory(1)
    encoded = json.dumps([e.to_dict() for e in events]).replace(f'"{old}"', f'"{new}"')
    changed = tuple(EventEnvelope(**e) for e in json.loads(encoded))
    with pytest.raises(ValueError, match="context.*target"):
        panel.add_judgment(
            "bob", changed, reconciliation_event_id="r1", measurement_id=m.measurement_id
        )
    submit(panel, "bob", 1)
    assert panel.open_discussion("2026-09-30T12:01:00Z")
