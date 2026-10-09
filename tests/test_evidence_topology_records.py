"""Strict topology receipts, capture measurements and complete content binding."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from artifact_fixtures import make_assessment, make_capture, make_protocol

# Local
from evaluation.evidence_topology import analyze_evidence_topology
from evaluation.evidence_topology_records import topology_protocol_digest


def test_topology_record_roundtrips():
    p = make_protocol()
    c = make_capture(p)
    a = make_assessment(p, c)
    for record in (p, p.opportunities[0], c, a, a.claim_verdicts[0]):
        assert type(record).from_dict(record.to_dict()) == record
        bad = record.to_dict()
        bad["authority"] = True
        with pytest.raises(ValueError):
            type(record).from_dict(bad)


@pytest.mark.parametrize(
    "change", ["rubric", "evaluator", "subject", "query", "snapshot", "capture"]
)
def test_receipt_content_binding(change):
    p = make_protocol()
    c = make_capture(p)
    a = make_assessment(p, c)
    if change == "rubric":
        p = replace(p, rubric_id="changed")
    elif change == "evaluator":
        p = replace(p, evaluator=replace(p.evaluator, evaluator_id="other"))
    elif change == "subject":
        p = replace(p, subject=replace(p.subject, expected_actions=("different",)))
    elif change == "query":
        p = replace(p, query=replace(p.query, principal_id="other"))
    elif change == "snapshot":
        from artifact_fixtures import make_topology

        s = replace(
            p.snapshot,
            artifacts=tuple(replace(x, availability="restricted") for x in p.snapshot.artifacts),
        )
        p = replace(p, snapshot=s, topology=make_topology(s, "redundant"))
    else:
        c = replace(c, response="different")
    with pytest.raises(ValueError):
        analyze_evidence_topology(p, c, assessment=a, enabled=True)


def test_query_binds_public_turns():
    p = make_protocol()
    with pytest.raises(ValueError):
        replace(p, query=replace(p.query, public_query="secret expected answer"))
    assert topology_protocol_digest(p) != topology_protocol_digest(replace(p, protocol_id="new"))
    with pytest.raises(ValueError):
        replace(p, opportunities=p.opportunities + (p.opportunities[0],))


@pytest.mark.parametrize(
    "field,bad",
    [
        ("latency_seconds", -1),
        ("latency_seconds", True),
        ("latency_seconds", float("nan")),
        ("retrieval_cost", float("inf")),
        ("retrieval_cost", -0.1),
        ("retrieval_cost", True),
        ("cost_unit", ""),
        ("cost_unit", None),
        ("retrieved_item_ids", ("a", "a")),
        ("attributed_item_ids", ("foreign",)),
        ("response", ""),
        ("evidence_refs", ()),
    ],
)
def test_capture_measurements(field, bad):
    c = make_capture(make_protocol())
    with pytest.raises(ValueError):
        replace(c, **{field: bad})


def test_capture_zero_missing_and_censored():
    c = make_capture(make_protocol())
    assert replace(c, retrieval_cost=0.0).retrieval_cost == 0.0
    assert replace(c, retrieval_cost=None, cost_unit=None).retrieval_cost is None
    with pytest.raises(ValueError):
        replace(c, status="censored")
    censored = replace(
        c,
        status="censored",
        response=None,
        actions=(),
        retrieved_item_ids=(),
        attributed_item_ids=(),
    )
    assert censored.status == "censored"
    with pytest.raises(ValueError):
        analyze_evidence_topology(
            make_protocol(),
            replace(c, retrieved_item_ids=("foreign",), attributed_item_ids=()),
            enabled=True,
        )
