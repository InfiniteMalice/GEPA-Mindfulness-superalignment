"""Fresh independent judgments cannot launder observations or inaccessible support."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from artifact_fixtures import make_assessment, make_capture, make_protocol

# Local
from evaluation.evidence_topology import analyze_evidence_topology


def analyze(p, c, a=None, **kw):
    return analyze_evidence_topology(
        p,
        c,
        assessment=a,
        enabled=True,
        authorize=kw.pop("authorize", lambda _: True),
        authenticate=kw.pop("authenticate", lambda _: True),
        **kw,
    )


@pytest.mark.parametrize("mode", ["absent", "false", "integer", "mutation", "exception", "human"])
def test_fresh_authentication(mode):
    p = make_protocol()
    c = make_capture(p)
    a = make_assessment(p, c)

    def mutate(x):
        object.__setattr__(x, "reason", "changed")
        return True

    def fail(x):
        raise RuntimeError("secret")

    cb = {
        "absent": None,
        "false": lambda _: False,
        "integer": lambda _: 1,
        "mutation": mutate,
        "exception": fail,
        "human": lambda _: True,
    }[mode]
    if mode == "human":
        a = replace(a, human_required=True)
    result = analyze(p, c, a, authenticate=cb)
    assert result["metrics"]["correctness"]["rate"] is None
    assert result["supported_claims"]["goal"]["status"] == "unresolved"
    assert a.reason != "changed"
    assert analyze(p, c, a)["source_claims"] == p.snapshot.state.to_dict()


@pytest.mark.parametrize("kind", ["missing", "censored", "unknown", "ineligible"])
def test_missingness_and_denominators(kind):
    p = make_protocol()
    c = make_capture(p)
    if kind == "missing":
        c = None
    elif kind == "censored":
        c = replace(
            c,
            status="censored",
            response=None,
            actions=(),
            retrieved_item_ids=(),
            attributed_item_ids=(),
        )
    a = make_assessment(p, c)
    if kind in ("unknown", "ineligible"):
        a = replace(
            a,
            verdicts=tuple(
                replace(v, value=None, eligible=False if kind == "ineligible" else True)
                for v in a.verdicts
            ),
        )
    out = analyze(p, c, a)
    metric = out["metrics"]["correctness"]
    assert metric["denominator"] == (0 if kind == "ineligible" else 1)
    assert metric["rate"] is None
    status = "unresolved" if kind == "unknown" else kind
    assert metric[status] == 1
    assert metric[status + "_ids"] == ["op:correctness"]
    assert out["verification_coverage"]["numerator"] == (6 if kind == "ineligible" else 0)
    assert len(out["cases"]) == 17
    empty = analyze(replace(p, opportunities=()), None)
    assert empty["metrics"]["correctness"]["rate"] is None


def test_missing_inapplicable_coverage_and_cohorts():
    p = make_protocol()
    p = replace(p, opportunities=tuple(replace(o, cohort=o.metric) for o in p.opportunities))
    a = make_assessment(p, None)
    a = replace(a, verdicts=tuple(replace(v, eligible=False, value=None) for v in a.verdicts))
    out = analyze(p, None, a)
    assert out["verification_coverage"]["rate"] == 1
    assert len(out["groups"]) == 6


def test_claim_support_requires_access_and_resolution():
    p = make_protocol()
    c = make_capture(p)
    a = make_assessment(p, c)
    assert analyze(p, c, a)["supported_claims"]["goal"]["status"] == "supported"
    denied = analyze(p, c, a, authorize=lambda _: False)
    assert denied["supported_claims"]["goal"]["status"] == "blocked"
    assert denied["producer_view"]["items"] == []
    assert denied["observed_access_violations"] == ["a", "b"]
    assert denied["metrics"]["correctness"]["rate"] == 1
    a = replace(a, claim_verdicts=(replace(a.claim_verdicts[0], contradictions_resolved=False),))
    assert analyze(p, c, a)["supported_claims"]["goal"]["status"] == "unsupported"


def test_source_state_is_unchanged():
    p = make_protocol()
    original = p.to_dict()
    c = make_capture(p)
    a = make_assessment(p, c)
    first = analyze(p, c, a)
    second = analyze(p, c, a, authorize=lambda _: False, authenticate=lambda _: False)
    assert first["source_claims"] == second["source_claims"] == p.snapshot.state.to_dict()
    assert p.to_dict() == original
    assert first["optimizer_input"] is False and first["confers_authority"] is False
    with pytest.raises(ValueError):
        analyze_evidence_topology(p, c)
