"""Inquiry selects discriminating checks and retains the existing causal estimator gate."""

from dataclasses import replace

import pytest
from test_temporal_estimator import config, initial, inputs

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.check_records import CheckRequest, CheckResult
from gepa_mindfulness.verification.temporal_estimator import ScalarTemporalEstimator


def test_discriminating_check_beats_irrelevant_and_detects_early_stop() -> None:
    from gepa_mindfulness.verification.uncertainty_inquiry import plan_inquiry

    ref = EvidenceReference("task", EvidenceSourceKind.EXTERNAL_RECORD)
    useful = CheckRequest("useful", "c", "discrimination", "measure", 0.1, 1, 1, 1, (ref,), "a")
    irrelevant = replace(useful, check_id="irrelevant", expected_information_gain=1)
    hypotheses = {"useful": {"h1": "hot", "h2": "cold"}, "irrelevant": {"h1": "red", "h2": "red"}}
    result = plan_inquiry(
        (irrelevant, useful),
        hypotheses,
        budget=1,
        unresolved_claims=("c",),
        requested_stop=True,
        enabled=True,
    )
    assert result["selected_check_ids"] == ("useful",)
    assert result["premature_stop"] is True
    assert result["information_gain_proxy"]["irrelevant"] == 0
    assert result["unresolved_claims"] == ("c",)
    assert plan_inquiry((useful,), {}, budget=1)["selected_check_ids"] == ()


def test_inquiry_reconciles_through_existing_temporal_estimator() -> None:
    from gepa_mindfulness.verification.uncertainty_inquiry import reconcile_inquiry

    estimator = ScalarTemporalEstimator(initial(), config())
    events, kwargs = inputs(estimator, 0.5)
    refs = kwargs["measurement"].evidence_refs
    check = CheckRequest("measure", "c", "discrimination", "measure", 1, 1, 1, 1, refs, "action-0")
    result = CheckResult("measure", "c", "action-0", "supported", refs, "verifier", None)
    update = reconcile_inquiry(estimator, events, check, result, enabled=True, **kwargs)
    assert update is not None
    assert estimator.estimate.state.values[0] == pytest.approx(1 / 3)
    assert estimator.estimate.evidence_refs == (
        EvidenceReference("initial-evidence", EvidenceSourceKind.EXTERNAL_RECORD),
        *refs,
    )
    with pytest.raises(ValueError):
        reconcile_inquiry(estimator, events, check, result, enabled=True, **kwargs)
    fresh = ScalarTemporalEstimator(initial(), config())
    with pytest.raises(ValueError):
        reconcile_inquiry(
            fresh, events, check, replace(result, verdict="unresolved"), enabled=True, **kwargs
        )
    with pytest.raises(ValueError):
        reconcile_inquiry(fresh, tuple(reversed(events)), check, result, enabled=True, **kwargs)


@pytest.mark.parametrize("unresolved", [("open",), ()])
def test_inquiry_does_not_spend_budget_on_resolved_claims(unresolved) -> None:
    from gepa_mindfulness.verification.uncertainty_inquiry import plan_inquiry

    ref = EvidenceReference("task", EvidenceSourceKind.EXTERNAL_RECORD)
    resolved = CheckRequest(
        "resolved", "closed", "discrimination", "measure", 1, 1, 1, 1, (ref,), "a"
    )
    open_check = replace(resolved, check_id="open", claim_id="open", verification_cost=2)
    predictions = {key: {"h1": "one", "h2": "two"} for key in ("resolved", "open")}
    result = plan_inquiry(
        (resolved, open_check),
        predictions,
        budget=1,
        unresolved_claims=unresolved,
        requested_stop=True,
        enabled=True,
    )
    assert result["selected_check_ids"] == ()
    assert result["predicted_cost"] == 0
    assert result["premature_stop"] is False
    assert result["budget_exhausted"] is bool(unresolved)
    affordable = plan_inquiry(
        (resolved, open_check),
        predictions,
        budget=2,
        unresolved_claims=unresolved,
        requested_stop=True,
        enabled=True,
    )
    assert affordable["selected_check_ids"] == (("open",) if unresolved else ())
    assert affordable["budget_exhausted"] is False


def test_budget_exhaustion_distinguishes_useless_checks_from_unaffordable_checks() -> None:
    from gepa_mindfulness.verification.uncertainty_inquiry import plan_inquiry

    ref = EvidenceReference("task", EvidenceSourceKind.EXTERNAL_RECORD)
    check = CheckRequest("check", "c", "discrimination", "measure", 1, 1, 1, 1, (ref,), "a")
    same = plan_inquiry(
        (check,),
        {"check": {"h1": "same", "h2": "same"}},
        budget=10,
        unresolved_claims=("c",),
        enabled=True,
    )
    assert same["budget_exhausted"] is False
    expensive = replace(check, check_id="expensive", verification_cost=2)
    partial = plan_inquiry(
        (check, expensive),
        {key: {"h1": "one", "h2": "two"} for key in ("check", "expensive")},
        budget=1,
        unresolved_claims=("c",),
        enabled=True,
    )
    assert partial["selected_check_ids"] == ("check",)
    assert partial["budget_exhausted"] is True
