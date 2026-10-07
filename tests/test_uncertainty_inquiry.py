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
