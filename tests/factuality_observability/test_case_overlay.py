from gepa_mindfulness.factuality_observability.schemas import CaseOverlayV2, ObservabilityTier


def test_case_overlay_label() -> None:
    overlay = CaseOverlayV2(base_case_label=10, observability_tier=ObservabilityTier.O5)
    assert overlay.final_case_overlay == "Case10-O5"


def test_case_overlay_allows_case0_fallback() -> None:
    overlay = CaseOverlayV2(base_case_label=0, observability_tier=ObservabilityTier.O0)
    assert overlay.final_case_overlay == "Case0-O0"


def test_case_overlay_rejects_boolean_case_label() -> None:
    try:
        CaseOverlayV2(base_case_label=True)
    except TypeError as exc:
        assert "base_case_label must be int, not bool or non-int" in str(exc)
        assert "final_case_overlay" in str(exc)
    else:
        raise AssertionError("boolean case labels should fail validation")
