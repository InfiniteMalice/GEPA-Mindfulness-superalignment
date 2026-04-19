from gepa_mindfulness.factuality_observability.schemas import CaseOverlayV2, ObservabilityTier


def test_case_overlay_label() -> None:
    overlay = CaseOverlayV2(base_case_label=10, observability_tier=ObservabilityTier.O5)
    assert overlay.final_case_overlay == "Case10-O5"


def test_case_overlay_allows_case0_fallback() -> None:
    overlay = CaseOverlayV2(base_case_label=0, observability_tier=ObservabilityTier.O0)
    assert overlay.final_case_overlay == "Case0-O0"
