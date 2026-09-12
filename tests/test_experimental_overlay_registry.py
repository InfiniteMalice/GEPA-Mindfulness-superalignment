from __future__ import annotations

import importlib.resources
from dataclasses import FrozenInstanceError
from typing import Any, Callable

import pytest

from evaluation import experimental_overlays, recommendations

EXPECTED_OVERLAYS = (
    (
        "competing_hypotheses",
        "Competing hypotheses",
        "competing_hypotheses",
        ("hypothesis_set",),
        ("REF-PEARL",),
        ("REC-011",),
    ),
    (
        "expected_information_gain_inquiry",
        "Expected information-gain inquiry",
        "expected_information_gain_inquiry",
        ("information_gain_question",),
        ("REF-PEARL",),
        ("REC-011",),
    ),
    (
        "adaptive_small_multi_agent_topology",
        "Adaptive small multi-agent topology",
        "adaptive_small_multi_agent_topology",
        ("topology_proposal",),
        ("REF-MASKILLS",),
        ("REC-012",),
    ),
    (
        "declarative_orchestration_scope",
        "Declarative orchestration scope",
        "declarative_orchestration_scope",
        ("orchestration_scope_declaration",),
        ("REF-AGENTSCOPE",),
        ("REC-013",),
    ),
    (
        "mechanistic_circuit_audit",
        "Mechanistic circuit audit",
        "mechanistic_circuit_audit",
        ("mechanistic_audit_reference",),
        ("REF-SAE",),
        ("REC-014",),
    ),
)


def test_registry_exposes_exact_disabled_experimental_overlays() -> None:
    loaded = experimental_overlays.load_experimental_overlay_registry()

    assert (
        tuple(
            (
                item.id,
                item.title,
                item.feature_flag,
                item.allowed_outputs,
                item.research_refs,
                item.recommendation_refs,
            )
            for item in loaded
        )
        == EXPECTED_OVERLAYS
    )
    assert all(item.maturity == "experimental" for item in loaded)
    assert all(item.enabled_by_default is False for item in loaded)
    assert len({item.feature_flag for item in loaded}) == 5


def test_every_overlay_prohibits_case_creation_and_direct_reward() -> None:
    loaded = experimental_overlays.load_experimental_overlay_registry()

    for overlay in loaded:
        assert "canonical_case_creation" in overlay.prohibited_effects
        assert "direct_optimizer_reward" in overlay.prohibited_effects
    mechanistic = next(item for item in loaded if item.id == "mechanistic_circuit_audit")
    assert "correlation_as_causation" in mechanistic.prohibited_effects


def test_registry_links_resolve_to_authored_recommendations_and_research() -> None:
    overlays = experimental_overlays.load_experimental_overlay_registry()
    recommendation_ids = {
        item.recommendation_id for item in recommendations.load_recommendation_registry()
    }
    reference_ids = {
        item.reference_id for item in recommendations.load_research_reference_registry()
    }

    assert {
        item for overlay in overlays for item in overlay.recommendation_refs
    } <= recommendation_ids
    assert {item for overlay in overlays for item in overlay.research_refs} <= reference_ids


def test_experimental_recommendations_name_pr7_code_docs_and_tests() -> None:
    expected_implementation = (
        "evaluation/experimental_overlays.py",
        "evaluation/experimental_records.py",
        "docs/experimental_v5_overlays.md",
    )
    expected_tests = (
        "tests/test_experimental_overlay_registry.py",
        "tests/test_experimental_overlay_flags.py",
        "tests/test_experimental_overlay_records.py",
    )
    by_id = {
        item.recommendation_id: item for item in recommendations.load_recommendation_registry()
    }

    for recommendation_id in ("REC-011", "REC-012", "REC-013", "REC-014"):
        recommendation = by_id[recommendation_id]
        assert recommendation.status == "experimental"
        assert recommendation.targets == (
            "evaluation/experimental_overlays.py",
            "evaluation/experimental_records.py",
        )
        assert recommendation.implementation_refs == expected_implementation
        assert recommendation.acceptance_tests == expected_tests


def test_overlay_records_are_frozen_and_resource_is_packaged() -> None:
    overlay = experimental_overlays.load_experimental_overlay_registry()[0]

    with pytest.raises(FrozenInstanceError):
        overlay.title = "changed"  # type: ignore[misc]
    assert (
        importlib.resources.files("evaluation.cases")
        .joinpath("experimental_overlays.yaml")
        .is_file()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(version="v5"), "registry_version"),
        (lambda payload: payload.update(extra=True), "registry fields"),
        (lambda payload: payload["overlays"][0].pop("title"), "overlay fields"),
        (lambda payload: payload["overlays"][0].update(maturity="stable"), "maturity"),
        (
            lambda payload: payload["overlays"][0].update(enabled_by_default=True),
            "enabled_by_default",
        ),
        (
            lambda payload: payload["overlays"][0].update(prohibited_effects=[]),
            "prohibited_effects",
        ),
        (
            lambda payload: payload["overlays"][0].update(research_refs=["REF-UNKNOWN"]),
            "unknown research",
        ),
        (
            lambda payload: payload["overlays"][0].update(recommendation_refs=["REC-999"]),
            "unknown recommendation",
        ),
    ],
)
def test_parser_rejects_malformed_or_unresolved_registry_entries(
    mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    payload = _valid_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        experimental_overlays._parse_experimental_overlay_registry(payload)


def test_parser_rejects_duplicate_ids_flags_and_outputs() -> None:
    payload = _valid_payload()
    duplicate = dict(payload["overlays"][0])
    payload["overlays"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate overlay IDs.*feature flags"):
        experimental_overlays._parse_experimental_overlay_registry(payload)

    payload = _valid_payload()
    payload["overlays"][0]["allowed_outputs"] = ["hypothesis_set", "hypothesis_set"]
    with pytest.raises(ValueError, match="allowed_outputs must be unique"):
        experimental_overlays._parse_experimental_overlay_registry(payload)


def test_parser_rejects_cross_wired_flags_and_output_kinds() -> None:
    payload = _valid_payload()
    payload["overlays"][0]["feature_flag"] = "expected_information_gain_inquiry"
    payload["overlays"][1]["feature_flag"] = "competing_hypotheses"

    with pytest.raises(ValueError, match="feature_flag must match overlay id"):
        experimental_overlays._parse_experimental_overlay_registry(payload)

    payload = _valid_payload()
    payload["overlays"][0]["allowed_outputs"] = ["topology_proposal"]
    with pytest.raises(ValueError, match="allowed_outputs do not match overlay"):
        experimental_overlays._parse_experimental_overlay_registry(payload)


def test_parser_rejects_cross_wired_known_traceability_ids() -> None:
    payload = _valid_payload()
    payload["overlays"][0]["research_refs"] = ["REF-SAE"]

    with pytest.raises(ValueError, match="research_refs do not match overlay"):
        experimental_overlays._parse_experimental_overlay_registry(payload)

    payload = _valid_payload()
    payload["overlays"][0]["recommendation_refs"] = ["REC-014"]
    with pytest.raises(ValueError, match="recommendation_refs do not match overlay"):
        experimental_overlays._parse_experimental_overlay_registry(payload)


def _valid_payload() -> dict[str, Any]:
    overlays = []
    for identifier, title, flag, outputs, research_refs, recommendation_refs in EXPECTED_OVERLAYS:
        prohibited = ["canonical_case_creation", "direct_optimizer_reward"]
        if identifier == "mechanistic_circuit_audit":
            prohibited.append("correlation_as_causation")
        overlays.append(
            {
                "id": identifier,
                "title": title,
                "maturity": "experimental",
                "enabled_by_default": False,
                "feature_flag": flag,
                "allowed_outputs": list(outputs),
                "prohibited_effects": prohibited,
                "research_refs": list(research_refs),
                "recommendation_refs": list(recommendation_refs),
            }
        )
    return {"registry_version": "17case-v5", "overlays": overlays}
