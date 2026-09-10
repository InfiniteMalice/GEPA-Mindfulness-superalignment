"""Behavioral contracts for the V5 recommendation registry."""

from __future__ import annotations

import importlib.resources
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import pytest

from evaluation import recommendations

EXPECTED_RECOMMENDATIONS = (
    (
        "REC-001",
        "Verified epistemic-process reward rule.",
        "P0",
        "implemented",
    ),
    (
        "REC-002",
        "Action-bound epistemic commitments.",
        "P0",
        "accepted",
    ),
    (
        "REC-003",
        "One canonical 17-case V5 manifest.",
        "P0",
        "implemented",
    ),
    (
        "REC-004",
        "Versioned recommendation/decision registry.",
        "P0",
        "implemented",
    ),
    (
        "REC-005",
        "Case × robustness stripe × repeat evaluation.",
        "P0",
        "accepted",
    ),
    (
        "REC-006",
        "World/artifact state != evidence/belief state.",
        "P1",
        "accepted",
    ),
    (
        "REC-007",
        "Structured failure graph.",
        "P1",
        "accepted",
    ),
    (
        "REC-008",
        "Runtime Planner/Executor/Verifier authority separation.",
        "P1",
        "accepted",
    ),
    (
        "REC-009",
        "Verified skill lifecycle.",
        "P1",
        "accepted",
    ),
    (
        "REC-010",
        "Online experience collection; offline harness/skill evolution.",
        "P1",
        "accepted",
    ),
    (
        "REC-011",
        "Multiple competing hypotheses + information-gain inquiry.",
        "P2",
        "experimental",
    ),
    (
        "REC-012",
        "Small adaptive multi-agent topology codebook.",
        "P2",
        "experimental",
    ),
    (
        "REC-013",
        "Declarative global/focus/local orchestration scope.",
        "P2",
        "experimental",
    ),
    (
        "REC-014",
        "Mechanistic/circuit audit of actual model changes.",
        "P2",
        "experimental",
    ),
)


def test_bundled_registry_has_the_approved_ordered_recommendations() -> None:
    loaded = recommendations.load_recommendation_registry()

    assert (
        tuple((item.recommendation_id, item.title, item.priority, item.status) for item in loaded)
        == EXPECTED_RECOMMENDATIONS
    )
    assert all(item.rationale for item in loaded)
    assert all(item.targets for item in loaded)
    assert all(item.repo_refs for item in loaded)
    assert all(item.acceptance_tests for item in loaded)
    assert all(isinstance(item.targets, tuple) for item in loaded)


def test_recommendation_records_are_frozen() -> None:
    recommendation = recommendations.load_recommendation_registry()[0]

    with pytest.raises(FrozenInstanceError):
        recommendation.status = "accepted"


def test_registry_loads_as_a_package_resource_outside_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    loaded = recommendations.load_recommendation_registry()
    package_files = importlib.resources.files("docs.recommendations")

    assert len(loaded) == 14
    assert package_files.joinpath("registry.yaml").is_file()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"registry_version": "17case-v5"}, "recommendation registry fields"),
        (
            {
                "registry_version": "17case-v5",
                "recommendations": [],
                "unexpected": True,
            },
            "unknown fields",
        ),
        ({"registry_version": "v5", "recommendations": []}, "registry_version"),
    ],
)
def test_loader_rejects_invalid_document_roots(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        recommendations._parse_recommendation_registry(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["recommendations"][0].pop("title"), "REC-001 fields"),
        (lambda payload: payload["recommendations"][0].update(extra="value"), "unknown fields"),
        (lambda payload: payload["recommendations"][0].update(priority="P3"), "priority"),
        (lambda payload: payload["recommendations"][0].update(status="pending"), "status"),
        (lambda payload: payload["recommendations"][0].update(rationale=""), "rationale"),
        (lambda payload: payload["recommendations"][0].update(targets=[]), "targets"),
        (
            lambda payload: payload["recommendations"][0].update(acceptance_tests=[]),
            "acceptance_tests",
        ),
    ],
)
def test_loader_rejects_invalid_recommendation_fields(
    mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    payload = _valid_registry_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        recommendations._parse_recommendation_registry(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["recommendations"][1].update(id="REC-001"), "duplicate IDs"),
        (lambda payload: payload["recommendations"][0].update(id="REC-015"), "ordered REC-001"),
        (lambda payload: payload["recommendations"][5].update(priority="P0"), "priority sequence"),
        (lambda payload: payload["recommendations"][0].update(dependencies=["REC-001"]), "self"),
        (lambda payload: payload["recommendations"][0].update(supersedes=["REC-001"]), "self"),
        (lambda payload: payload["recommendations"][0].update(dependencies=["REC-999"]), "unknown"),
        (lambda payload: payload["recommendations"][0].update(supersedes=["REC-999"]), "unknown"),
    ],
)
def test_loader_rejects_invalid_recommendation_relationships(
    mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    payload = _valid_registry_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        recommendations._parse_recommendation_registry(payload)


def test_loader_accepts_an_allowed_status_transition() -> None:
    payload = _valid_registry_payload()
    payload["recommendations"][0]["status"] = "accepted"

    parsed = recommendations._parse_recommendation_registry(payload)

    assert parsed[0].status == "accepted"


def _valid_registry_payload() -> dict[str, Any]:
    records = []
    for recommendation_id, title, priority, status in EXPECTED_RECOMMENDATIONS:
        records.append(
            {
                "id": recommendation_id,
                "title": title,
                "priority": priority,
                "status": status,
                "rationale": f"Rationale for {recommendation_id}.",
                "targets": ["evaluation"],
                "supersedes": [],
                "dependencies": [],
                "research_refs": [],
                "repo_refs": ["tests/test_recommendation_registry.py"],
                "acceptance_tests": ["python -m pytest tests/test_recommendation_registry.py -q"],
                "implementation_refs": [],
            }
        )
    return {"registry_version": "17case-v5", "recommendations": records}
