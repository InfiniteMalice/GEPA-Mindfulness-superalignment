"""Behavioral contracts for V5 research metadata and traceability."""

from __future__ import annotations

import importlib.resources
import re
from dataclasses import FrozenInstanceError
from os.path import relpath
from pathlib import Path
from typing import Any, Callable

import pytest

from evaluation import recommendations

REPOSITORY_ROOT = Path(__file__).parents[1]
TRACEABILITY_PATH = REPOSITORY_ROOT / "docs" / "recommendations" / "RESEARCH_TRACEABILITY.md"

EXPECTED_REFERENCES = (
    (
        "REF-HEART",
        "2609.01736",
        "Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives",
    ),
    (
        "REF-PEARL",
        "2609.02216",
        "Path-Entity Aligned Relational Learning with Contextual Subgraphs for Inductive "
        "Knowledge Graph Completion",
    ),
    (
        "REF-SEGOS",
        "2609.08228",
        "Self-Evolving Graph-of-Skills for Skill Library at Scale",
    ),
    (
        "REF-COEVOLVE",
        "2609.09134",
        "Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up "
        "Where Imitation Fails",
    ),
    (
        "REF-CONSISTENCY",
        "2609.08832",
        "Closing the Consistency Gap: Self-Evolving Agents That Learn to Stay on Course",
    ),
    (
        "REF-DSR",
        "2609.05824",
        "Beyond Top-k Skill Retrieval: Diversity-Aware Skill Routing for LLM Agents",
    ),
    (
        "REF-EDGEMEM",
        "2609.05553",
        "EdgeMem: LLM-Free Agent Memory Construction and Retrieval via Evidence-Preserving "
        "Multi-Anchor Hypergraph",
    ),
    (
        "REF-GRAPHMEM",
        "2609.08599",
        "Graph-Based Personalized Memory for LLM Agents: Representation, Evolution, Retrieval, "
        "and Evaluation",
    ),
    (
        "REF-SHEAVES",
        "2609.09056",
        "Time-Varying Data as Sheaves: an Invitation to Narratives",
    ),
    (
        "REF-HERO",
        "2609.08189",
        "HeRo: History-Aware Routing for Efficient LLM Inference",
    ),
    (
        "REF-SAE",
        "2609.09113",
        "SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?",
    ),
    (
        "REF-BIOMETRIC-MEM",
        "2609.08558",
        "Personalizing LLM Agent Memory Using Biometrics",
    ),
    (
        "REF-HOH",
        "2609.01481",
        "Harness-of-Harness: Multi-Day Autonomous Software Development with Continual "
        "Improvement",
    ),
    (
        "REF-AGENTSCOPE",
        "2609.02371",
        "Diagnosing with Insights: Structured Analysis of Agent Failures via Behavioral "
        "Abstractions",
    ),
    (
        "REF-REPOTOSKILL",
        "2609.02749",
        "Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills",
    ),
    (
        "REF-SKILLGLOW",
        "2609.02217",
        "SkillGLoW: Procedural-Family Skill Consolidation for Self-Improving Agents on "
        "Long-Horizon Task Streams",
    ),
    (
        "REF-MASKILLS",
        "2609.02094",
        "MASkills: Continual Skills Optimization for Multi-Agent LLM Systems",
    ),
    (
        "REF-WMLLM",
        "2609.01608",
        "WMLLM: Self-Evolving Optimization Agents via Predict-Then-Act World Modeling",
    ),
    (
        "REF-DWM",
        "2609.02885",
        "Discriminative World Models for Web Agents",
    ),
    (
        "REF-LEXICAL-PERTURB",
        "2608.22140",
        "Lexical Perturbations Disrupt LLM Reasoning: An Empirical Study of Attention Diversion",
    ),
    (
        "REF-TOKENIZER-BETRAYAL",
        "2601.14658",
        "Say Anything but This: When Tokenizer Betrays Reasoning in LLMs",
    ),
)

EXPECTED_RECOMMENDATION_LINKS = {
    "REF-HEART": ("REC-001", "REC-008"),
    "REF-PEARL": ("REC-011",),
    "REF-SEGOS": ("REC-009",),
    "REF-COEVOLVE": ("REC-010",),
    "REF-CONSISTENCY": ("REC-005",),
    "REF-DSR": ("REC-009",),
    "REF-EDGEMEM": ("REC-006",),
    "REF-GRAPHMEM": ("REC-006",),
    "REF-SHEAVES": ("REC-002",),
    "REF-HERO": ("REC-002",),
    "REF-SAE": ("REC-014",),
    "REF-BIOMETRIC-MEM": ("REC-008",),
    "REF-HOH": ("REC-010",),
    "REF-AGENTSCOPE": ("REC-007", "REC-013"),
    "REF-REPOTOSKILL": ("REC-009",),
    "REF-SKILLGLOW": ("REC-009",),
    "REF-MASKILLS": ("REC-012",),
    "REF-WMLLM": ("REC-002",),
    "REF-DWM": ("REC-002",),
    "REF-LEXICAL-PERTURB": ("REC-005",),
    "REF-TOKENIZER-BETRAYAL": ("REC-005",),
}


def test_bundled_registry_has_every_requested_reference_once() -> None:
    loaded = recommendations.load_research_reference_registry()

    assert (
        tuple((item.reference_id, item.arxiv_id, item.supplied_title) for item in loaded)
        == EXPECTED_REFERENCES
    )
    assert len({item.reference_id for item in loaded}) == len(loaded)
    assert len({item.arxiv_id for item in loaded}) == len(loaded)
    assert all(item.metadata_status == "resolved" for item in loaded)


def test_resolved_metadata_is_complete_and_uses_canonical_arxiv_urls() -> None:
    for reference in recommendations.load_research_reference_registry():
        assert reference.title
        assert reference.authors
        assert reference.year == 2026
        assert reference.doi == f"10.48550/arXiv.{reference.arxiv_id}"
        assert reference.canonical_url == f"https://arxiv.org/abs/{reference.arxiv_id}"
        assert isinstance(reference.authors, tuple)


def test_reference_records_are_frozen() -> None:
    reference = recommendations.load_research_reference_registry()[0]

    with pytest.raises(FrozenInstanceError):
        reference.metadata_status = "unresolved"


def test_registry_loads_as_a_package_resource_outside_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    loaded = recommendations.load_research_reference_registry()
    package_files = importlib.resources.files("docs.recommendations")

    assert len(loaded) == len(EXPECTED_REFERENCES)
    assert package_files.joinpath("references.yaml").is_file()


def test_recommendation_links_resolve_and_match_both_registries() -> None:
    references = recommendations.load_research_reference_registry()
    registered = recommendations.load_recommendation_registry()
    known_recommendations = {item.recommendation_id for item in registered}
    reverse_links = {item.recommendation_id: set(item.research_refs) for item in registered}

    for reference in references:
        assert reference.recommendation_ids == EXPECTED_RECOMMENDATION_LINKS[reference.reference_id]
        assert set(reference.recommendation_ids) <= known_recommendations
        for recommendation_id in reference.recommendation_ids:
            assert reference.reference_id in reverse_links[recommendation_id]


def test_every_local_repository_note_exists() -> None:
    for reference in recommendations.load_research_reference_registry():
        assert reference.local_repo_notes
        for note in reference.local_repo_notes:
            assert (REPOSITORY_ROOT / note).is_file(), (reference.reference_id, note)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"registry_version": "17case-v5"}, "research reference registry fields"),
        (
            {
                "registry_version": "17case-v5",
                "references": [],
                "unexpected": True,
            },
            "unknown fields",
        ),
        ({"registry_version": "v5", "references": []}, "registry_version"),
    ],
)
def test_loader_rejects_invalid_document_roots(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        recommendations._parse_research_reference_registry(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["references"][0].pop("title"), "REF-HEART fields"),
        (lambda payload: payload["references"][0].update(extra=True), "unknown fields"),
        (
            lambda payload: payload["references"][0].update(metadata_status="pending"),
            "metadata_status",
        ),
        (lambda payload: payload["references"][0].update(authors=[]), "authors"),
        (lambda payload: payload["references"][0].update(year=None), "year"),
        (
            lambda payload: payload["references"][0].update(canonical_url="https://example.com"),
            "canonical_url",
        ),
        (
            lambda payload: payload["references"][0].update(recommendation_ids=[]),
            "recommendation_ids",
        ),
        (
            lambda payload: payload["references"][0].update(recommendation_ids=["REC-999"]),
            "unknown IDs",
        ),
        (
            lambda payload: payload["references"][0].update(local_repo_notes=[]),
            "local_repo_notes",
        ),
    ],
)
def test_loader_rejects_invalid_reference_fields(
    mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    payload = _valid_reference_registry_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        recommendations._parse_research_reference_registry(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["references"][1].update(reference_id="REF-HEART"),
            "duplicate IDs",
        ),
        (
            lambda payload: payload["references"][1].update(
                arxiv_id="2609.01736",
                canonical_url="https://arxiv.org/abs/2609.01736",
            ),
            "duplicate arXiv IDs",
        ),
        (
            lambda payload: payload["references"][0].update(
                arxiv_id="2609.99999",
                canonical_url="https://arxiv.org/abs/2609.99999",
            ),
            "ordered supplied arXiv sequence",
        ),
        (
            lambda payload: payload["references"][0].update(reference_id="REF-UNKNOWN"),
            "ordered requested sequence",
        ),
    ],
)
def test_loader_rejects_invalid_reference_identity(
    mutation: Callable[[dict[str, Any]], Any], message: str
) -> None:
    payload = _valid_reference_registry_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        recommendations._parse_research_reference_registry(payload)


@pytest.mark.parametrize(
    ("field", "unsupported_value"),
    [
        ("title", "Unverified canonical title"),
        ("authors", ["Unverified Author"]),
        ("year", 2026),
        ("doi", "10.0000/unverified"),
        ("venue_status", "Unverified venue"),
    ],
)
def test_unresolved_metadata_rejects_unsupported_bibliographic_fields(
    field: str, unsupported_value: Any
) -> None:
    payload = _valid_reference_registry_payload()
    _mark_unresolved(payload["references"][0])
    payload["references"][0][field] = unsupported_value

    with pytest.raises(ValueError, match="unresolved metadata"):
        recommendations._parse_research_reference_registry(payload)


def test_loader_accepts_unresolved_metadata_without_a_source_claim() -> None:
    payload = _valid_reference_registry_payload()
    _mark_unresolved(payload["references"][0])

    parsed = recommendations._parse_research_reference_registry(payload)

    assert parsed[0].metadata_status == "unresolved"
    assert parsed[0].supplied_title == EXPECTED_REFERENCES[0][2]
    assert parsed[0].title is None
    assert parsed[0].authors == ()
    assert parsed[0].source_demonstrates.startswith("No source claim has been verified")


def test_traceability_reader_mirrors_registry_identity_and_required_labels() -> None:
    reader = TRACEABILITY_PATH.read_text(encoding="utf-8")

    for reference in recommendations.load_research_reference_registry():
        section = _reference_section(reader, reference.reference_id)
        displayed_title = reference.title or reference.supplied_title
        assert f'<a id="{reference.reference_id.lower()}"></a>' in reader
        assert f"## {reference.reference_id} — {displayed_title}" in section
        assert f"- Metadata status: `{reference.metadata_status}`" in section
        assert f"- Supplied title: {reference.supplied_title}" in section
        assert f"- Authors: {', '.join(reference.authors)}" in section
        assert f"- Year: {reference.year}" in section
        assert f"- DOI: `{reference.doi}`" in section
        assert f"- arXiv: [`{reference.arxiv_id}`]({reference.canonical_url})" in section
        for recommendation_id in reference.recommendation_ids:
            assert f"`{recommendation_id}`" in section
        for note in reference.local_repo_notes:
            link_target = Path(relpath(REPOSITORY_ROOT / note, TRACEABILITY_PATH.parent))
            assert f"]({link_target.as_posix()})" in section
        assert "**Source demonstrates:**" in section
        assert "**Repository inference:**" in section
        assert "**Maturity:**" in section


def test_traceability_reader_relative_links_resolve() -> None:
    reader = TRACEABILITY_PATH.read_text(encoding="utf-8")

    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", reader):
        if target.startswith(("#", "https://")):
            continue
        path_target = target.partition("#")[0]
        assert (TRACEABILITY_PATH.parent / path_target).resolve().is_file(), target


def test_traceability_connections_do_not_claim_the_sources_prove_the_architecture() -> None:
    for reference in recommendations.load_research_reference_registry():
        connection = " ".join(
            (
                reference.source_demonstrates,
                reference.repository_inference,
                reference.maturity,
            )
        )
        assert re.search(r"\b(proves?|proven)\b", connection, flags=re.IGNORECASE) is None


def _valid_reference_registry_payload() -> dict[str, Any]:
    records = []
    for reference_id, arxiv_id, supplied_title in EXPECTED_REFERENCES:
        records.append(
            {
                "reference_id": reference_id,
                "metadata_status": "resolved",
                "supplied_title": supplied_title,
                "title": f"Verified title for {reference_id}",
                "authors": ["Verified Author"],
                "year": 2026,
                "arxiv_id": arxiv_id,
                "doi": f"10.48550/arXiv.{arxiv_id}",
                "venue_status": None,
                "canonical_url": f"https://arxiv.org/abs/{arxiv_id}",
                "recommendation_ids": list(EXPECTED_RECOMMENDATION_LINKS[reference_id]),
                "source_demonstrates": "The source reports a related mechanism.",
                "repository_inference": "The result motivates a bounded repository decision.",
                "maturity": "Research inspiration; repository behavior is independently tested.",
                "local_repo_notes": ["history/2026-09-10-gepa-v5-unified-architecture-design.md"],
            }
        )
    return {"registry_version": "17case-v5", "references": records}


def _mark_unresolved(record: dict[str, Any]) -> None:
    record.update(
        metadata_status="unresolved",
        title=None,
        authors=[],
        year=None,
        doi=None,
        venue_status=None,
        source_demonstrates=(
            "No source claim has been verified because official metadata was unavailable."
        ),
    )


def _reference_section(reader: str, reference_id: str) -> str:
    match = re.search(
        rf"^## {re.escape(reference_id)}\b.*?(?=^## REF-|\Z)",
        reader,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing traceability section for {reference_id}"
    return match.group(0)
