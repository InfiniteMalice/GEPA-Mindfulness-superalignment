"""Behavioral contracts for V5 research metadata and traceability."""

from __future__ import annotations

import importlib.resources
import re
from dataclasses import FrozenInstanceError
from os.path import relpath
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from evaluation import recommendations

REPOSITORY_ROOT = Path(__file__).parents[1]
TRACEABILITY_PATH = REPOSITORY_ROOT / "docs" / "recommendations" / "RESEARCH_TRACEABILITY.md"


def test_quantizer_publication_status_matches_official_arxiv_comments():
    """Retain the acceptance status reported by arXiv:2609.25014's Comments field."""
    references = recommendations.load_research_reference_registry()
    reference = next(r for r in references if r.reference_id == "REF-4BIT-QUANTIZERS")
    assert reference.venue_status == "Accepted for publication at SBSeg 2026"


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
    ("REF-SOT", "2609.16055", "State of Thought Enables Endogenous Reasoning"),
    ("REF-KALMAN", None, "A New Approach to Linear Filtering and Prediction Problems"),
    (
        "REF-FTA",
        "2609.35732",
        "Failure-Transparent Agents: Benchmarking Post-Failure Reporting "
        "in Tool-Using Language Models",
    ),
    (
        "REF-PINNFORGE",
        "2609.23023",
        "PINNsForge",
    ),
    (
        "REF-C3-JEPA",
        "2609.30214",
        "C3-JEPA",
    ),
    (
        "REF-AI-NEUROSCIENTIST",
        "2609.25254",
        "AI Neuroscientist",
    ),
    ("REF-GRUET", "2609.24831", "GRUET"),
    ("REF-DUAL-FRONTIER", "2609.26293", "Dual-Frontier"),
    ("REF-DEEPO", "2609.28570", "DEEPO"),
    ("REF-CI", None, "Julier & Uhlmann 1997 — Covariance Intersection"),
    ("REF-UNANIMITY", "2609.26145", "Unanimity Without Persuasion"),
    ("REF-WSQEM", "2609.25555", "Weakly Supervised Quantum Error Mitigation"),
    ("REF-MEMCALIB", "2609.24259", "MemCalib"),
    ("REF-JITMEM", "2609.27334", "JITMEM"),
    ("REF-COMPKV", "2609.26300", "CompKV"),
    ("REF-QWEN-PLANNER", "2609.29892", "Qwen-Planner-Agent"),
    ("REF-SHARE-BORNE", "2609.35576", "Share-Borne AI Virus"),
    ("REF-A2M", "2609.26761", "A2M/MCP semantic hijacking"),
    (
        "REF-EVOFLINT",
        "2609.00487",
        "EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities",
    ),
    (
        "REF-COMM-BOTTLENECK",
        "2609.21509",
        "The Communication Bottleneck: A Round-Trip Study of Tree-Structured Expression "
        "Serialization in Language Models",
    ),
    (
        "REF-LOGICTRACK",
        "2609.21492",
        "LogicTrack: Auditing Reasoning Trajectories of Large Language Models with Formal "
        "Logic Solvers",
    ),
    (
        "REF-LATENT-LANGUAGE-GAP",
        "2609.21662",
        "When Steering Fails in Latent Reasoning: A Latent-to-Language Transition Gap",
    ),
    ("REF-CDR", "2609.22239", "Coverage-Directed Revision"),
    ("REF-TTSE", "2609.24289", "TTSE"),
    ("REF-JEV-MEM", "2609.23986", "Jev-Mem"),
    ("REF-CLM", None, "Contrastive Language Models"),
    ("REF-TOOLLERY", "2609.22218", "Toollery"),
    ("REF-SEEK", "2609.29803", "SEEK"),
    ("REF-LADDER", "2609.24346", "LADDER"),
    ("REF-REASONING-TOPOLOGY", "2609.24710", "Reasoning Topology Matters"),
    ("REF-ARISE", "2609.35532", "ARISE"),
    ("REF-TABPFN-35", "2609.17895", "TabPFN-3.5"),
    ("REF-VGCOMPILER", "2609.22327", "VGCompiler"),
    ("REF-PHYSICAL-LANGUAGES", "2609.23381", "Discovering Physical Representation Languages"),
    (
        "REF-GENERALIZED-TAMP",
        "2609.30233",
        "Coding Agents for Generalized Task and Motion Planning Problems",
    ),
    ("REF-CHART", "2609.22247", "CHART / Harness-Rotation"),
    ("REF-SPECTRAL-GROKKING", "2609.26679", "Spectral Theory of Grokking"),
    ("REF-LOW-BIT-OPD", "2609.26708", "On-Policy Distillation for Low-Bit Reasoning"),
    ("REF-P-TTT", "2609.35109", "P-TTT"),
    ("REF-MECHBENCH", "2609.35515", "MechBench"),
    ("REF-QUANTUM-THINK", "2609.23016", "Watching Quantum Models Think"),
    ("REF-CAT-SEARCH", "2609.25575", "Compute-Aligned Training for Search"),
    ("REF-PAWS", "2609.28547", "PAWS"),
    ("REF-4BIT-QUANTIZERS", "2609.25014", "Not All 4-bit Quantizers Are Equal"),
    ("REF-SRHARNESS", "2609.35501", "SRHarness"),
    ("REF-DOATLAS-2", "2609.35107", "DoAtlas-2"),
    ("REF-ABGAZE", "2609.35296", "AbGaze"),
)

EXPECTED_RECOMMENDATION_LINKS = {
    "REF-SRHARNESS": ("REC-011",),
    "REF-DOATLAS-2": ("REC-011",),
    "REF-ABGAZE": ("REC-011",),
    "REF-4BIT-QUANTIZERS": ("REC-010",),
    "REF-PAWS": ("REC-010",),
    "REF-CAT-SEARCH": ("REC-010",),
    "REF-P-TTT": ("REC-010", "REC-014"),
    "REF-MECHBENCH": ("REC-010", "REC-014"),
    "REF-QUANTUM-THINK": ("REC-014",),
    "REF-CHART": ("REC-010",),
    "REF-SPECTRAL-GROKKING": ("REC-010",),
    "REF-LOW-BIT-OPD": ("REC-010",),
    "REF-TABPFN-35": ("REC-005", "REC-010"),
    "REF-VGCOMPILER": ("REC-005", "REC-010"),
    "REF-PHYSICAL-LANGUAGES": ("REC-005", "REC-010", "REC-011"),
    "REF-GENERALIZED-TAMP": ("REC-005", "REC-010"),
    "REF-ARISE": ("REC-009",),
    "REF-JEV-MEM": ("REC-008",),
    "REF-CLM": ("REC-008", "REC-010"),
    "REF-TOOLLERY": ("REC-008",),
    "REF-SEEK": ("REC-007", "REC-008", "REC-009"),
    "REF-LADDER": ("REC-008",),
    "REF-REASONING-TOPOLOGY": ("REC-008",),
    "REF-EVOFLINT": ("REC-016",),
    "REF-COMM-BOTTLENECK": ("REC-017",),
    "REF-LOGICTRACK": ("REC-018",),
    "REF-LATENT-LANGUAGE-GAP": ("REC-019",),
    "REF-MEMCALIB": ("REC-002", "REC-015"),
    "REF-JITMEM": ("REC-002", "REC-015"),
    "REF-CDR": ("REC-014", "REC-015"),
    "REF-TTSE": ("REC-015",),
    "REF-COMPKV": ("REC-002",),
    "REF-QWEN-PLANNER": ("REC-002", "REC-005", "REC-007", "REC-009", "REC-010"),
    "REF-SHARE-BORNE": ("REC-002",),
    "REF-A2M": ("REC-002",),
    "REF-CI": ("REC-002",),
    "REF-UNANIMITY": ("REC-002",),
    "REF-WSQEM": ("REC-002",),
    "REF-GRUET": ("REC-002", "REC-010"),
    "REF-DUAL-FRONTIER": ("REC-002", "REC-011"),
    "REF-DEEPO": ("REC-002", "REC-010"),
    "REF-FTA": ("REC-002", "REC-007", "REC-010"),
    "REF-PINNFORGE": ("REC-002", "REC-007"),
    "REF-C3-JEPA": ("REC-002", "REC-010", "REC-014"),
    "REF-AI-NEUROSCIENTIST": ("REC-002",),
    "REF-KALMAN": ("REC-002", "REC-010"),
    "REF-SOT": ("REC-010", "REC-015"),
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

PUBLIC_REGISTRY_LOADERS = (
    "load_recommendation_registry",
    "load_research_reference_registry",
)


def test_bundled_registry_has_every_requested_reference_once() -> None:
    loaded = recommendations.load_research_reference_registry()

    assert (
        tuple((item.reference_id, item.arxiv_id, item.supplied_title) for item in loaded)
        == EXPECTED_REFERENCES
    )
    assert len({item.reference_id for item in loaded}) == len(loaded)
    arxiv_ids = [item.arxiv_id for item in loaded if item.arxiv_id is not None]
    assert len(set(arxiv_ids)) == len(arxiv_ids)
    assert all(item.metadata_status == "resolved" for item in loaded)


def test_resolved_metadata_is_complete_and_uses_canonical_arxiv_urls() -> None:
    for reference in recommendations.load_research_reference_registry():
        assert reference.title
        assert reference.authors
        if reference.reference_id == "REF-CLM":
            assert reference.year == 2026
            assert reference.arxiv_id is None
            assert reference.doi is None
            assert reference.canonical_url == "https://github.com/Contrastive-LM/CLM"
            continue
        if reference.reference_id == "REF-CI":
            assert reference.year == 1997
            assert reference.arxiv_id is None
            assert reference.doi == "10.1109/ACC.1997.609105"
            assert reference.canonical_url == "https://doi.org/10.1109/ACC.1997.609105"
            continue
        if reference.reference_id == "REF-KALMAN":
            assert reference.year == 1960
            assert reference.arxiv_id is None
            assert reference.doi == "10.1115/1.3662552"
            assert reference.canonical_url == "https://doi.org/10.1115/1.3662552"
            continue
        assert reference.year == 2026
        assert reference.doi == f"10.48550/arXiv.{reference.arxiv_id}"
        assert reference.canonical_url == f"https://arxiv.org/abs/{reference.arxiv_id}"
        assert isinstance(reference.authors, tuple)


def test_reference_records_are_frozen() -> None:
    reference = recommendations.load_research_reference_registry()[0]

    with pytest.raises(FrozenInstanceError):
        reference.metadata_status = "unresolved"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"doi": None}, "arxiv_id or doi"),
        ({"canonical_url": "https://example.com/paper"}, "canonical_url"),
        ({"arxiv_id": ""}, "arxiv_id"),
    ],
)
def test_doi_only_source_rejects_missing_identity_and_wrong_url(changes, message) -> None:
    payload = _valid_reference_registry_payload()
    source = next(item for item in payload["references"] if item["reference_id"] == "REF-KALMAN")
    source.update(changes)
    with pytest.raises(ValueError, match=message):
        recommendations._parse_research_reference_registry(payload)


def test_clm_project_identity_requires_its_exact_primary_url() -> None:
    _, payload = _authored_registry_payloads()
    source = next(item for item in payload["references"] if item["reference_id"] == "REF-CLM")
    assert source["arxiv_id"] is None and source["doi"] is None
    recommendations._parse_research_reference_registry(payload)
    source["canonical_url"] = "https://example.com/clm"
    with pytest.raises(ValueError, match="canonical_url"):
        recommendations._parse_research_reference_registry(payload)


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


@pytest.mark.parametrize("loader_name", PUBLIC_REGISTRY_LOADERS)
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda recommendations_payload, _: recommendations_payload["recommendations"][0][
                "research_refs"
            ].append("REF-UNKNOWN"),
            "recommendation registry research_refs reference unknown IDs",
        ),
        (
            lambda _, references_payload: references_payload["references"][0].update(
                recommendation_ids=["REC-999"]
            ),
            "research reference registry recommendation_ids reference unknown IDs",
        ),
        (
            lambda _, references_payload: references_payload["references"][0][
                "recommendation_ids"
            ].remove("REC-001"),
            "recommendation registry edges missing from research reference registry",
        ),
        (
            lambda recommendations_payload, _: recommendations_payload["recommendations"][0][
                "research_refs"
            ].append("REF-PEARL"),
            "recommendation registry edges missing from research reference registry",
        ),
        (
            lambda recommendations_payload, _: recommendations_payload["recommendations"][0][
                "research_refs"
            ].remove("REF-HEART"),
            "research reference registry edges missing from recommendation registry",
        ),
        (
            lambda _, references_payload: references_payload["references"][1][
                "recommendation_ids"
            ].append("REC-001"),
            "research reference registry edges missing from recommendation registry",
        ),
    ],
    ids=(
        "unknown-reference-from-recommendation",
        "unknown-recommendation-from-reference",
        "missing-reference-side-reciprocal",
        "extra-recommendation-side-edge",
        "missing-recommendation-side-reciprocal",
        "extra-reference-side-edge",
    ),
)
def test_public_loaders_reject_invalid_cross_registry_edges(
    loader_name: str,
    mutation: Callable[[dict[str, Any], dict[str, Any]], Any],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recommendations_payload, references_payload = _authored_registry_payloads()
    mutation(recommendations_payload, references_payload)
    _redirect_registry_resources(
        tmp_path,
        monkeypatch,
        recommendations_payload,
        references_payload,
    )

    with pytest.raises(ValueError, match=message):
        getattr(recommendations, loader_name)()


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


def test_traceability_reader_has_exact_unique_reference_inventory() -> None:
    reader = TRACEABILITY_PATH.read_text(encoding="utf-8")
    expected_reference_ids = tuple(item[0] for item in EXPECTED_REFERENCES)
    anchors = tuple(
        match.group("anchor")
        for match in re.finditer(
            r'^<a id="(?P<anchor>ref-[a-z0-9-]+)"></a>$',
            reader,
            flags=re.MULTILINE,
        )
    )
    headings = tuple(
        match.group("reference_id")
        for match in re.finditer(
            r"^## (?P<reference_id>REF-[A-Z0-9-]+) — .+$",
            reader,
            flags=re.MULTILINE,
        )
    )

    assert anchors == tuple(reference_id.lower() for reference_id in expected_reference_ids)
    assert headings == expected_reference_ids
    assert len(anchors) == len(set(anchors))
    assert len(headings) == len(set(headings))


def test_traceability_reader_exactly_mirrors_registry_fields() -> None:
    reader = TRACEABILITY_PATH.read_text(encoding="utf-8")

    for reference in recommendations.load_research_reference_registry():
        section = _reference_section(reader, reference.reference_id)
        displayed_title = reference.title or reference.supplied_title
        expected_venue = (
            reference.venue_status
            if reference.venue_status is not None
            else "Not supplied by official arXiv metadata."
        )

        assert _reference_heading(section, reference.reference_id) == displayed_title
        assert _markdown_list_value(section, "Metadata status") == (
            f"`{reference.metadata_status}`"
        )
        assert _markdown_list_value(section, "Supplied title") == reference.supplied_title
        assert _markdown_list_value(section, "Authors") == ", ".join(reference.authors)
        assert _markdown_list_value(section, "Year") == str(reference.year)
        assert _markdown_list_value(section, "DOI") == f"`{reference.doi}`"
        if reference.arxiv_id is None:
            assert _markdown_list_value(section, "arXiv") == "Not applicable."
            assert reference.canonical_url in section
        else:
            assert _markdown_list_value(section, "arXiv") == (
                f"[`{reference.arxiv_id}`]({reference.canonical_url})"
            )
        assert _markdown_list_value(section, "Venue/status") == expected_venue
        assert _markdown_list_value(section, "Recommendations influenced") == ", ".join(
            f"`{recommendation_id}`" for recommendation_id in reference.recommendation_ids
        )
        assert _markdown_paragraph_value(section, "Source demonstrates") == (
            reference.source_demonstrates
        )
        assert _markdown_paragraph_value(section, "Repository inference") == (
            reference.repository_inference
        )
        assert _markdown_paragraph_value(section, "Maturity") == reference.maturity
        for note in reference.local_repo_notes:
            link_target = Path(relpath(REPOSITORY_ROOT / note, TRACEABILITY_PATH.parent))
            assert f"]({link_target.as_posix()})" in section


def test_traceability_reader_relative_links_resolve() -> None:
    reader = TRACEABILITY_PATH.read_text(encoding="utf-8")

    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", reader):
        if target.startswith(("#", "https://")):
            continue
        path_target = target.partition("#")[0]
        assert (TRACEABILITY_PATH.parent / path_target).resolve().is_file(), target


def test_traceability_reader_avoids_prohibited_proof_language() -> None:
    reader = TRACEABILITY_PATH.read_text(encoding="utf-8")

    assert (
        re.search(
            r"\b(proofs?|proves?|proven|guarantees?|guaranteed|confirms?|confirmed)\b",
            reader,
            flags=re.IGNORECASE,
        )
        is None
    )


def _valid_reference_registry_payload() -> dict[str, Any]:
    records = []
    for reference_id, arxiv_id, supplied_title in EXPECTED_REFERENCES:
        doi = (
            f"10.48550/arXiv.{arxiv_id}"
            if arxiv_id
            else "10.1109/ACC.1997.609105" if reference_id == "REF-CI" else "10.1115/1.3662552"
        )
        records.append(
            {
                "reference_id": reference_id,
                "metadata_status": "resolved",
                "supplied_title": supplied_title,
                "title": f"Verified title for {reference_id}",
                "authors": ["Verified Author"],
                "year": 2026,
                "arxiv_id": arxiv_id,
                "doi": doi,
                "venue_status": None,
                "canonical_url": (
                    f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://doi.org/{doi}"
                ),
                "recommendation_ids": list(EXPECTED_RECOMMENDATION_LINKS[reference_id]),
                "source_demonstrates": "The source reports a related mechanism.",
                "repository_inference": "The result motivates a bounded repository decision.",
                "maturity": "Research inspiration; repository behavior is independently tested.",
                "local_repo_notes": ["history/2026-09-10-gepa-v5-unified-architecture-design.md"],
            }
        )
    return {"registry_version": "17case-v5", "references": records}


def _authored_registry_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    recommendation_path = REPOSITORY_ROOT / "docs" / "recommendations" / "registry.yaml"
    reference_path = REPOSITORY_ROOT / "docs" / "recommendations" / "references.yaml"
    return (
        yaml.safe_load(recommendation_path.read_text(encoding="utf-8")),
        yaml.safe_load(reference_path.read_text(encoding="utf-8")),
    )


def _redirect_registry_resources(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    recommendations_payload: dict[str, Any],
    references_payload: dict[str, Any],
) -> None:
    (directory / "registry.yaml").write_text(
        yaml.safe_dump(recommendations_payload, sort_keys=False),
        encoding="utf-8",
    )
    (directory / "references.yaml").write_text(
        yaml.safe_dump(references_payload, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(recommendations.resources, "files", lambda _: directory)


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


def _reference_heading(section: str, reference_id: str) -> str:
    matches = re.findall(
        rf"^## {re.escape(reference_id)} — (?P<title>.+)$",
        section,
        flags=re.MULTILINE,
    )
    assert len(matches) == 1, f"expected one heading for {reference_id}"
    return matches[0]


def _markdown_list_value(section: str, label: str) -> str:
    return _wrapped_markdown_value(section, f"- {label}: ")


def _markdown_paragraph_value(section: str, label: str) -> str:
    return _wrapped_markdown_value(section, f"**{label}:** ")


def _wrapped_markdown_value(section: str, prefix: str) -> str:
    lines = section.splitlines()
    indexes = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    assert len(indexes) == 1, f"expected one Markdown field with prefix {prefix!r}"
    first_index = indexes[0]
    parts = [lines[first_index][len(prefix) :]]
    for line in lines[first_index + 1 :]:
        if not line:
            break
        if line.startswith(("- ", "**", "## ", "<a ")):
            break
        parts.append(line)
    return " ".join(parts)
